"""R033 — Cache per-image quantities for COCO val 2017 × Mask2Former-COCO-Panoptic.

For each image i in COCO val 2017 (n=5000), cache:
- per_image_miou[i]: mIoU of predicted vs ground-truth semantic segmentation (built from panoptic).
- per_image_pixacc[i]: per-pixel accuracy (alternative loss formulation; potentially lower-variance).
- per_image_loss[i] = 1 - per_image_miou[i]
- per_image_loss_pixacc[i] = 1 - per_image_pixacc[i]
- per_image_g_softmax[i]: mean per-pixel softmax-max confidence (primary acceptance score)
- per_image_g_entropy[i]: 1 - mean(entropy)/log(N) (alternative)

Why COCO (5000 imgs) over ADE20K (2000): Ours's variance-adaptive payoff
needs n · π_min ≫ ~150. ADE20K at n=2000, π_min=0.10 → 200 (borderline failure).
COCO at n_cal=4000, π_min=0.10 → 400 — should let Ours dominate Hoeffding-CRC
*if* the continuous mIoU loss has variance structure that benefits Bernstein.

Model: facebook/mask2former-swin-base-coco-panoptic (~400 MB), 133-class panoptic output
       (80 things + 53 stuff merged categories per COCO panoptic spec).

Compute: ~30 min on 1 H100.

Output:
  $SCORC_DATA_DIR/coco_data/val_mask2former_coco_miou.npy        (5000,) float32
  $SCORC_DATA_DIR/coco_data/val_mask2former_coco_loss.npy        (5000,) float32 = 1 - mIoU
  $SCORC_DATA_DIR/coco_data/val_mask2former_coco_pixacc.npy      (5000,) float32
  $SCORC_DATA_DIR/coco_data/val_mask2former_coco_loss_pixacc.npy (5000,) float32
  $SCORC_DATA_DIR/coco_data/val_mask2former_coco_g_softmax.npy   (5000,) float32
  $SCORC_DATA_DIR/coco_data/val_mask2former_coco_g_entropy.npy   (5000,) float32

"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import os
# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

COCO_ROOT = Path(f"{DATA_ROOT}/coco_data")
VAL_IMG_DIR = COCO_ROOT / "val2017"
PANOPTIC_DIR = COCO_ROOT / "panoptic_anns/annotations/panoptic_val2017"
PANOPTIC_JSON = COCO_ROOT / "panoptic_anns/annotations/panoptic_val2017.json"


def per_image_miou(pred: np.ndarray, gt: np.ndarray, ignore_value: int = -1) -> float:
    """Per-image mIoU over classes present in pred ∪ gt (excluding ignore)."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    valid = gt != ignore_value
    if not valid.any():
        return 0.0
    pred_v = pred[valid]
    gt_v = gt[valid]
    classes_present = np.union1d(np.unique(pred_v), np.unique(gt_v))
    if -1 in classes_present:
        classes_present = classes_present[classes_present != -1]
    ious = []
    for c in classes_present:
        if c < 0:
            continue
        inter = ((pred_v == c) & (gt_v == c)).sum()
        union = ((pred_v == c) | (gt_v == c)).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def per_image_pixel_accuracy(pred: np.ndarray, gt: np.ndarray, ignore_value: int = -1) -> float:
    valid = gt != ignore_value
    if not valid.any():
        return 0.0
    return float((pred[valid] == gt[valid]).mean())


def md5_of_path(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon_name(s: str) -> str:
    """Normalize a class name for fuzzy matching: lowercase + strip non-alnum."""
    import re
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def build_coco_cat_id_to_m2f_index(m2f_id2label: dict, coco_categories: list) -> dict:
    """Build mapping COCO category_id -> Mask2Former label index.

    Mask2Former-COCO-Panoptic has 133 classes (80 things + 53 stuff merged). COCO
    panoptic category_ids are sparse (1..200 with gaps); model's id2label is dense (0..132).
    Match by exact name first, then canonicalised name as fallback.
    Raise on any missing category or non-permutation mapping to catch silent bugs.
    """
    id2label = {int(k): v for k, v in m2f_id2label.items()}
    exact = {v: k for k, v in id2label.items()}
    canon = {_canon_name(v): k for k, v in id2label.items()}

    coco_id_to_m2f = {}
    missing = []
    for cat in coco_categories:
        name = cat["name"]
        if name in exact:
            coco_id_to_m2f[cat["id"]] = exact[name]
        elif _canon_name(name) in canon:
            coco_id_to_m2f[cat["id"]] = canon[_canon_name(name)]
        else:
            missing.append((cat["id"], name))

    if missing:
        raise ValueError(
            f"COCO categories not found in Mask2Former id2label: {missing}. "
            f"Mask2Former id2label sample: {list(id2label.items())[:5]}"
        )
    # Assert dense permutation: every model label should be covered exactly once
    used = sorted(coco_id_to_m2f.values())
    if used != list(range(len(id2label))):
        raise ValueError(
            f"COCO->Mask2Former mapping is not a dense permutation of {len(id2label)} labels. "
            f"used={used[:10]}...{used[-5:]}"
        )
    print(f"[cache:coco] mapped {len(coco_categories)}/{len(coco_categories)} COCO categories "
          f"to Mask2Former indices (dense permutation verified)")
    return coco_id_to_m2f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="number of val images (default 5000 = full)")
    parser.add_argument("--limit", type=int, default=None, help="for smoke test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not VAL_IMG_DIR.is_dir() or not PANOPTIC_DIR.is_dir() or not PANOPTIC_JSON.is_file():
        raise FileNotFoundError(
            f"COCO val + panoptic annotations not at expected paths: "
            f"VAL_IMG_DIR={VAL_IMG_DIR}, PANOPTIC_DIR={PANOPTIC_DIR}, PANOPTIC_JSON={PANOPTIC_JSON}"
        )

    print(f"[cache:coco] loading panoptic JSON: {PANOPTIC_JSON}")
    with open(PANOPTIC_JSON) as f:
        coco = json.load(f)
    coco_categories = coco["categories"]
    coco_anns_by_image = {a["image_id"]: a for a in coco["annotations"]}
    coco_images_meta = {im["id"]: im for im in coco["images"]}
    print(f"[cache:coco] {len(coco['images'])} images, {len(coco_categories)} categories, "
          f"{len(coco['annotations'])} annotations")

    # Sorted by image_id
    val_image_ids = sorted(coco_images_meta.keys())
    if args.limit is not None:
        val_image_ids = val_image_ids[: args.limit]
    elif args.n < len(val_image_ids):
        val_image_ids = val_image_ids[: args.n]
    n = len(val_image_ids)
    print(f"[cache:coco] processing {n} images")

    print(f"[cache:coco] loading Mask2Former-COCO-Panoptic ...")
    from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
    processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
    n_classes = len(model.config.id2label)
    log_n = float(np.log(n_classes))
    print(f"[cache:coco] model has {n_classes} class labels")

    coco_id_to_m2f = build_coco_cat_id_to_m2f_index(model.config.id2label, coco_categories)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[cache:coco] device = {device}")
    model = model.to(device).eval()
    if torch.cuda.is_available():
        model = model.half()

    miou = np.zeros(n, dtype=np.float32)
    pixacc = np.zeros(n, dtype=np.float32)
    g_softmax = np.zeros(n, dtype=np.float32)
    g_entropy = np.zeros(n, dtype=np.float32)
    t0 = time.time()

    from torch.nn.functional import interpolate

    for i, image_id in enumerate(val_image_ids):
        meta = coco_images_meta[image_id]
        img_path = VAL_IMG_DIR / meta["file_name"]
        ann = coco_anns_by_image.get(image_id)
        if ann is None:
            print(f"[cache:coco] WARN: no panoptic annotation for image {image_id}; skipping")
            miou[i] = 0.0
            pixacc[i] = 0.0
            g_softmax[i] = 0.0
            g_entropy[i] = 0.0
            continue
        pan_png_path = PANOPTIC_DIR / ann["file_name"]
        if not pan_png_path.exists():
            raise FileNotFoundError(f"missing panoptic PNG: {pan_png_path}")

        img = Image.open(img_path).convert("RGB")
        # Build per-pixel category_id (semantic GT) from panoptic encoding
        # Cast to int32 before arithmetic; formula R + 256G + 256²B matches panopticapi.rgb2id
        pan = np.asarray(Image.open(pan_png_path).convert("RGB"), dtype=np.int32)
        seg_id = pan[..., 0] + 256 * pan[..., 1] + 256 * 256 * pan[..., 2]
        seg_to_cat = {s["id"]: s["category_id"] for s in ann["segments_info"]}
        # Guard: COCO panoptic reserves segment_id=0 for void; flag if seen as a labeled segment
        if 0 in seg_to_cat:
            raise ValueError(f"Unexpected segment id 0 in annotation {ann['file_name']}")
        gt_m2f = np.full(seg_id.shape, -1, dtype=np.int32)  # -1 = ignore
        # Map each segment to model's label index space
        for sid, coco_cat in seg_to_cat.items():
            mask = seg_id == sid
            m2f_idx = coco_id_to_m2f.get(coco_cat, -1)
            gt_m2f[mask] = m2f_idx
        # Pixels with segment_id == 0 (no annotation in panoptic) stay -1 (ignore)

        # Forward pass
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
                out = model(**inputs)

        H, W = img.size[1], img.size[0]
        seg_maps = processor.post_process_semantic_segmentation(out, target_sizes=[(H, W)])
        pred = seg_maps[0].cpu().numpy().astype(np.int32)

        # Per-pixel class probability (float32 + normalise)
        class_logits = out.class_queries_logits.float()
        mask_logits = out.masks_queries_logits.float()
        class_prob = torch.softmax(class_logits, dim=-1)[..., :-1]
        mask_prob = torch.sigmoid(mask_logits)
        seg_prob = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob).clamp_min(0.0)
        seg_prob = seg_prob / seg_prob.sum(dim=1, keepdim=True).clamp(min=1e-12)
        seg_prob = interpolate(seg_prob, size=(H, W), mode="bilinear", align_corners=False)
        seg_prob = seg_prob.clamp(min=1e-12)
        seg_prob = seg_prob / seg_prob.sum(dim=1, keepdim=True)

        pixel_max = seg_prob.max(dim=1).values  # (1, H, W)
        gs = float(pixel_max.mean().item())

        entropy = -(seg_prob * torch.log(seg_prob)).sum(dim=1)  # (1, H, W)
        ge = float(1.0 - (entropy.mean().item() / log_n))
        ge = max(0.0, min(1.0, ge))

        m_i = per_image_miou(pred, gt_m2f, ignore_value=-1)
        a_i = per_image_pixel_accuracy(pred, gt_m2f, ignore_value=-1)
        if not np.isfinite(m_i) or not np.isfinite(a_i):
            raise ValueError(f"non-finite metric at idx {i}")
        miou[i] = float(np.clip(m_i, 0.0, 1.0))
        pixacc[i] = float(np.clip(a_i, 0.0, 1.0))
        g_softmax[i] = float(np.clip(gs, 0.0, 1.0))
        g_entropy[i] = float(np.clip(ge, 0.0, 1.0))

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(
                f"[cache:coco] {i+1}/{n} ({rate:.2f} im/s, eta {eta:.0f}s) "
                f"mean mIoU={miou[: i+1].mean():.4f} mean pixacc={pixacc[: i+1].mean():.4f} "
                f"mean g_soft={g_softmax[: i+1].mean():.4f}"
            )

    total = time.time() - t0
    print(f"[cache:coco] DONE in {total:.0f}s; mean mIoU = {miou.mean():.4f}; mean pixacc = {pixacc.mean():.4f}")

    out_dir = COCO_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    if not np.all(np.isfinite(miou)) or not np.all(np.isfinite(pixacc)):
        raise ValueError("non-finite in cache after compute")
    loss_miou = np.clip(1.0 - miou, 0.0, 1.0).astype(np.float32)
    loss_pixacc = np.clip(1.0 - pixacc, 0.0, 1.0).astype(np.float32)

    np.save(out_dir / "val_mask2former_coco_miou.npy", miou)
    np.save(out_dir / "val_mask2former_coco_loss.npy", loss_miou)
    np.save(out_dir / "val_mask2former_coco_pixacc.npy", pixacc)
    np.save(out_dir / "val_mask2former_coco_loss_pixacc.npy", loss_pixacc)
    np.save(out_dir / "val_mask2former_coco_g_softmax.npy", g_softmax)
    np.save(out_dir / "val_mask2former_coco_g_entropy.npy", g_entropy)

    md5s = {}
    for k in ["miou", "loss", "pixacc", "loss_pixacc", "g_softmax", "g_entropy"]:
        p = out_dir / f"val_mask2former_coco_{k}.npy"
        md5s[p.name] = md5_of_path(p)
    with open(out_dir / "val_mask2former_coco_md5.json", "w") as f:
        json.dump({
            "model_hf": "facebook/mask2former-swin-base-coco-panoptic",
            "n_images": n,
            "n_classes": n_classes,
            "mean_miou": float(miou.mean()),
            "mean_pixacc": float(pixacc.mean()),
            "mean_g_softmax": float(g_softmax.mean()),
            "mean_g_entropy": float(g_entropy.mean()),
            "runtime_seconds": total,
            "md5s": md5s,
        }, f, indent=2)
    print(f"[cache:coco] saved 6 .npy + MD5 json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
