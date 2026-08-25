"""R026 + R027 — Cache per-image quantities for ADE20K val × {Mask2Former, SegFormer}.

For each image i in ADE20K val (n=2000), cache:
- per_image_miou[i]: mIoU of predicted vs ground-truth segmentation in [0, 1]
- per_image_loss[i] = 1 - per_image_miou[i] in [0, 1]  (bounded loss L)
- per_image_g_softmax[i]: mean per-pixel softmax-max confidence (primary acceptance score)
- per_image_g_entropy[i]: -mean per-pixel prediction entropy / log(150) (alternative, for G.4)

Outputs:
  $SCORC_DATA_DIR/ade20k_data/val_{model}_miou.npy        # (2000,) float32
  $SCORC_DATA_DIR/ade20k_data/val_{model}_loss.npy        # (2000,) float32  L = 1 - mIoU
  $SCORC_DATA_DIR/ade20k_data/val_{model}_g_softmax.npy   # (2000,) float32 in [0, 1]
  $SCORC_DATA_DIR/ade20k_data/val_{model}_g_entropy.npy   # (2000,) float32 in [0, 1]

Models:
  --model mask2former   facebook/mask2former-swin-base-ade-semantic
  --model segformer     nvidia/segformer-b2-finetuned-ade-512-512

Compute: ~30 min per backbone on 1 H100.

Spec: refine-logs/EXPERIMENT_PLAN_ADE20K_ADDENDUM.md §G.1.
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

ADE_ROOT = Path(f"{DATA_ROOT}/ade20k_data/ADEChallengeData2016")
OUT_DIR = Path(f"{DATA_ROOT}/ade20k_data")
N_CLASSES = 150
LOG_N_CLASSES = float(np.log(N_CLASSES))  # normalisation for entropy

MODEL_REGISTRY = {
    "mask2former": "facebook/mask2former-swin-base-ade-semantic",
    "segformer": "nvidia/segformer-b2-finetuned-ade-512-512",
}


def per_image_miou(pred: np.ndarray, gt: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """Per-image mIoU (mean over classes present in pred ∪ gt, excluding ignore=0).

    ADE20K convention: gt pixel value 1..150 = class; 0 = ignore.
    Pred output is 0..149.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    valid = gt > 0
    if not valid.any():
        return 0.0
    pred_v = pred[valid]
    gt_v = gt[valid] - 1
    classes_present = np.union1d(np.unique(pred_v), np.unique(gt_v))
    ious = []
    for c in classes_present:
        inter = ((pred_v == c) & (gt_v == c)).sum()
        union = ((pred_v == c) | (gt_v == c)).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def md5_of_path(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_mask2former(img: Image.Image, processor, model, device):
    """Returns (pred_HxW int, g_softmax float, g_entropy float)."""
    from torch.nn.functional import interpolate
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    H, W = img.size[1], img.size[0]
    seg_maps = processor.post_process_semantic_segmentation(out, target_sizes=[(H, W)])
    pred = seg_maps[0].cpu().numpy().astype(np.int32)

    # Per-pixel class probability: softmax(class_logits)[..., :-1] weighted by sigmoid(mask_logits)
    # Numerical-safety fix: upcast to float32 for entropy + normalisation safety
    class_logits = out.class_queries_logits.float()  # (1, Q, C+1)
    mask_logits = out.masks_queries_logits.float()  # (1, Q, H', W')
    class_prob = torch.softmax(class_logits, dim=-1)[..., :-1]  # drop no-object
    mask_prob = torch.sigmoid(mask_logits)
    seg_prob = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob).clamp_min(0.0)
    # Renormalise per pixel so the sum over classes = 1
    seg_prob = seg_prob / seg_prob.sum(dim=1, keepdim=True).clamp(min=1e-12)

    # Resize to original (already float32)
    seg_prob = interpolate(seg_prob, size=(H, W), mode="bilinear", align_corners=False)
    seg_prob = seg_prob.clamp(min=1e-12)
    seg_prob = seg_prob / seg_prob.sum(dim=1, keepdim=True)

    pixel_max = seg_prob.max(dim=1).values  # (1, H, W)
    g_softmax = float(pixel_max.mean().item())

    # Entropy (normalised by log(N)); seg_prob is float32, clamp at 1e-12 is safe
    entropy = -(seg_prob * torch.log(seg_prob)).sum(dim=1)  # (1, H, W)
    g_entropy = float(1.0 - (entropy.mean().item() / LOG_N_CLASSES))  # higher = more confident
    g_entropy = max(0.0, min(1.0, g_entropy))

    return pred, g_softmax, g_entropy


def run_segformer(img: Image.Image, processor, model, device):
    from torch.nn.functional import interpolate
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    # SegFormer logits: (1, C, h, w) — h, w are 1/4 of input
    # Numerical-safety fix: upcast logits to float32 before softmax + interpolate (fp16 safety)
    logits = out.logits.float()  # (1, 150, h, w)
    H, W = img.size[1], img.size[0]
    logits_full = interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    seg_prob = torch.softmax(logits_full, dim=1)  # (1, C, H, W) float32
    pred = seg_prob.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    pixel_max = seg_prob.max(dim=1).values  # (1, H, W)
    g_softmax = float(pixel_max.mean().item())

    seg_prob_safe = seg_prob.clamp(min=1e-12)
    entropy = -(seg_prob_safe * torch.log(seg_prob_safe)).sum(dim=1)
    g_entropy = float(1.0 - (entropy.mean().item() / LOG_N_CLASSES))
    g_entropy = max(0.0, min(1.0, g_entropy))

    return pred, g_softmax, g_entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--n", type=int, default=2000, help="number of val images (default 2000 = full)")
    parser.add_argument("--limit", type=int, default=None, help="for smoke test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    val_img_dir = ADE_ROOT / "images" / "validation"
    val_ann_dir = ADE_ROOT / "annotations" / "validation"
    val_imgs = sorted(val_img_dir.glob("*.jpg"))
    val_anns = sorted(val_ann_dir.glob("*.png"))
    assert len(val_imgs) == 2000, f"expected 2000, got {len(val_imgs)}"
    # Explicit stem alignment check
    assert [p.stem for p in val_imgs] == [p.stem for p in val_anns], \
        "image/annotation stems do not align"

    if args.limit is not None:
        val_imgs = val_imgs[: args.limit]
        val_anns = val_anns[: args.limit]
        n = args.limit
    else:
        # --n is honoured here
        if args.n > len(val_imgs):
            raise ValueError(f"requested n={args.n} > {len(val_imgs)}")
        val_imgs = val_imgs[: args.n]
        val_anns = val_anns[: args.n]
        n = args.n

    print(f"[cache:{args.model}] {n} images, model = {MODEL_REGISTRY[args.model]}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[cache:{args.model}] device = {device}")

    if args.model == "mask2former":
        from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
        processor = Mask2FormerImageProcessor.from_pretrained(MODEL_REGISTRY[args.model])
        model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_REGISTRY[args.model])
        runner = run_mask2former
    elif args.model == "segformer":
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        processor = SegformerImageProcessor.from_pretrained(MODEL_REGISTRY[args.model])
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_REGISTRY[args.model])
        runner = run_segformer
    else:
        raise ValueError(args.model)

    model = model.to(device).eval()
    if torch.cuda.is_available():
        model = model.half()  # fp16 inference for speed
        print(f"[cache:{args.model}] model in fp16 on {device}")

    miou = np.zeros(n, dtype=np.float32)
    g_softmax = np.zeros(n, dtype=np.float32)
    g_entropy = np.zeros(n, dtype=np.float32)
    t0 = time.time()

    for i, (ip, ap) in enumerate(zip(val_imgs, val_anns)):
        img = Image.open(ip).convert("RGB")
        gt = np.array(Image.open(ap))
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            pred, gs, ge = runner(img, processor, model, device)
        m_i = per_image_miou(pred, gt)
        # Enforce finite + clip to [0, 1] at cache time
        if not np.isfinite(m_i):
            raise ValueError(f"non-finite mIoU at image index {i}")
        miou[i] = float(np.clip(m_i, 0.0, 1.0))
        g_softmax[i] = float(np.clip(gs, 0.0, 1.0))
        g_entropy[i] = float(np.clip(ge, 0.0, 1.0))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(
                f"[cache:{args.model}] {i+1}/{n} ({rate:.2f} im/s, eta {eta:.0f}s) "
                f"mean mIoU={miou[: i+1].mean():.4f} mean g_soft={g_softmax[: i+1].mean():.4f}"
            )

    total = time.time() - t0
    print(f"[cache:{args.model}] DONE in {total:.0f}s; mean mIoU = {miou.mean():.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Verify finite + clip before save
    if not np.all(np.isfinite(miou)):
        raise ValueError("non-finite mIoU in cache after compute")
    loss = np.clip(1.0 - miou, 0.0, 1.0).astype(np.float32)
    np.save(OUT_DIR / f"val_{args.model}_miou.npy", miou)
    np.save(OUT_DIR / f"val_{args.model}_loss.npy", loss)
    np.save(OUT_DIR / f"val_{args.model}_g_softmax.npy", g_softmax)
    np.save(OUT_DIR / f"val_{args.model}_g_entropy.npy", g_entropy)

    # MD5 for reproducibility
    md5s = {}
    for k in ["miou", "loss", "g_softmax", "g_entropy"]:
        p = OUT_DIR / f"val_{args.model}_{k}.npy"
        md5s[p.name] = md5_of_path(p)
    md5_file = OUT_DIR / f"val_{args.model}_md5.json"
    with open(md5_file, "w") as f:
        json.dump({
            "model_hf": MODEL_REGISTRY[args.model],
            "n_images": n,
            "mean_miou": float(miou.mean()),
            "mean_g_softmax": float(g_softmax.mean()),
            "mean_g_entropy": float(g_entropy.mean()),
            "runtime_seconds": total,
            "md5s": md5s,
        }, f, indent=2)
    print(f"[cache:{args.model}] saved 4 .npy + MD5 json to {OUT_DIR}")
    print(f"[cache:{args.model}] mean mIoU = {miou.mean():.4f}; mean g_softmax = {g_softmax.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
