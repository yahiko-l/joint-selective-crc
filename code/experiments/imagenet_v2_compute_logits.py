"""Compute ResNet-50 V2 logits on ImageNet-V2 matched-frequency and cache as .npy.

ImageNet-V2 (Recht et al. 2019) is a re-collected ImageNet validation set
with the same 1000 class labels. Used here for B11 distribution-shift
robustness analysis.

Loads (directory-of-classes format):
  $SCORC_DATA_DIR/imagenet_v2_data/imagenetv2-matched-frequency-format-val/
    0/{img1.jpeg, ..., img10.jpeg}
    1/{img1.jpeg, ..., img10.jpeg}
    ...
    999/{img1.jpeg, ..., img10.jpeg}
  Total: 1000 class folders × 10 images = 10,000 JPEG files

Outputs:
  $SCORC_DATA_DIR/imagenet_v2_data/val_logits_v2.npy   # (10000, 1000) float32
  $SCORC_DATA_DIR/imagenet_v2_data/val_labels_v2.npy   # (10000,) int64

Model: torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
       — identical to the model used for the canonical ImageNet val cache,
       so direct distribution-shift comparison is valid.

Expected top-1 accuracy: ~0.68-0.70 (Recht 2019 reports 10-13% drop from val).
"""

from __future__ import annotations

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

def main():
    v2_root = Path(f"{DATA_ROOT}/imagenet_v2_data/imagenetv2-matched-frequency-format-val")
    out_dir = Path(f"{DATA_ROOT}/imagenet_v2_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not v2_root.is_dir():
        raise FileNotFoundError(f"ImageNet-V2 root not found at {v2_root}")

    # Enumerate class folders (named 0..999); these are the canonical ImageNet class indices
    class_dirs = sorted(
        [d for d in v2_root.iterdir() if d.is_dir()],
        key=lambda d: int(d.name),
    )
    if len(class_dirs) != 1000:
        raise ValueError(f"Expected 1000 class folders, found {len(class_dirs)}")
    print(f"[v2 logits] found {len(class_dirs)} class folders under {v2_root}")

    # Load model
    import torchvision.models as tvm
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    print(f"[v2 logits] loading ResNet-50 V2 weights ({weights}) ...")
    model = tvm.resnet50(weights=weights).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[v2 logits] model on {device}")

    # Iterate class folders, preloading file paths and labels
    file_paths = []
    labels = []
    for cls_dir in class_dirs:
        cls_label = int(cls_dir.name)
        imgs = sorted(cls_dir.glob("*.jpeg"))
        if not imgs:
            imgs = sorted(cls_dir.glob("*.jpg"))
        if not imgs:
            imgs = sorted(cls_dir.glob("*.png"))
        for img_path in imgs:
            file_paths.append(img_path)
            labels.append(cls_label)
    labels = np.array(labels, dtype=np.int64)
    print(f"[v2 logits] enumerated {len(file_paths)} images, labels shape={labels.shape}")

    # Batch inference
    batch_size = 128
    n_total = len(file_paths)
    all_logits = []

    t0 = time.time()
    for b_start in range(0, n_total, batch_size):
        b_end = min(b_start + batch_size, n_total)
        tensors = []
        for i in range(b_start, b_end):
            img = Image.open(file_paths[i]).convert("RGB")
            tensors.append(transform(img))
        batch = torch.stack(tensors).to(device, non_blocking=True)
        with torch.no_grad():
            out = model(batch).cpu().numpy().astype(np.float32)
        all_logits.append(out)

        if (b_start // batch_size) % 10 == 0:
            pct = 100.0 * b_end / n_total
            print(f"  {b_end}/{n_total} ({pct:.1f}%) in {time.time()-t0:.1f}s", flush=True)

    logits = np.concatenate(all_logits, axis=0)
    print(f"\n[v2 logits] total: logits shape={logits.shape}, labels shape={labels.shape}")
    acc = float((logits.argmax(axis=1) == labels).mean())
    print(f"[v2 logits] top-1 accuracy on ImageNet-V2 matched-frequency: {acc:.4f}")
    print(f"           (Recht 2019: expect ~0.68-0.70 for ResNet-50 V2; ~10-13pp below val)")

    logits_path = out_dir / "val_logits_v2.npy"
    labels_path = out_dir / "val_labels_v2.npy"
    np.save(logits_path, logits)
    np.save(labels_path, labels)
    print(f"[v2 logits] saved {logits_path} ({logits.nbytes/1e6:.1f} MB)")
    print(f"[v2 logits] saved {labels_path} ({labels.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
