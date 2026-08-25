"""Compute ResNet-152 V2 logits on real ImageNet val (HF parquet shards) and cache as .npy.

Identical pipeline to imagenet_compute_logits.py and imagenet_compute_logits_resnet101.py,
but uses torchvision ResNet-152 V2 (~82.5% top-1 acc on ImageNet val).

Outputs:
  $SCORC_DATA_DIR/imagenet_data/val_logits_resnet152.npy  # (50000, 1000) float32
"""

from __future__ import annotations

import io
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
    parquet_dir = Path(f"{DATA_ROOT}/imagenet_data/data")
    out_dir = Path(f"{DATA_ROOT}/imagenet_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("validation-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No validation-*.parquet under {parquet_dir}")
    print(f"[RN152] found {len(parquet_files)} parquet shards")

    import torchvision.models as tvm
    weights = tvm.ResNet152_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    print(f"[RN152] loading ResNet-152 V2 weights ({weights}) ...")
    model = tvm.resnet152(weights=weights).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[RN152] model on {device}")

    all_logits = []
    all_labels = []
    batch_size = 96  # smaller than RN50 (RN152 has 60M params)

    import pyarrow.parquet as pq

    for shard_idx, pq_path in enumerate(parquet_files):
        print(f"\n[RN152] shard {shard_idx+1}/{len(parquet_files)}: {pq_path.name}", flush=True)
        t0 = time.time()
        pq_file = pq.ParquetFile(pq_path)
        n_rows = pq_file.metadata.num_rows

        tbl = pq_file.read()
        images = tbl.column("image").to_pylist()
        labels = tbl.column("label").to_numpy()

        shard_logits = []
        for b_start in range(0, n_rows, batch_size):
            b_end = min(b_start + batch_size, n_rows)
            tensors = []
            for i in range(b_start, b_end):
                img_dict = images[i]
                img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
                if img_bytes is None:
                    raise ValueError(f"Row {i} has no image bytes")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                tensors.append(transform(img))
            batch = torch.stack(tensors).to(device, non_blocking=True)
            with torch.no_grad():
                out = model(batch).cpu().numpy().astype(np.float32)
            shard_logits.append(out)
            if (b_start // batch_size) % 20 == 0:
                pct = 100.0 * b_end / n_rows
                print(f"  {b_end}/{n_rows} ({pct:.1f}%) in {time.time()-t0:.1f}s", flush=True)

        shard_logits_np = np.concatenate(shard_logits, axis=0)
        all_logits.append(shard_logits_np)
        all_labels.append(labels)
        print(f"  shard done in {time.time()-t0:.1f}s")

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    acc = float((logits.argmax(axis=1) == labels).mean())
    print(f"\n[RN152] total logits shape={logits.shape}, top-1 = {acc:.4f}")
    print(f"        (expected ~0.82-0.83 for ResNet-152 V2)")

    logits_path = out_dir / "val_logits_resnet152.npy"
    np.save(logits_path, logits)
    print(f"[RN152] saved {logits_path} ({logits.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
