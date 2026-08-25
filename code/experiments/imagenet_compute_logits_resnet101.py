"""Compute ResNet-101 V2 logits on real ImageNet val (HF parquet shards) and cache as .npy.

Loads:
  $SCORC_DATA_DIR/imagenet_data/data/validation-*.parquet (14 shards, ~50k rows total)
Outputs:
  $SCORC_DATA_DIR/imagenet_data/val_logits_resnet101.npy   # (50000, 1000) float32
  $SCORC_DATA_DIR/imagenet_data/val_labels.npy   # (50000,) int64

Model: torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
       — Torchvision published top-1 ≈ 0.8189 (V2 weights); our run achieved 0.8191.

This is the second-architecture supplement, confirming
the C2 regime claim on a stronger ImageNet model than ResNet-50 V2 (the
primary). Sister script to experiments/imagenet_compute_logits.py (ResNet-50 V2).
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
    print(f"[logits] found {len(parquet_files)} parquet shards under {parquet_dir}")

    # Load model
    import torchvision.models as tvm
    weights = tvm.ResNet101_Weights.IMAGENET1K_V2
    transform = weights.transforms()  # standard 256→224 center crop + normalize
    print(f"[logits] loading ResNet-101 V2 weights ({weights}) ...")
    model = tvm.resnet101(weights=weights).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[logits] model on {device}")

    # Iterate shards
    all_logits = []
    all_labels = []
    batch_size = 128

    import pyarrow.parquet as pq

    for shard_idx, pq_path in enumerate(parquet_files):
        print(f"\n[logits] shard {shard_idx+1}/{len(parquet_files)}: {pq_path.name}", flush=True)
        t0 = time.time()
        pq_file = pq.ParquetFile(pq_path)
        n_rows = pq_file.metadata.num_rows
        print(f"  {n_rows} rows in this shard")

        # Iterate in batches via parquet's row group iteration
        shard_logits = []
        shard_labels = []

        # Read whole shard (memory-fits since each shard is < 500 MB)
        tbl = pq_file.read()
        images = tbl.column("image").to_pylist()
        labels = tbl.column("label").to_numpy()
        # images is a list of dicts {"bytes": b'...', "path": '...'}
        # decode each to PIL, then transform

        for b_start in range(0, n_rows, batch_size):
            b_end = min(b_start + batch_size, n_rows)
            tensors = []
            for i in range(b_start, b_end):
                img_dict = images[i]
                img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
                if img_bytes is None:
                    # Some HF rows have path-only; skip (shouldn't happen for val)
                    raise ValueError(f"Row {i} has no image bytes")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                tensors.append(transform(img))
            batch = torch.stack(tensors).to(device, non_blocking=True)
            with torch.no_grad():
                out = model(batch).cpu().numpy().astype(np.float32)
            shard_logits.append(out)
            shard_labels.append(labels[b_start:b_end])

            if (b_start // batch_size) % 20 == 0:
                pct = 100.0 * b_end / n_rows
                print(f"  {b_end}/{n_rows} ({pct:.1f}%) in {time.time()-t0:.1f}s", flush=True)

        shard_logits_np = np.concatenate(shard_logits, axis=0)
        shard_labels_np = np.concatenate(shard_labels, axis=0)
        all_logits.append(shard_logits_np)
        all_labels.append(shard_labels_np)
        print(f"  shard done in {time.time()-t0:.1f}s; shape={shard_logits_np.shape}")

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    print(f"\n[logits] total: logits shape={logits.shape}, labels shape={labels.shape}")
    acc = float((logits.argmax(axis=1) == labels).mean())
    print(f"[logits] top-1 accuracy on val: {acc:.4f} (expected ~0.8189 for ResNet-101 V2)")

    logits_path = out_dir / "val_logits_resnet101.npy"
    labels_path = out_dir / "val_labels.npy"
    np.save(logits_path, logits)
    np.save(labels_path, labels)
    print(f"[logits] saved {logits_path} ({logits.nbytes/1e6:.1f} MB)")
    print(f"[logits] saved {labels_path} ({labels.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
