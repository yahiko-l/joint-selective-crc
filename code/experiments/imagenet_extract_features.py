"""Cache ResNet-50 (IMAGENET1K_V2) penultimate features on real ImageNet val.

Loads the same parquet shards, in the same order, with the same transform, batch
size (128) and default numerical backend settings as `imagenet_compute_logits.py`,
and stores the 2048-d pooled features that feed the final fully connected layer.
Nothing is saved unless the run reproduces the cached arrays: the labels must equal
`val_labels.npy` exactly, and the pretrained fc layer applied to the new features
must reproduce every row of `val_logits.npy` within LOGIT_ATOL (a misaligned row
would differ by whole logits, not by rounding), with argmax agreement of at least
MIN_ARGMAX_AGREEMENT.

Outputs (next to the cached logits):
  $SCORC_DATA_DIR/imagenet_data/val_features_resnet50_v2.npy   # (50000, 2048) float32
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.models as tvm
from PIL import Image

# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

DATA = Path(DATA_ROOT) / "imagenet_data"
LOGIT_ATOL = 1e-3      # rounding differences are of order 1e-5
MIN_ARGMAX_AGREEMENT = 0.999


class ShardImages(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        self.images, self.transform = images, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        item = self.images[i]
        raw = item["bytes"] if isinstance(item, dict) else item
        return self.transform(Image.open(io.BytesIO(raw)).convert("RGB"))


def main():
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
    model = tvm.resnet50(weights=weights).eval().cuda()
    backbone = torch.nn.Sequential(*list(model.children())[:-1]).eval()
    transform = weights.transforms()

    feats, labels = [], []
    for shard in sorted((DATA / "data").glob("validation-*.parquet")):
        t0 = time.time()
        table = pq.ParquetFile(shard).read()
        loader = torch.utils.data.DataLoader(
            ShardImages(table.column("image").to_pylist(), transform),
            batch_size=128, shuffle=False, num_workers=16, pin_memory=True)
        out = []
        with torch.no_grad():
            for batch in loader:
                out.append(backbone(batch.cuda(non_blocking=True)).flatten(1).cpu().numpy())
        feats.append(np.concatenate(out).astype(np.float32))
        labels.append(table.column("label").to_numpy())
        print(f"{shard.name}: {len(feats[-1])} rows in {time.time() - t0:.1f}s", flush=True)

    feats = np.concatenate(feats)
    labels = np.concatenate(labels).astype(np.int64)
    cached_labels = np.load(DATA / "val_labels.npy")
    cached = np.load(DATA / "val_logits.npy")
    assert feats.shape == (len(cached_labels), 2048), feats.shape
    assert np.array_equal(labels, cached_labels), "label order differs from val_labels.npy"

    with torch.no_grad():
        logits = model.fc(torch.from_numpy(feats).cuda()).cpu().numpy()
    assert logits.shape == cached.shape, (logits.shape, cached.shape)
    row_err = np.abs(logits - cached).max(axis=1)
    agree = float((logits.argmax(1) == cached.argmax(1)).mean())
    print(f"features {feats.shape}; fc(features) vs cached logits: "
          f"max |diff| {row_err.max():.3e} (99.9th pct of row maxima {np.quantile(row_err, 0.999):.3e}), "
          f"argmax agreement {agree:.5f}, top-1 {(logits.argmax(1) == labels).mean():.4f}; "
          f"a shifted pairing (row i against cached row i+1) differs by a median of "
          f"{np.median(np.abs(logits[:-1] - cached[1:]).max(axis=1)):.2f}")
    assert row_err.max() <= LOGIT_ATOL, f"a row differs by {row_err.max():.3f} > {LOGIT_ATOL}"
    assert agree >= MIN_ARGMAX_AGREEMENT, f"argmax agreement {agree:.5f} < {MIN_ARGMAX_AGREEMENT}"
    np.save(DATA / "val_features_resnet50_v2.npy", feats)
    print(f"saved {DATA / 'val_features_resnet50_v2.npy'}")


if __name__ == "__main__":
    main()
