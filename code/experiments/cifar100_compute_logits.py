"""Compute real CIFAR-100 logits using chenyaofo/pytorch-cifar-models ResNet (default ResNet-56).

Caches test logits + labels to $SCORC_DATA_DIR/cifar100_data/.

Note: chenyaofo's hub provides resnet20 / resnet32 / resnet44 / resnet56 for
CIFAR-100. The original user request was ResNet-110, but that model is not in
the chenyaofo hub; ResNet-56 is the closest available standard checkpoint
(top-1 ≈ 0.7261). The accuracy difference is immaterial for our certification
diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T


import os
# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=f"{DATA_ROOT}/cifar100_data")
    parser.add_argument("--model", default="cifar100_resnet56",
                       help="One of chenyaofo/pytorch-cifar-models entries (resnet20/32/44/56)")
    parser.add_argument("--data-root", default=f"{DATA_ROOT}/cifar100_data/data")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"[cifar100] loading {args.model} from chenyaofo/pytorch-cifar-models hub")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", args.model,
                          pretrained=True, trust_repo=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[cifar100] using device: {device}")

    # CIFAR-100 normalization (chenyaofo's expected preprocessing)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]),
    ])
    test_set = torchvision.datasets.CIFAR100(
        root=str(data_root), train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                         shuffle=False, num_workers=4)

    all_logits, all_labels = [], []
    print(f"[cifar100] running inference on {len(test_set)} test images...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    top1 = float((logits.argmax(1) == labels).mean())
    print(f"[cifar100] logits shape: {logits.shape}, labels shape: {labels.shape}")
    print(f"[cifar100] top-1 accuracy: {top1:.4f}")

    logits_path = out_dir / f"test_logits_{args.model}.npy"
    labels_path = out_dir / "test_labels.npy"
    np.save(logits_path, logits.astype(np.float32))
    np.save(labels_path, labels.astype(np.int64))
    print(f"[cifar100] wrote {logits_path} ({logits_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[cifar100] wrote {labels_path}")


if __name__ == "__main__":
    main()
