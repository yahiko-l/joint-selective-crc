# joint-selective-crc

Code release for *"A Joint Finite-Sample Certificate for Adaptive Selective Conformal
Risk Control"* (SCoRC).

A selective risk controller has two legs that are estimated on the same calibration
sample: the risk it controls and the acceptance rate it buys. SCoRC certifies both
jointly at finite sample size. A configuration enters the certified set only when its
empirical-Bernstein risk bound and its acceptance-rate lower bound hold together, and
the reported configuration is the one maximizing a certified utility lower bound over
that set.

## What is here

- [`code/`](code/) the certifier, the baselines, the verification and analysis
  scripts, the cache-generation scripts, and a small cached COCO val2017 slice. The
  headline utility-leg results reproduce from the bundle alone, with no external
  dataset and no model checkpoint. [`code/README.md`](code/README.md) is the full
  guide: layout, environment, per-script reference, and reproduction conventions.

The precomputed result files behind the paper's tables and figures, and the
table/figure generators that render them, are released when the paper is published.

## Quick start

Python 3.12, NumPy and SciPy only:

```bash
cd code
pip install -r requirements.txt
python experiments/verify_oracle_nonvacuity.py
```

This reproduces the COCO headline and the three rungs of the utility ladder directly
from the bundled cache, together with the oracle non-emptiness and certified-set
optimality checks. It is deterministic: rerunning it reproduces its output exactly.

Two further entry points run from the bundle as well:

```bash
python experiments/verify_oracle_heldout.py   # held-out external-oracle certification
python experiments/audit_cor9_regime.py       # per-pair regime audit, 2,700 COCO cells
```

Everything else needs the per-image loss caches. Point `SCORC_DATA_DIR` at the
directory holding them first, as described in [`code/README.md`](code/README.md).

## Integrity

```bash
cd code && sha256sum -c CHECKSUMS.txt
```

## Citation

Citation details are added once the paper is published.
