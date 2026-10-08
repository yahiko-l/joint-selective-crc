# SCoRC: joint selective conformal risk control

Reference implementation for *"A Joint Finite-Sample Certificate for Adaptive
Selective Conformal Risk Control."*

It contains the certifier, the baselines, the analysis, verification and experiment
scripts, the cache-generation scripts, a small cached COCO data slice that makes the
utility-leg results reproducible with no external dataset and no model checkpoint, the
precomputed result files behind the paper's tables and figures, and the scripts that
render those tables and figures.

## Layout

```
code/
├── src/selective_crc/     Core library
│   ├── bounds.py            Maurer–Pontil empirical-Bernstein, Clopper–Pearson, Chernoff
│   ├── certify.py           certify_grid(): EB ≤ 0 ∧ p_LCB ≥ π_min, argmax U_LCB over Ĝ
│   ├── baselines.py         Hoeffding–CRC, A(π_min), A(p_LCB), sign-aware A(CP±),
│   │                        per-pair Bernstein, WSR
│   └── grid_search.py, protocol.py
├── experiments/           Verification, analysis, experiment and cache-generation scripts
├── results/               Precomputed result files behind the paper's tables and figures
├── paper/                 Scripts that render the paper's tables and figures from results/
├── configs/               YAML config for the ImageNet runner
└── data/coco/             Cached COCO val2017 per-image arrays (6 files, 120 KB)
```

## Environment

Python 3.12. The certifier and both verification scripts need only:

```
numpy==1.26.4
scipy==1.16.3
```

(`requirements.txt`). The `*_compute_logits.py` scripts, which recompute per-image
losses from scratch, additionally need `torch` and `transformers` plus the public
datasets and checkpoints, neither of which is bundled. `imagenet_extract_features.py`
needs `torch`, `torchvision`, `pyarrow` and `Pillow`, and
`learned_deferral_imagenet.py` needs `torch` and `torchvision`. The scripts in `paper/`
need `matplotlib` as well; the paper's figures were rendered with matplotlib 3.10.7.

**Dataset cache location.** Scripts that read per-image caches resolve them under
`$SCORC_DATA_DIR`, defaulting to this bundle's `data/` directory. Set it to the
directory holding `imagenet_data/`, `imagenet_v2_data/`, `cifar100_data/`,
`coco_data/`, `ade20k_data/`:

```bash
export SCORC_DATA_DIR=/path/to/caches
```

The two verification scripts below need none of this: they read the bundled
`data/coco/` arrays (and also honour `SCORC_COCO_DIR` if you keep COCO elsewhere).

## Runs with nothing but this bundle

```bash
python experiments/verify_oracle_nonvacuity.py
```

Reproduces the COCO headline and the three rungs of the utility ladder of Section 4.7
directly from the bundled cache. Verified output, medians over 20 COCO softmax splits
at α = 0.10:

| quantity | reproduced | paper |
|---|---|---|
| certified `p_acc` | 0.221 | 0.221 |
| accepted-loss `σ̂²` | 0.0058 | 0.0058 |
| certified `U_LCB` | 0.199 | 0.199 |
| `|Ĝ|` | 4 | 4 |

together with:

- the **worst-case** Theorem 1 margin oracle is empty at α = 0.10 (γ_r ≈ 0.713 > α): 0/20 splits;
- the **certified-set** optimality of Corollary 6 holds 20/20, median slack 2γ_u = 0.062;
- a calibration **plug-in** surrogate of the Corollary 7 variance-adaptive oracle is
  non-empty at α = 0.15 (`U^margin_va = 0.446`), an empirical diagnostic rather than a
  population-level certification;
- a synthetic existence demo (α = 0.30, π_min = 0.20, n = 25,000): the literal
  Theorem 1 oracle is non-empty, `U^margin = 0.484`, guarantee `0.484 ≥ 0.459`.

```bash
python experiments/verify_oracle_heldout.py
```

Certifies external-oracle non-emptiness on held-out data: population-valid
`U* ≥ 0.476` at 95% confidence on the controlled surface, and an empty result on every
scanned COCO configuration.

```bash
python experiments/audit_cor9_regime.py
```

Per-(grid pair, seed) regime audit. The COCO blocks run from the bundled cache and
cover 2,700 cells, on which the leading-order rule is tested directly against the
realised per-pair winner. The ImageNet block re-reads the cached run in
`results/imagenet_primary_real/`.

Both verification scripts are deterministic: rerunning them reproduces their output
exactly.

## Needs the per-image caches

These scripts reproduce the paper's analyses from cached per-image losses. Point
`SCORC_DATA_DIR` at a directory holding those caches first; some also re-read result
files in `results/`.

| script | analysis |
|---|---|
| `analysis_signaware_valid_ratio.py` | sign-aware valid-vs-valid UCB excess comparison |
| `analysis_joint_baseline.py` | matched joint Hoeffding ablation (joint-H) |
| `analysis_vc_sensitivity.py` | utility-parameter sensitivity in v and c |
| `analysis_resplit_effectsize.py` | re-split effect sizes and split-randomization intervals |
| `analysis_infeasibility_census.py` | infeasibility census across every evaluation surface |
| `analysis_cert_frontier.py` | per-pair certified frontier |
| `analysis_certified_decision_payoff.py` | certified-decision payoff across backbones |
| `ablation_sensitivity_star.py` | hyperparameter star design and small-calibration block |
| `nonmonotone_synthetic.py`, `nonmonotone_imagenet.py` | non-monotone losses: synthetic couplings with CRC and RCPS threshold rules (needs no cache), and an ImageNet top-k set loss |
| `learned_deferral_imagenet.py` | learned abstention head on cached ResNet-50 V2 features, certified next to softmax scores |
| `runtime_grid_scaling.py` | certifier runtime versus grid size |
| `imagenet_scale.py`, `synthetic_full.py` | the ImageNet and synthetic runners |

`experiments/cifar100.py` ships as the shared acceptance-set helper imported by the
ImageNet runner; its standalone CIFAR-100 sanity entry point is not part of the
released pipeline.

## Regenerating the per-image caches

`{imagenet,cifar100,imagenet_v2}_compute_logits*.py`, `coco_compute_logits.py` and
`ade20k_compute_logits.py` recompute the per-image losses and acceptance scores from
the public datasets (ImageNet-1k val, CIFAR-100 test, COCO val2017 panoptic, ADE20K)
and the checkpoints named in the paper (ResNet-50/101/152 V2, Mask2Former-Swin-B,
SegFormer-MiT-B2). Datasets and checkpoints are large and externally hosted, so they
are not bundled. `imagenet_extract_features.py` caches the ResNet-50 V2 penultimate
features that `learned_deferral_imagenet.py` reads, and saves them only when the
pretrained classifier layer reproduces the cached ImageNet logits.

## Result files

`results/` holds the precomputed result files behind the paper's tables and figures, in
the directories that the scripts above write to:

| directory | contents |
|---|---|
| `ablation_supplement/` | ablations, sweeps, sensitivity analyses, baseline comparisons, the segmentation runs, and the certifier timings |
| `analysis/` | certified-decision payoff and frontier, infeasibility census, per-cell regime audit, held-out oracle certification |
| `imagenet_primary_real/`, `cifar100_full/`, `synthetic_full/` | per-seed validity runs on ImageNet, CIFAR-100 and the synthetic surface |
| `nonmonotone/`, `learned_deferral/` | the non-monotone-loss and learned-abstention experiments |

## Tables and figures

Each script in `paper/` renders one of the paper's tables or figures from `results/`
and writes it to `paper/out/`:

| script | output |
|---|---|
| `table_validity.py` | held-out validity across the four evaluation surfaces |
| `table_cert_decision.py` | certified-decision payoff on three ImageNet backbones |
| `table_effect_sizes.py` | re-split effect sizes for the headline comparisons |
| `table_ingredients.py` | eight-ingredient stress matrix |
| `fig_validity.py` | per-seed realized test-risk margins on six evaluation surfaces |
| `fig_width_scaling.py` | closed-form width scaling, three panels |
| `fig_variance_adaptive.py` | variance-adaptive width ratios on ImageNet |
| `fig_frontier.py` | certified operating frontier on three ImageNet backbones |
| `fig_coco_acceptance.py` | COCO certified acceptance and realized test behaviour, two panels |

## Integrity

`CHECKSUMS.txt` lists the SHA-256 of the bundled cached arrays:

```bash
sha256sum -c CHECKSUMS.txt
```

## Conventions

- COCO headline runs use 30 random calibration/test re-splits per configuration; on the
  headline configuration 29 of 30 are feasible. Medians of certified quantities are
  taken over the feasible re-splits. The re-split effect sizes instead use the
  all-split convention, in which a re-split on which the certifier abstains contributes
  a zero certified-acceptance gap: an evaluation convention, not a certified value.
- Re-split intervals describe the split-randomization distribution conditional on one
  fixed evaluation pool. They do not quantify generalization beyond that pool, and no
  re-split p-value is reported anywhere.
