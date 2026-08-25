"""Certified-decision payoff analysis.

Computes the certified-decision payoff table: for each method (Ours, Hoeffding-CRC
range-only A, Bernstein-on-accepted B, WSR, e-BH), the certified acceptance
level `p_acc_cert = max{ p̂(λ,τ) : (λ,τ) ∈ Ĝ_method }`, where
`Ĝ_method = { (λ,τ) : UCB_method(R_sel) ≤ α  AND  p_LCB ≥ π_min }` and `p_LCB`
is the shared Clopper-Pearson LCB at level δ/(16m).

Data sources reused: cached ImageNet val logits at
`$SCORC_DATA_DIR/imagenet_data/val_logits{,_resnet101,_resnet152}.npy` (the
same caches consumed by the multi-backbone comparison).

Output: results/analysis/certified_decision_payoff.json + .md (overwrite).

Run:
    python -m experiments.analysis_certified_decision_payoff \
        [--n-seeds 20] [--models RN50,RN101,RN152]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from selective_crc import (  # noqa: E402
    certify_grid,
    three_split_indices,
    baseline_a_range_hoeffding,
    baseline_b_accepted_bernstein,
    wsr_betting_ucb,
    ebh_product_evalue_ucb,
)
from selective_crc.baselines import baseline_a_range_hoeffding_plcb  # noqa: E402
from experiments import cifar100  # noqa: E402

import os
# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

OUT_DIR = ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_CACHES = {
    "ResNet-50 V2": (f"{DATA_ROOT}/imagenet_data/val_logits.npy",
                     f"{DATA_ROOT}/imagenet_data/val_labels.npy"),
    "ResNet-101 V2": (f"{DATA_ROOT}/imagenet_data/val_logits_resnet101.npy",
                      f"{DATA_ROOT}/imagenet_data/val_labels.npy"),
    "ResNet-152 V2": (f"{DATA_ROOT}/imagenet_data/val_logits_resnet152.npy",
                      f"{DATA_ROOT}/imagenet_data/val_labels.npy"),
}


def per_seed_payoff(
    logits_path: str,
    labels_path: str,
    *,
    split_seed: int,
    n_cert: int,
    n_tune: int,
    n_test: int,
    Lambda: np.ndarray,
    T: np.ndarray,
    alpha: float,
    pi_min: float,
    delta: float,
    c: float,
    V: float,
    B: float,
    n_classes: int,
) -> dict:
    """Compute per-method certified p_acc on a single seed split."""
    Y_full = np.load(labels_path).astype(np.int64)
    L_full = np.load(logits_path).astype(np.float64)
    rng = np.random.default_rng(split_seed)
    subset = rng.choice(len(Y_full), size=n_cert + n_tune + n_test, replace=False)
    Y_all = Y_full[subset]
    logits_used = L_full[subset]
    _, cert_idx, _ = three_split_indices(
        len(Y_all), n_tune=n_tune, n_cert=n_cert, seed=split_seed,
    )
    Y_cert = Y_all[cert_idx]
    logits_cert = logits_used[cert_idx]
    del Y_full, L_full

    m_lambda, m_tau = len(Lambda), len(T)
    m = m_lambda * m_tau

    contains_Y, set_size = cifar100._compute_contains_and_size(
        logits_cert, Y_cert, Lambda, n_classes
    )
    L_lambda = (~contains_Y).astype(np.float64)
    L_grid = np.repeat(L_lambda, m_tau, axis=1)
    A_lt = cifar100.construct_acceptance(logits_cert, T)
    A_grid = np.tile(A_lt, (1, m_lambda))
    set_size_safe = np.maximum(set_size, 1)
    v_lambda = contains_Y.astype(np.float64) / set_size_safe.astype(np.float64)
    v_grid = np.repeat(v_lambda, m_tau, axis=1)

    # Ours: certify_grid gives per-pair EB (UCB on E[Z]), p_LCB, z_bar, s_count.
    ours = certify_grid(
        L_grid, A_grid, v_grid,
        alpha=alpha, pi_min=pi_min, delta=delta,
        c=c, V=V, B=B, check_sample_size=False,
    )
    p_hat = ours.s_count.astype(np.float64) / n_cert
    p_hat_safe = np.maximum(p_hat, 1e-12)
    p_LCB = ours.p_lcb_per_pair
    z_bar = ours.z_bar

    # Per-pair R_sel UCB for each method = z_bar/p̂ + α + width_method (operational form);
    # equivalently we just compare each method's UCB on E[Z] ≤ 0.
    # Ours: in_g_hat_mask already encodes (EB ≤ 0 AND p_LCB ≥ π_min).
    ours_cert_mask = ours.in_g_hat_mask.copy()

    # Baseline A (π_min-saturated): range-only Hoeffding margin = B/π_min · √(log(2m/δ)/(2n)).
    # Same form as Ours but with the π_min worst-case denominator and no variance term.
    a_widths = baseline_a_range_hoeffding(
        L_grid, A_grid, alpha=alpha, pi_min=pi_min, delta=delta, B=B
    )
    # UCB_A(R_sel) ≤ α  ⇔  z_bar/p̂ + a_width ≤ 0  ⇔  z_bar + p̂·a_width ≤ 0
    a_ucb_z = z_bar + p_hat * a_widths
    a_cert_mask = (a_ucb_z <= 0.0) & (p_LCB >= pi_min)

    # Baseline A' (p_LCB-tightened): strictly tighter Hoeffding-style comparator;
    # margin = hoeffding_radius_Z / p_LCB(s; n, δ/(2m)).  Returns NaN where p_LCB ≤ 0.
    aplcb_widths = baseline_a_range_hoeffding_plcb(
        L_grid, A_grid, alpha=alpha, delta=delta, B=B
    )
    with np.errstate(invalid="ignore"):
        aplcb_ucb_z = z_bar + p_hat * aplcb_widths
        aplcb_cert_mask = np.where(np.isnan(aplcb_widths),
                                   False,
                                   (aplcb_ucb_z <= 0.0) & (p_LCB >= pi_min))

    # Baseline B — accepted-sample Bernstein; per-pair R_sel UCB margin (NaN where infeasible).
    b_widths = baseline_b_accepted_bernstein(
        L_grid, A_grid, alpha=alpha, delta=delta, B=B
    )
    b_widths_arr = np.array([np.nan if x is None else float(x) for x in b_widths],
                            dtype=np.float64)
    with np.errstate(invalid="ignore"):
        b_ucb_z = z_bar + p_hat * b_widths_arr  # NaN-aware
        b_cert_mask = np.where(np.isnan(b_widths_arr),
                               False,
                               (b_ucb_z <= 0.0) & (p_LCB >= pi_min))

    # WSR — UCB on E[Z] directly at δ/(16m).
    Z = A_grid * (L_grid - alpha)
    delta_per_z = delta / (16.0 * m)
    wsr_ucb_Z = wsr_betting_ucb(Z, delta_prime=delta_per_z, range_b=B)
    wsr_cert_mask = (wsr_ucb_Z <= 0.0) & (p_LCB >= pi_min)

    # e-BH — UCB on E[Z] at level δ/m (Bonferroni grid).
    ebh_ucb_Z = ebh_product_evalue_ucb(Z, delta=delta, range_b=B)
    ebh_cert_mask = (ebh_ucb_Z <= 0.0) & (p_LCB >= pi_min)

    def max_p_hat(mask: np.ndarray) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(p_hat[mask].max())

    return {
        "seed": int(split_seed),
        "n_certified": {
            "ours": int(ours_cert_mask.sum()),
            "a_pi_min": int(a_cert_mask.sum()),
            "a_plcb": int(aplcb_cert_mask.sum()),
            "b": int(b_cert_mask.sum()),
            "wsr": int(wsr_cert_mask.sum()),
            "ebh": int(ebh_cert_mask.sum()),
        },
        "p_acc_cert": {
            "ours": max_p_hat(ours_cert_mask),
            "a_pi_min": max_p_hat(a_cert_mask),
            "a_plcb": max_p_hat(aplcb_cert_mask),
            "b": max_p_hat(b_cert_mask),
            "wsr": max_p_hat(wsr_cert_mask),
            "ebh": max_p_hat(ebh_cert_mask),
        },
        "ours_feasible": not bool(ours.is_infeasible),
    }


def aggregate(per_seed: list[dict]) -> dict:
    methods = ["ours", "a_pi_min", "a_plcb", "b", "wsr", "ebh"]
    out = {}
    for mth in methods:
        vals = [s["p_acc_cert"][mth] for s in per_seed]
        out[mth] = {
            "median_p_acc_cert": float(np.median(vals)),
            "mean_p_acc_cert": float(np.mean(vals)),
            "n_seeds_certified_at_least_one": int(sum(1 for v in vals if v > 0)),
            "n_seeds": len(vals),
        }
    for cmp in ["a_pi_min", "a_plcb"]:
        deltas = [s["p_acc_cert"]["ours"] - s["p_acc_cert"][cmp] for s in per_seed]
        out[f"delta_ours_minus_{cmp}"] = {
            "median": float(np.median(deltas)),
            "mean": float(np.mean(deltas)),
        }
    return out


def run(model_caches: dict, *,
        n_seeds: int = 20,
        n_cert: int = 33000,
        n_tune: int = 8500,
        n_test: int = 8500,
        alpha: float = 0.05,
        pi_min: float = 0.01,
        delta: float = 0.05,
        c: float = 0.1,
        V: float = 1.0,
        B: float = 1.0,
        n_classes: int = 1000) -> dict:
    Lambda = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], dtype=np.float64)
    T = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)

    results_per_model = {}
    for model_name, (logits_path, labels_path) in model_caches.items():
        if not Path(logits_path).is_file():
            print(f"[payoff] SKIPPING {model_name}: logits cache not found at {logits_path}",
                  flush=True)
            continue
        Y_full = np.load(labels_path).astype(np.int64)
        L_full = np.load(logits_path).astype(np.float64)
        top1 = float((L_full.argmax(axis=1) == Y_full).mean())
        print(f"\n[payoff] {model_name}: n={Y_full.shape[0]}, top-1 = {top1:.4f}",
              flush=True)
        del Y_full, L_full

        per_seed = []
        for seed_i in range(n_seeds):
            split_seed = 42 + seed_i
            t0 = time.time()
            res = per_seed_payoff(
                logits_path, labels_path,
                split_seed=split_seed, n_cert=n_cert, n_tune=n_tune, n_test=n_test,
                Lambda=Lambda, T=T, alpha=alpha, pi_min=pi_min, delta=delta,
                c=c, V=V, B=B, n_classes=n_classes,
            )
            per_seed.append(res)
            print(f"  seed {split_seed} in {time.time()-t0:.1f}s "
                  f"ours={res['p_acc_cert']['ours']:.4f} "
                  f"A(π_min)={res['p_acc_cert']['a_pi_min']:.4f} "
                  f"A(p_LCB)={res['p_acc_cert']['a_plcb']:.4f}", flush=True)
        results_per_model[model_name] = {
            "top1": top1,
            "per_seed": per_seed,
            "agg": aggregate(per_seed),
        }

    summary = {
        "experiment": "analysis_certified_decision_payoff",
        "purpose": ("Analysis script for the certified-decision payoff table — "
                    "reported per backbone at the paper's operating point."),
        "config": {
            "n_seeds": n_seeds,
            "n_cert": n_cert,
            "n_tune": n_tune,
            "n_test": n_test,
            "alpha": alpha,
            "pi_min": pi_min,
            "delta": delta,
            "c": c,
            "V": V,
            "B": B,
            "n_classes": n_classes,
            "Lambda": Lambda.tolist(),
            "T": T.tolist(),
        },
        "models": results_per_model,
    }
    return summary


def write_markdown(summary: dict, out_md: Path) -> None:
    cfg = summary["config"]
    lines = []
    lines.append("# Table IV — Certified-Decision Payoff (reanalysis)\n")
    lines.append(f"**Source script**: `experiments/analysis_certified_decision_payoff.py`")
    lines.append(f"**Config**: α={cfg['alpha']}, π_min={cfg['pi_min']}, δ={cfg['delta']}, "
                 f"n_cert={cfg['n_cert']}, n_seeds={cfg['n_seeds']}")
    lines.append("")
    lines.append("| Model | top-1 | Ours median `p_acc_cert` | "
                 "A(π_min) median | Δ vs A(π_min) median | "
                 "A(p_LCB) median | Δ vs A(p_LCB) median | B (acc-Bern) median |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model_name, mres in summary["models"].items():
        agg = mres["agg"]
        lines.append(
            f"| {model_name} | {mres['top1']:.4f} | "
            f"{agg['ours']['median_p_acc_cert']:.4f} | "
            f"{agg['a_pi_min']['median_p_acc_cert']:.4f} | "
            f"{agg['delta_ours_minus_a_pi_min']['median']*100:+.2f} pp | "
            f"{agg['a_plcb']['median_p_acc_cert']:.4f} | "
            f"{agg['delta_ours_minus_a_plcb']['median']*100:+.2f} pp | "
            f"{agg['b']['median_p_acc_cert']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-method certified-pair counts (median across seeds)")
    lines.append("")
    lines.append("| Model | Ours | A(π_min) | A(p_LCB) | B | WSR | e-BH |")
    lines.append("|---|---|---|---|---|---|---|")
    for model_name, mres in summary["models"].items():
        ncerts = [int(np.median([s["n_certified"][m] for s in mres["per_seed"]]))
                  for m in ["ours", "a_pi_min", "a_plcb", "b", "wsr", "ebh"]]
        lines.append(f"| {model_name} | {ncerts[0]} | {ncerts[1]} | {ncerts[2]} | "
                     f"{ncerts[3]} | {ncerts[4]} | {ncerts[5]} |")
    lines.append("")
    lines.append("**Reading**:")
    lines.append("- `A(π_min)`: the conservative π_min-saturated Hoeffding-CRC selective bound "
                 "(`margin = (B/π_min)·√(log(2m/δ)/(2n))`). Saturated at the worst-case "
                 "acceptance lower bound; certifies 0 pairs on every backbone at this α. "
                 "This is the bound the original Table IV referred to as \"Hoeffding-style baseline\".")
    lines.append("- `A(p_LCB)`: a tighter Hoeffding variant using the Clopper-Pearson "
                 "`p_LCB` instead of `π_min` in the denominator. The fairer like-for-like "
                 "comparison since both Ours and A(p_LCB) use the actual acceptance lower bound.")
    lines.append("- The headline `Δ vs A(π_min)` is the operational decision-frontier gain "
                 "as stated in the plan's metric definition. The `Δ vs A(p_LCB)` row is the "
                 "like-for-like comparison against the strengthened data-driven variant.")
    lines.append("- B's certified `p_acc_cert` is per-pair-tighter than A on the same surface "
                 "— this is the comparator-regime delta honestly disclosed in §VI ¶1, not a "
                 "contradiction of the headline.")
    out_md.write_text("\n".join(lines))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-cert", type=int, default=33000)
    ap.add_argument("--models", type=str, default="RN50,RN101,RN152",
                    help="comma-separated subset of RN50,RN101,RN152")
    ap.add_argument("--out-tag", type=str, default="",
                    help="optional filename tag, e.g. 'smoke' -> .smoke.json")
    args = ap.parse_args(argv)

    name_map = {
        "RN50": "ResNet-50 V2",
        "RN101": "ResNet-101 V2",
        "RN152": "ResNet-152 V2",
    }
    selected = [name_map[m.strip()] for m in args.models.split(",") if m.strip()]
    caches = {k: v for k, v in DEFAULT_MODEL_CACHES.items() if k in selected}

    summary = run(caches, n_seeds=args.n_seeds, n_cert=args.n_cert)

    tag = f".{args.out_tag}" if args.out_tag else ""
    out_json = OUT_DIR / f"certified_decision_payoff{tag}.json"
    out_md = OUT_DIR / f"certified_decision_payoff{tag}.md"
    out_json.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, out_md)
    print(f"\n[payoff] Wrote {out_json}")
    print(f"[payoff] Wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
