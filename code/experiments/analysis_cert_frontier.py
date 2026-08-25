"""Certified-decision FRONTIER dump --- per-grid-pair data behind Fig 5.

Companion to `analysis_certified_decision_payoff.py` (Table IV). Table IV reports
only the SCALAR endpoint `p_acc_cert = max{p̂ : certified}` and the certified-pair
COUNT; that scalar makes Ours and the strengthened comparator A(p_LCB) look tied
(`Δ = +0.00`). This script persists the per-pair geometry the scalar hides so the
frontier figure can show STRICT dominance: for every grid pair (λ, τ) and every
method it dumps the certified upper bound on the selected risk, `UCB(R_sel)`, and
the binary certification flag (risk-UCB ≤ α AND p_LCB ≥ π_min).

Certification is the source of truth (EB ≤ 0 on E[Z]). The displayed y-quantity is
the same condition mapped to the R_sel scale:

    UCB(R_sel) = α + UCB(E[Z]) / p̂ ,    Z := A·(L − α),   E[Z] = p_acc·(R_sel − α)

so a pair is certified by a method  iff  UCB(R_sel)_method ≤ α  AND  p_LCB ≥ π_min.
The map sends the boundary UCB(E[Z])=0 to exactly α regardless of p̂, so the y-axis
threshold line at α IS the certification boundary (faithful, not a re-derivation).

Reuses the SAME machinery and α/π_min/δ/n_cert config as the Table IV script, so
the right endpoints (.275/.805/.829) and counts (21/33/33 vs 7/27/27) must match.
Does NOT touch results/analysis/certified_decision_payoff.json.

Output: results/analysis/certified_decision_frontier.json (overwrite).

Run:
    python -m experiments.analysis_cert_frontier [--n-seeds 20] [--models RN50,RN101,RN152]
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


def per_seed_frontier(
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
    """Per-pair certified risk-UCB + certification flags for one seed split.

    Returns arrays of length m = |Lambda|*|T| (grid pair k: lambda_idx = k // |T|,
    tau_idx = k % |T|), matching certify_grid's repeat/tile column order.
    """
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

    contains_Y, set_size = cifar100._compute_contains_and_size(
        logits_cert, Y_cert, Lambda, n_classes
    )
    L_lambda = (~contains_Y).astype(np.float64)
    L_grid = np.repeat(L_lambda, m_tau, axis=1)          # col k -> lambda k//m_tau
    A_lt = cifar100.construct_acceptance(logits_cert, T)
    A_grid = np.tile(A_lt, (1, m_lambda))                # col k -> tau   k%m_tau
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
    eb = ours.eb_per_pair                                # Ours UCB on E[Z]

    acc_ok = p_LCB >= pi_min

    # --- per-method UCB on E[Z] (certify iff <= 0 AND acc_ok) ---
    ours_ucb_z = eb
    ours_cert = (ours_ucb_z <= 0.0) & acc_ok

    a_widths = baseline_a_range_hoeffding(
        L_grid, A_grid, alpha=alpha, pi_min=pi_min, delta=delta, B=B
    )
    a_ucb_z = z_bar + p_hat * a_widths
    a_cert = (a_ucb_z <= 0.0) & acc_ok

    aplcb_widths = baseline_a_range_hoeffding_plcb(
        L_grid, A_grid, alpha=alpha, delta=delta, B=B
    )
    with np.errstate(invalid="ignore"):
        aplcb_ucb_z = z_bar + p_hat * aplcb_widths       # NaN where width NaN
        aplcb_cert = np.where(np.isnan(aplcb_widths),
                              False,
                              (aplcb_ucb_z <= 0.0) & acc_ok)

    # --- map UCB on E[Z] to the R_sel scale: UCB(R_sel) = alpha + UCB(E[Z]) / p_hat ---
    def to_rsel(ucb_z):
        return alpha + ucb_z / p_hat_safe

    ucb_rsel_ours = to_rsel(ours_ucb_z)
    ucb_rsel_a = to_rsel(a_ucb_z)
    with np.errstate(invalid="ignore"):
        ucb_rsel_aplcb = to_rsel(aplcb_ucb_z)            # NaN-preserving

    def nan_to_none(arr):
        return [None if not np.isfinite(x) else float(x) for x in arr]

    return {
        "seed": int(split_seed),
        "p_hat": [float(x) for x in p_hat],
        "p_lcb": [float(x) for x in p_LCB],
        "ucb_rsel": {
            "ours": nan_to_none(ucb_rsel_ours),
            "a_pi_min": nan_to_none(ucb_rsel_a),
            "a_plcb": nan_to_none(ucb_rsel_aplcb),
        },
        "cert": {
            "ours": [bool(x) for x in ours_cert],
            "a_pi_min": [bool(x) for x in a_cert],
            "a_plcb": [bool(x) for x in aplcb_cert],
        },
        # scalar cross-checks vs Table IV
        "p_acc_cert": {
            "ours": float(p_hat[ours_cert].max()) if ours_cert.any() else 0.0,
            "a_pi_min": float(p_hat[a_cert].max()) if a_cert.any() else 0.0,
            "a_plcb": float(p_hat[aplcb_cert].max()) if aplcb_cert.any() else 0.0,
        },
        "n_certified": {
            "ours": int(ours_cert.sum()),
            "a_pi_min": int(a_cert.sum()),
            "a_plcb": int(aplcb_cert.sum()),
        },
    }


def _nanmedian_col(vals: list[list]) -> list:
    """Column-wise median over seeds, ignoring None; None if all-None."""
    arr = np.array([[np.nan if v is None else v for v in row] for row in vals],
                   dtype=np.float64)
    out = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        col = col[np.isfinite(col)]
        out.append(None if col.size == 0 else float(np.median(col)))
    return out


def aggregate_pairs(per_seed: list[dict], Lambda, T) -> list[dict]:
    """Per grid-pair median over seeds + certified fraction per method."""
    m_tau = len(T)
    m = len(Lambda) * m_tau
    methods = ["ours", "a_pi_min", "a_plcb"]

    p_hat_med = np.median(np.array([s["p_hat"] for s in per_seed]), axis=0)
    p_lcb_med = np.median(np.array([s["p_lcb"] for s in per_seed]), axis=0)
    ucb_med = {mth: _nanmedian_col([s["ucb_rsel"][mth] for s in per_seed])
               for mth in methods}
    cert_frac = {mth: np.mean(np.array([s["cert"][mth] for s in per_seed]), axis=0)
                 for mth in methods}

    pairs = []
    for k in range(m):
        li, ti = k // m_tau, k % m_tau
        pairs.append({
            "k": k,
            "lambda_idx": li, "tau_idx": ti,
            "lambda": float(Lambda[li]), "tau": float(T[ti]),
            "p_acc": float(p_hat_med[k]),
            "p_lcb": float(p_lcb_med[k]),
            "ucb_rsel": {mth: ucb_med[mth][k] for mth in methods},
            "cert_frac": {mth: float(cert_frac[mth][k]) for mth in methods},
            # majority rule: certified if a strict majority of seeds certify it
            "cert": {mth: bool(cert_frac[mth][k] >= 0.5) for mth in methods},
        })
    return pairs


def per_seed_checks(per_seed: list[dict], m_tau: int) -> dict:
    """Verify the claims the caption makes, on EVERY seed (no averaging).

    - A(pi_min) certifies nothing on every seed;
    - Ours's certified set count >= A(p_LCB)'s on every seed;
    - strict dominance: at every grid pair where BOTH certify, Ours's risk-UCB
      is <= A(p_LCB)'s (the variance-adaptive tightness, per seed).
    """
    apim_zero = all(s["n_certified"]["a_pi_min"] == 0 for s in per_seed)
    ours_ge = all(s["n_certified"]["ours"] >= s["n_certified"]["a_plcb"]
                  for s in per_seed)
    strict = True
    for s in per_seed:
        uo, ua = s["ucb_rsel"]["ours"], s["ucb_rsel"]["a_plcb"]
        co, ca = s["cert"]["ours"], s["cert"]["a_plcb"]
        for k in range(len(uo)):
            if co[k] and ca[k] and uo[k] is not None and ua[k] is not None:
                if uo[k] > ua[k] + 1e-12:
                    strict = False
    return {
        "n_seeds": len(per_seed),
        "all_seeds_a_pi_min_certifies_zero": bool(apim_zero),
        "all_seeds_ours_count_ge_a_plcb": bool(ours_ge),
        "all_seeds_strict_dominance_on_shared_pairs": bool(strict),
    }


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
            print(f"[frontier] SKIPPING {model_name}: cache not found at {logits_path}",
                  flush=True)
            continue
        Y_full = np.load(labels_path).astype(np.int64)
        L_full = np.load(logits_path).astype(np.float64)
        top1 = float((L_full.argmax(axis=1) == Y_full).mean())
        print(f"\n[frontier] {model_name}: n={Y_full.shape[0]}, top-1 = {top1:.4f}",
              flush=True)
        del Y_full, L_full

        per_seed = []
        for seed_i in range(n_seeds):
            split_seed = 42 + seed_i
            t0 = time.time()
            res = per_seed_frontier(
                logits_path, labels_path,
                split_seed=split_seed, n_cert=n_cert, n_tune=n_tune, n_test=n_test,
                Lambda=Lambda, T=T, alpha=alpha, pi_min=pi_min, delta=delta,
                c=c, V=V, B=B, n_classes=n_classes,
            )
            per_seed.append(res)
            print(f"  seed {split_seed} in {time.time()-t0:.1f}s "
                  f"ncert ours={res['n_certified']['ours']} "
                  f"A(pl)={res['n_certified']['a_plcb']} "
                  f"A(pi)={res['n_certified']['a_pi_min']}", flush=True)

        pairs = aggregate_pairs(per_seed, Lambda, T)
        # median-across-seeds scalar cross-checks vs Table IV
        sc = {mth: {
            "median_p_acc_cert": float(np.median([s["p_acc_cert"][mth] for s in per_seed])),
            "median_n_certified": float(np.median([s["n_certified"][mth] for s in per_seed])),
        } for mth in ["ours", "a_pi_min", "a_plcb"]}
        results_per_model[model_name] = {
            "top1": top1,
            "pairs": pairs,
            "scalar_check": sc,
            "per_seed_checks": per_seed_checks(per_seed, len(T)),
        }

    return {
        "experiment": "analysis_cert_frontier",
        "purpose": ("Per-grid-pair certified risk-UCB behind Fig 5 (frontier). "
                    "Companion to Table IV; reveals strict dominance the scalar hides."),
        "config": {
            "n_seeds": n_seeds, "n_cert": n_cert, "n_tune": n_tune, "n_test": n_test,
            "alpha": alpha, "pi_min": pi_min, "delta": delta,
            "c": c, "V": V, "B": B, "n_classes": n_classes,
            "Lambda": Lambda.tolist(), "T": T.tolist(),
        },
        "models": results_per_model,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-cert", type=int, default=33000)
    ap.add_argument("--models", type=str, default="RN50,RN101,RN152")
    ap.add_argument("--out-tag", type=str, default="")
    args = ap.parse_args(argv)

    name_map = {"RN50": "ResNet-50 V2", "RN101": "ResNet-101 V2", "RN152": "ResNet-152 V2"}
    selected = [name_map[m.strip()] for m in args.models.split(",") if m.strip()]
    caches = {k: v for k, v in DEFAULT_MODEL_CACHES.items() if k in selected}

    summary = run(caches, n_seeds=args.n_seeds, n_cert=args.n_cert)

    tag = f".{args.out_tag}" if args.out_tag else ""
    out_json = OUT_DIR / f"certified_decision_frontier{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[frontier] Wrote {out_json}")

    # quick console cross-check
    print("\n[frontier] scalar cross-check (median across seeds) vs Table IV:")
    for name, mres in summary["models"].items():
        sc = mres["scalar_check"]
        print(f"  {name}: p_acc_cert ours={sc['ours']['median_p_acc_cert']:.3f} "
              f"A(pl)={sc['a_plcb']['median_p_acc_cert']:.3f} | "
              f"#cert ours={sc['ours']['median_n_certified']:.0f} "
              f"A(pl)={sc['a_plcb']['median_n_certified']:.0f} "
              f"A(pi)={sc['a_pi_min']['median_n_certified']:.0f}")
    print("\n[frontier] per-seed checks (caption claims, must all be True):")
    for name, mres in summary["models"].items():
        c = mres["per_seed_checks"]
        print(f"  {name}: A(pi_min)=0 all seeds={c['all_seeds_a_pi_min_certifies_zero']}, "
              f"ours_count>=A(pl) all={c['all_seeds_ours_count_ge_a_plcb']}, "
              f"strict_dom_on_shared all={c['all_seeds_strict_dominance_on_shared_pairs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
