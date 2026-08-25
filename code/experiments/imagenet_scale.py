"""B2/B3 ImageNet-scale experiments.

Per EXPERIMENT_PLAN.md M3/M4:
- B2 PRIMARY (PC2a): π_min=0.01, n_cert=33000 → expect Ours ≤ 0.75× Baseline A on low-π̂_acc
- B3 SECONDARY (PC2b): π_min=0.02, n_cert=25000 → expect Ours ≈ Baseline B (tied)

Preferred path (used by configs/imagenet_*_real.yaml): real ImageNet val logits
and labels precomputed by `experiments/imagenet_compute_logits.py` (ResNet-50
V2, top-1 = 0.8084). The headline PC2a 0.118 ratio in `results/imagenet_primary_real/`
comes from this real-data path.

Legacy path (`_make_synthetic_imagenet_data`): synthetic-realistic logits +
synthetic labels at 50k × 1000 scale, retained for back-compat with the
non-`_real.yaml` configs only. New runs should use the real-data path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from selective_crc import (  # noqa: E402
    certify_grid,
    build_lambda_tau_grid,
    three_split_indices,
    baseline_a_range_hoeffding,
    baseline_b_accepted_bernstein,
)
from selective_crc.certify import INFEASIBLE  # noqa: E402

import cifar100  # for _compute_contains_and_size and construct_acceptance


def _load_real_imagenet_data(
    logits_path: str, labels_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load real ImageNet val logits + labels precomputed by
    experiments/imagenet_compute_logits.py via ResNet-50 ImageNet-V2 weights.
    """
    import os
    if not os.path.exists(logits_path):
        raise FileNotFoundError(
            f"Real ImageNet logits not found at {logits_path}. "
            f"Run `python experiments/imagenet_compute_logits.py` first."
        )
    Y = np.load(labels_path).astype(np.int64)
    logits = np.load(logits_path).astype(np.float32)
    assert logits.shape[0] == Y.shape[0], "logits/labels row mismatch"
    print(f"[imagenet] loaded REAL ImageNet val: logits {logits.shape}, labels {Y.shape}")
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    max_softmax = probs.max(axis=1)
    acc = float((logits.argmax(axis=1) == Y).mean())
    print(f"[imagenet] real top-1 = {acc:.4f}, max-softmax: "
          f"min={max_softmax.min():.3f}, "
          f"25%={np.percentile(max_softmax, 25):.3f}, "
          f"median={np.median(max_softmax):.3f}, "
          f"75%={np.percentile(max_softmax, 75):.3f}, "
          f"max={max_softmax.max():.3f}")
    return Y, logits


def _make_synthetic_imagenet_data(
    n_total: int, n_classes: int, accuracy_target: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(LEGACY) Generate synthetic ImageNet-scale (Y, logits) for sanity testing.

    Real ImageNet pipeline should use `_load_real_imagenet_data` instead.
    """
    rng = np.random.default_rng(seed)
    print(f"[imagenet] generating SYNTHETIC data (NOT real ImageNet): n={n_total}, n_classes={n_classes}, target acc={accuracy_target}")
    Y = rng.integers(0, n_classes, size=n_total)
    logits = rng.normal(0, 1.0, size=(n_total, n_classes))
    correct_mask = rng.uniform(0, 1, size=n_total) < accuracy_target
    bump_amount = np.where(
        correct_mask,
        rng.uniform(5.0, 9.0, size=n_total),
        rng.uniform(4.0, 8.0, size=n_total),
    )
    bump_class = np.where(
        correct_mask, Y, rng.integers(0, n_classes, size=n_total)
    )
    wrong_idx = np.where(~correct_mask)[0]
    for i in wrong_idx:
        while bump_class[i] == Y[i]:
            bump_class[i] = rng.integers(0, n_classes)
    rows = np.arange(n_total)
    logits[rows, bump_class] += bump_amount
    achieved_acc = float((logits.argmax(axis=1) == Y).mean())
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    max_softmax = probs.max(axis=1)
    print(f"[imagenet] synthetic top-1 = {achieved_acc:.3f}; "
          f"max-softmax median = {np.median(max_softmax):.3f}, "
          f"25%/75% = ({np.percentile(max_softmax, 25):.3f}, {np.percentile(max_softmax, 75):.3f})")
    return Y, logits


def run_one_seed(cfg, split_seed, Y_full, logits_all):
    n_total = len(Y_full)
    requested_total = cfg["data"]["n_cert"] + cfg["data"]["n_tune"] + cfg["data"]["n_test"]

    sub_rng = np.random.default_rng(split_seed)
    subset_idx = sub_rng.choice(n_total, size=requested_total, replace=False)
    Y_all = Y_full[subset_idx]
    logits_used = logits_all[subset_idx]

    tune_idx, cert_idx, test_idx = three_split_indices(
        len(Y_all), n_tune=cfg["data"]["n_tune"], n_cert=cfg["data"]["n_cert"], seed=split_seed
    )

    Lambda = np.array(cfg["grid"]["Lambda"], dtype=np.float64)
    T = np.array(cfg["grid"]["T"], dtype=np.float64)
    Lambda_flat, T_flat = build_lambda_tau_grid(Lambda, T)
    m_lambda, m_tau = len(Lambda), len(T)
    m = m_lambda * m_tau
    n_classes = cfg["data"]["n_classes"]

    Y_cert = Y_all[cert_idx]
    logits_cert = logits_used[cert_idx]
    contains_Y, set_size = cifar100._compute_contains_and_size(
        logits_cert, Y_cert, Lambda, n_classes
    )
    L_lambda = (~contains_Y).astype(np.float64)
    L_grid = np.repeat(L_lambda, m_tau, axis=1)
    A_lambda_tau = cifar100.construct_acceptance(logits_cert, T)
    A_grid = np.tile(A_lambda_tau, (1, m_lambda))
    set_size_safe = np.maximum(set_size, 1)
    v_lambda = contains_Y.astype(np.float64) / set_size_safe.astype(np.float64)
    v_grid = np.repeat(v_lambda, m_tau, axis=1)

    result = certify_grid(
        L_grid, A_grid, v_grid,
        alpha=cfg["parameters"]["alpha"],
        pi_min=cfg["parameters"]["pi_min"],
        delta=cfg["parameters"]["delta"],
        c=cfg["parameters"]["c"],
        V=cfg["parameters"]["V"],
        B=cfg["parameters"]["B"],
        check_sample_size=cfg.get("check_sample_size", True),
    )

    baseline_a_width = baseline_a_range_hoeffding(
        L_grid, A_grid,
        alpha=cfg["parameters"]["alpha"],
        pi_min=cfg["parameters"]["pi_min"],
        delta=cfg["parameters"]["delta"],
        B=cfg["parameters"]["B"],
    )
    baseline_b_width = baseline_b_accepted_bernstein(
        L_grid, A_grid,
        alpha=cfg["parameters"]["alpha"],
        delta=cfg["parameters"]["delta"],
        B=cfg["parameters"]["B"],
    )

    # Test split for held-out R_sel
    Y_test = Y_all[test_idx]
    logits_test = logits_used[test_idx]
    contains_Y_test, _ = cifar100._compute_contains_and_size(
        logits_test, Y_test, Lambda, n_classes
    )
    L_test_lambda = (~contains_Y_test).astype(np.float64)
    L_test_grid = np.repeat(L_test_lambda, m_tau, axis=1)
    A_test_lambda_tau = cifar100.construct_acceptance(logits_test, T)
    A_test_grid = np.tile(A_test_lambda_tau, (1, m_lambda))
    p_acc_test = A_test_grid.mean(axis=0)
    sum_AL_test = (A_test_grid * L_test_grid).sum(axis=0)
    sum_A_test = A_test_grid.sum(axis=0)
    R_sel_test = np.divide(
        sum_AL_test, sum_A_test, out=np.full_like(sum_AL_test, np.inf), where=sum_A_test > 0
    )

    # ours per-pair "R_sel UCB margin" = MP radius on E[Z] divided by p_hat
    ours_radius_z = result.eb_per_pair - result.z_bar
    p_hat = result.s_count / cfg["data"]["n_cert"]
    p_hat_safe = np.maximum(p_hat, 1e-9)
    ours_r_margin = ours_radius_z / p_hat_safe

    # Low-acceptance regime mask: p̂_acc ≤ 2·π_min
    low_acc_mask = p_hat <= 2.0 * cfg["parameters"]["pi_min"]
    n_low = int(low_acc_mask.sum())

    return {
        "split_seed": split_seed,
        "is_infeasible": bool(result.is_infeasible),
        "n_certified": int(result.n_certified),
        "selected_k": None if result.is_infeasible else int(result.selected),
        "selected_lambda": float(Lambda_flat[result.selected]) if not result.is_infeasible else None,
        "selected_tau": float(T_flat[result.selected]) if not result.is_infeasible else None,
        "test_R_sel_selected": float(R_sel_test[result.selected]) if not result.is_infeasible and R_sel_test[result.selected] != np.inf else None,
        "test_p_acc_selected": float(p_acc_test[result.selected]) if not result.is_infeasible else None,
        "vio_r_selected": bool(R_sel_test[result.selected] > cfg["parameters"]["alpha"]) if not result.is_infeasible and R_sel_test[result.selected] != np.inf else None,
        "vio_p_selected": bool(p_acc_test[result.selected] < cfg["parameters"]["pi_min"]) if not result.is_infeasible else None,
        # ratios
        "ours_r_margin": ours_r_margin.tolist(),
        "baseline_a_width": baseline_a_width.tolist(),
        "baseline_b_width": [None if np.isnan(x) else float(x) for x in baseline_b_width],
        "p_hat": p_hat.tolist(),
        "low_acc_pair_count": n_low,
    }


def main(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text())
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load (Y, logits) once: prefer real ImageNet, fall back to synthetic if missing.
    source = cfg["data"].get("dataset", "imagenet_synthetic")
    if source == "imagenet_real":
        Y_full, logits_all = _load_real_imagenet_data(
            logits_path=cfg["data"]["logits_path"],
            labels_path=cfg["data"]["labels_path"],
        )
    else:
        Y_full, logits_all = _make_synthetic_imagenet_data(
            n_total=cfg["data"]["n_total"],
            n_classes=cfg["data"]["n_classes"],
            accuracy_target=cfg["data"]["synthetic_accuracy_target"],
            seed=cfg["data"]["logits_seed"],
        )

    n_seeds = cfg["seeds"]["n_seeds"]
    per_seed = []
    print(f"\n[{cfg['experiment']['name']}] running {n_seeds} seeds at n_cert={cfg['data']['n_cert']}, π_min={cfg['parameters']['pi_min']}, α={cfg['parameters']['alpha']}")
    for i in range(n_seeds):
        split_seed = 42 + i
        print(f"\n[seed {i+1}/{n_seeds}] split_seed={split_seed}", flush=True)
        out = run_one_seed(cfg, split_seed, Y_full, logits_all)
        per_seed.append(out)
        print(f"  infeasible={out['is_infeasible']}, |Ĝ|={out['n_certified']}, "
              f"selected_λ={out['selected_lambda']}, selected_τ={out['selected_tau']}, "
              f"vio_r={out['vio_r_selected']} vio_p={out['vio_p_selected']}")

    feasible = [s for s in per_seed if not s["is_infeasible"]]
    n_feasible = len(feasible)
    n_vio_r = sum(1 for s in feasible if s["vio_r_selected"])
    n_vio_p = sum(1 for s in feasible if s["vio_p_selected"])
    n_vio_joint = sum(1 for s in feasible if s["vio_r_selected"] or s["vio_p_selected"])

    # Per-pair ratios, filtered by low-acceptance regime (p_hat ≤ 2·π_min) for PC2a
    pi_min = cfg["parameters"]["pi_min"]
    ratios_a_low = []
    ratios_a_all = []
    ratios_b_all = []
    for s in per_seed:
        for k in range(len(s["ours_r_margin"])):
            ours = s["ours_r_margin"][k]
            a = s["baseline_a_width"][k]
            b = s["baseline_b_width"][k]
            p = s["p_hat"][k]
            if ours > 0 and a > 0:
                ratio_a = ours / a
                ratios_a_all.append(ratio_a)
                if p <= 2.0 * pi_min:
                    ratios_a_low.append(ratio_a)
            if b is not None and b > 0:
                ratios_b_all.append(ours / b)

    summary = {
        "experiment": cfg["experiment"]["name"],
        "n_seeds": n_seeds,
        "n_feasible": n_feasible,
        "vio_r_rate": n_vio_r / n_feasible if n_feasible > 0 else None,
        "vio_p_rate": n_vio_p / n_feasible if n_feasible > 0 else None,
        "vio_joint_rate": n_vio_joint / n_feasible if n_feasible > 0 else None,
        "median_ratio_ours_a_LOW_ACCEPTANCE": float(np.median(ratios_a_low)) if ratios_a_low else None,
        "median_ratio_ours_a_ALL": float(np.median(ratios_a_all)) if ratios_a_all else None,
        "median_ratio_ours_b_ALL": float(np.median(ratios_b_all)) if ratios_b_all else None,
        "n_ratio_samples_low_acc": len(ratios_a_low),
        "n_ratio_samples_all": len(ratios_a_all),
        "config": cfg["parameters"],
        "per_seed": per_seed,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[{cfg['experiment']['name']}] wrote {json_path}")
    print(f"  n_feasible = {n_feasible}/{n_seeds}")
    print(f"  vio_joint_rate = {summary['vio_joint_rate']}")
    print(f"  MEDIAN_RATIO_OURS_A (LOW-ACC, p̂ ≤ 2π_min={2*pi_min}) = {summary['median_ratio_ours_a_LOW_ACCEPTANCE']}")
    print(f"  MEDIAN_RATIO_OURS_A (ALL pairs) = {summary['median_ratio_ours_a_ALL']}")
    print(f"  MEDIAN_RATIO_OURS_B (ALL pairs) = {summary['median_ratio_ours_b_ALL']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    main(args.config)
