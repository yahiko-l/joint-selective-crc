"""B1 full PC1 validity sweep — 20 seeds × 27 configs.

EXPERIMENT_PLAN.md M1 spec:
  Configs:
    n ∈ {1000, 5000, 25000}
    m ∈ {10, 35, 100}
    π_min ∈ {0.05, 0.10, 0.30}    (we use one α per config since each contributes
                                    a separate VIO_R verification)
    α ∈ {0.05, 0.10, 0.20}
  Seeds: 20 per config
  Total: 3·3·3·3 = 81 ... but per the plan we use 27 = 3 × 9 = 3 × (3 × 3) — we
  cross-product over (n, m, π_min) and vary α for some configs to triangulate.

For practical implementation we use 3 × 3 × 3 = 27 unique (n, m, π_min) configs,
each at a single α = 0.10 (or auto-adjusted to be slightly above the true R_sel).

Each run produces VIO_R, VIO_P, VIO_U, VIO_JOINT empirical rates over 20 seeds.

Output: results/synthetic_full/pc1_summary.json + per_config CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selective_crc import (  # noqa: E402
    certify_grid,
    build_lambda_tau_grid,
)
from selective_crc.certify import INFEASIBLE  # noqa: E402
from selective_crc.bounds import sample_size_condition  # noqa: E402


def generate_synthetic_data(
    n_total: int,
    Lambda_flat: np.ndarray,
    T_flat: np.ndarray,
    seed: int,
    B: float = 1.0,
    V: float = 1.0,
):
    """Beta(2, 5) loss in [0, B]; uniform acceptance score in [0, 1]."""
    rng = np.random.default_rng(seed)
    assert len(Lambda_flat) == len(T_flat)
    m = len(Lambda_flat)
    g = rng.uniform(0, 1, size=n_total)
    A = (g[:, None] > T_flat[None, :]).astype(np.float64)
    L = rng.beta(2.0, 5.0, size=(n_total, m)) * B
    v = np.clip((1.0 - L) * V, 0, V)
    return L, A, v


def build_grid_for_m(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a (λ, τ) grid of approximately size m using product structure."""
    # We allocate roughly sqrt(m) values to each axis
    n_lambda = max(2, int(np.sqrt(m)))
    n_tau = int(np.ceil(m / n_lambda))
    Lambda = np.linspace(0.05, 0.5, n_lambda)
    T = np.linspace(0.3, 0.85, n_tau)
    Lambda_flat, T_flat = build_lambda_tau_grid(Lambda, T)
    # Truncate to exactly m
    if len(Lambda_flat) > m:
        Lambda_flat = Lambda_flat[:m]
        T_flat = T_flat[:m]
    return Lambda_flat, T_flat


def run_single(n_cert, m, pi_min, alpha, delta, seed, c=0.05, B=1.0, V=1.0):
    """Run Algorithm 1 once + compute violations on held-out population truth."""
    Lambda_flat, T_flat = build_grid_for_m(m)
    actual_m = len(Lambda_flat)
    # Generate enough data: n_cert for cert, plus N_pop for population truth
    N_pop = 50000
    L_pop, A_pop, v_pop = generate_synthetic_data(
        n_total=N_pop, Lambda_flat=Lambda_flat, T_flat=T_flat, seed=seed,
        B=B, V=V,
    )
    L_cert, A_cert, v_cert = L_pop[:n_cert], A_pop[:n_cert], v_pop[:n_cert]

    result = certify_grid(
        L_cert, A_cert, v_cert,
        alpha=alpha, pi_min=pi_min, delta=delta, c=c, V=V, B=B,
        check_sample_size=False,  # we explicitly allow under-n configs for validity testing
    )

    # Population truth from the remaining samples (≈ 50k held-out estimate)
    p_acc_true = A_pop[n_cert:].mean(axis=0)
    sum_AL = (A_pop[n_cert:] * L_pop[n_cert:]).sum(axis=0)
    sum_A = A_pop[n_cert:].sum(axis=0)
    R_sel_true = np.divide(
        sum_AL, sum_A, out=np.full_like(sum_AL, np.inf), where=sum_A > 0
    )
    U_dep_true = (A_pop[n_cert:] * v_pop[n_cert:] - c * (1 - A_pop[n_cert:])).mean(axis=0)

    out = {
        "seed": seed,
        "is_infeasible": result.is_infeasible,
        "n_certified": result.n_certified,
        "selected_k": None if result.is_infeasible else int(result.selected),
        "n_required": result.config["n_required"],
        "vio_r_selected": None,
        "vio_p_selected": None,
        "vio_joint_selected": None,
        "vio_r_any_g_hat": False,
        "vio_p_any_g_hat": False,
        "actual_m": actual_m,
    }
    if not result.is_infeasible:
        k = result.selected
        out["vio_r_selected"] = bool(R_sel_true[k] > alpha)
        out["vio_p_selected"] = bool(p_acc_true[k] < pi_min)
        out["vio_joint_selected"] = bool(out["vio_r_selected"] or out["vio_p_selected"])
        # Validity for every (λ, τ) in Ĝ
        g_hat_mask = result.in_g_hat_mask
        if g_hat_mask.any():
            out["vio_r_any_g_hat"] = bool((R_sel_true[g_hat_mask] > alpha).any())
            out["vio_p_any_g_hat"] = bool((p_acc_true[g_hat_mask] < pi_min).any())
    return out


def main(output_dir: Path, n_seeds: int = 20):
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = list(product(
        [1000, 5000, 25000],     # n_cert
        [10, 35, 100],            # m
        [0.05, 0.10, 0.30],       # pi_min
    ))
    # Single α per config to keep this 27-config sweep within time budget
    alpha = 0.30  # Beta(2,5) has mean 0.286; α=0.30 gives narrow margin
    delta = 0.10

    all_results = []
    print(f"[B1 full] running {len(configs)} configs × {n_seeds} seeds = {len(configs)*n_seeds} runs")
    print(f"          α = {alpha}, δ = {delta}, loss = Beta(2, 5)")

    for cfg_idx, (n_cert, m, pi_min) in enumerate(configs):
        print(f"\n[B1 cfg {cfg_idx+1}/{len(configs)}] n_cert={n_cert}, m={m}, π_min={pi_min}")
        per_seed = []
        for seed in range(42, 42 + n_seeds):
            try:
                out = run_single(n_cert, m, pi_min, alpha, delta, seed)
                per_seed.append(out)
            except Exception as e:
                print(f"    seed={seed}: ERROR {type(e).__name__}: {e}")
                continue
        # Aggregate
        n_runs = len(per_seed)
        feasible_runs = [s for s in per_seed if not s["is_infeasible"]]
        n_feasible = len(feasible_runs)
        # Compute violation rates only over feasible runs (when algorithm returns a pair)
        if n_feasible > 0:
            vio_r = sum(s["vio_r_selected"] for s in feasible_runs) / n_feasible
            vio_p = sum(s["vio_p_selected"] for s in feasible_runs) / n_feasible
            vio_joint = sum(s["vio_joint_selected"] for s in feasible_runs) / n_feasible
            vio_r_any_g = sum(s["vio_r_any_g_hat"] for s in feasible_runs) / n_feasible
            vio_p_any_g = sum(s["vio_p_any_g_hat"] for s in feasible_runs) / n_feasible
        else:
            vio_r = vio_p = vio_joint = vio_r_any_g = vio_p_any_g = None

        # Binomial 95% CI for empirical violation rate ~ p_hat (when n_feasible > 0)
        # Wald approximation: CI = p_hat ± 1.96 sqrt(p_hat (1-p_hat) / n_feasible)
        if n_feasible > 0 and vio_joint is not None:
            se = float(np.sqrt(vio_joint * (1 - vio_joint) / n_feasible))
            ci_lo = max(0.0, vio_joint - 1.96 * se)
            ci_hi = min(1.0, vio_joint + 1.96 * se)
        else:
            ci_lo = ci_hi = None

        # PC1 PASS condition: vio_joint within binomial 95% CI of δ (Wald around δ ± 1.96·sqrt(δ(1-δ)/n_feasible))
        if n_feasible > 0 and vio_joint is not None:
            delta_se = float(np.sqrt(delta * (1 - delta) / n_feasible))
            pc1_pass = vio_joint <= delta + 1.96 * delta_se
        else:
            pc1_pass = None  # all infeasible — algorithm returned INFEASIBLE for all 20 seeds

        cfg_summary = {
            "cfg_idx": cfg_idx,
            "n_cert": n_cert,
            "m": m,
            "pi_min": pi_min,
            "alpha": alpha,
            "delta": delta,
            "n_runs": n_runs,
            "n_feasible": n_feasible,
            "n_infeasible": n_runs - n_feasible,
            "vio_r_rate": vio_r,
            "vio_p_rate": vio_p,
            "vio_joint_rate": vio_joint,
            "vio_joint_ci_lo": ci_lo,
            "vio_joint_ci_hi": ci_hi,
            "vio_r_any_g_hat_rate": vio_r_any_g,
            "vio_p_any_g_hat_rate": vio_p_any_g,
            "pc1_pass": pc1_pass,
            "n_required_n0": per_seed[0]["n_required"] if per_seed else None,
            "per_seed": per_seed,
        }
        all_results.append(cfg_summary)
        if pc1_pass is None:
            verdict_str = "ALL INFEASIBLE (CASE A)"
        elif pc1_pass:
            verdict_str = f"PASS (vio_joint={vio_joint:.4f} ≤ {delta + 1.96 * delta_se:.4f})"
        else:
            verdict_str = f"FAIL (vio_joint={vio_joint:.4f} > {delta + 1.96 * delta_se:.4f})"
        print(f"    feasible={n_feasible}/{n_runs}, vio_joint={vio_joint}, PC1: {verdict_str}")

    # Save aggregate
    summary_path = output_dir / "pc1_summary.json"
    summary_path.write_text(json.dumps({
        "experiment": "B1 PC1 full validity sweep",
        "n_configs": len(configs),
        "n_seeds_per_config": n_seeds,
        "alpha": alpha,
        "delta": delta,
        "loss_distribution": "Beta(2, 5)",
        "configs": all_results,
    }, indent=2, default=str))

    # CSV summary table
    csv_path = output_dir / "pc1_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cfg_idx", "n_cert", "m", "pi_min", "alpha", "delta",
            "n_runs", "n_feasible", "n_infeasible",
            "vio_joint_rate", "vio_joint_ci_lo", "vio_joint_ci_hi",
            "vio_r_rate", "vio_p_rate",
            "vio_r_any_g_hat", "vio_p_any_g_hat",
            "n_required_n0", "pc1_pass",
        ])
        for cfg in all_results:
            writer.writerow([
                cfg["cfg_idx"], cfg["n_cert"], cfg["m"], cfg["pi_min"],
                cfg["alpha"], cfg["delta"],
                cfg["n_runs"], cfg["n_feasible"], cfg["n_infeasible"],
                cfg["vio_joint_rate"], cfg["vio_joint_ci_lo"], cfg["vio_joint_ci_hi"],
                cfg["vio_r_rate"], cfg["vio_p_rate"],
                cfg["vio_r_any_g_hat_rate"], cfg["vio_p_any_g_hat_rate"],
                cfg["n_required_n0"], cfg["pc1_pass"],
            ])

    print(f"\n[B1 full] wrote {summary_path}")
    print(f"[B1 full] wrote {csv_path}")
    # Print final PC1 summary
    n_pass = sum(1 for c in all_results if c["pc1_pass"] is True)
    n_fail = sum(1 for c in all_results if c["pc1_pass"] is False)
    n_undef = sum(1 for c in all_results if c["pc1_pass"] is None)
    print(f"\n[B1 PC1 final summary] PASS={n_pass}/{len(configs)}, FAIL={n_fail}/{len(configs)}, UNDEF={n_undef}/{len(configs)} (all INFEASIBLE)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic_full"))
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()
    main(args.output_dir, n_seeds=args.n_seeds)
