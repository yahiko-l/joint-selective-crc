"""Hyperparameter-sensitivity star design and small-calibration regime.

One-at-a-time sweeps around the anchor shared by the released A-series
diagnostics, (alpha, delta, pi_min, m, n_cert) = (0.05, 0.05, 0.02, 35, 25000),
on cached ImageNet ResNet-50 V2 logits:

- pi_min axis : replicates A1 seed-for-seed (n_tune = n_test = 12500) and
  asserts feasibility / held-out violation counts against the committed
  A1_imagenet_resnet50v2_pi_min_sweep.json.
- m axis      : replicates A9 seed-for-seed (same protocol, A9's equispaced
  lambda grids) and asserts against A9_imagenet_resnet50v2_grid_size_sweep.json.
- n_cert axis : replicates A8 seed-for-seed (n_tune = n_test = 5000) and
  asserts against A8_imagenet_resnet50v2_n_cert_sweep.json.
- alpha axis  : NEW, {0.02, 0.05, 0.10, 0.15, 0.20} on the A1 protocol.
- delta axis  : NEW, {0.01, 0.05, 0.10, 0.20} on the A1 protocol.
- small-n     : NEW, n_cert in {250, 500, 750, 1000} x alpha in {0.05, 0.20}
  on the A8 protocol with 30 seeds (feasibility boundary + held-out validity
  of every emitted certificate).

Every cell records: feasible runs, held-out joint violations among feasible
runs, and medians over feasible runs of |G_hat|, the deployed pair's U_LCB,
and the deployed pair's nominal risk half-width (EB - Z_bar) / p_hat.

CPU-only recomputation from cached logits; no new model inference.
Output: results/ablation_supplement/A18_imagenet_resnet50v2_sensitivity_star.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.selective_crc import certify_grid, three_split_indices
from experiments import cifar100


import os
# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ablation_supplement"

LOGITS = f"{DATA_ROOT}/imagenet_data/val_logits.npy"
LABELS = f"{DATA_ROOT}/imagenet_data/val_labels.npy"

# Anchor (shared by A1 / A8 / A9)
ALPHA0, DELTA0, PIMIN0 = 0.05, 0.05, 0.02
C0, V0, B0 = 0.1, 1.0, 1.0
LAMBDA0 = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
TAU0 = [0.5, 0.6, 0.7, 0.8, 0.9]
N_CLASSES = 1000

# Axis values
PIMIN_AXIS = [0.005, 0.01, 0.02, 0.05, 0.10]           # = A1
NCERT_AXIS = [2000, 5000, 10000, 15000, 25000]         # = A8
MLAMBDA_AXIS = [3, 7, 14, 16]                          # = A9 (m = 15/35/70/80)
ALPHA_AXIS = [0.02, 0.05, 0.10, 0.15, 0.20]            # NEW
DELTA_AXIS = [0.01, 0.05, 0.10, 0.20]                  # NEW
SMALL_N = [250, 500, 750, 1000]                        # NEW
SMALL_N_ALPHAS = [0.05, 0.20]                          # NEW

N_SEEDS_STAR = 10
N_SEEDS_SMALL = 30

# Lambda-grid construction of the released grid-size diagnostic, reproduced verbatim
A9_LAMBDA_POOL = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5,
                  0.7, 1.0, 1.5, 2.0, 3.0, 5.0]


def build_seed_data(logits_full, labels_full, split_seed, n_cert, n_tune, n_test,
                    Lambda, T):
    """Reproduce the A-series per-seed data pipeline exactly."""
    rng = np.random.default_rng(split_seed)
    req_total = n_cert + n_tune + n_test
    subset = rng.choice(len(labels_full), size=req_total, replace=False)
    Y_all = labels_full[subset]
    logits_used = logits_full[subset]
    tune_idx, cert_idx, test_idx = three_split_indices(
        len(Y_all), n_tune=n_tune, n_cert=n_cert, seed=split_seed
    )
    m_lambda, m_tau = len(Lambda), len(T)

    Y_cert = Y_all[cert_idx]
    logits_cert = logits_used[cert_idx]
    contains_Y, set_size = cifar100._compute_contains_and_size(
        logits_cert, Y_cert, Lambda, N_CLASSES
    )
    L_lambda = (~contains_Y).astype(np.float64)
    L_grid = np.repeat(L_lambda, m_tau, axis=1)
    A_lt = cifar100.construct_acceptance(logits_cert, T)
    A_grid = np.tile(A_lt, (1, m_lambda))
    set_size_safe = np.maximum(set_size, 1)
    v_lambda = contains_Y.astype(np.float64) / set_size_safe.astype(np.float64)
    v_grid = np.repeat(v_lambda, m_tau, axis=1)

    Y_test = Y_all[test_idx]
    logits_test = logits_used[test_idx]
    contains_Y_t, _ = cifar100._compute_contains_and_size(
        logits_test, Y_test, Lambda, N_CLASSES
    )
    L_test = (~contains_Y_t).astype(np.float64)
    L_t_grid = np.repeat(L_test, m_tau, axis=1)
    A_t_lt = cifar100.construct_acceptance(logits_test, T)
    A_t_grid = np.tile(A_t_lt, (1, m_lambda))
    p_acc_t = A_t_grid.mean(axis=0)
    sum_AL = (A_t_grid * L_t_grid).sum(axis=0)
    sum_A = A_t_grid.sum(axis=0)
    R_t = np.divide(sum_AL, sum_A, out=np.full_like(sum_AL, np.inf),
                    where=sum_A > 0)
    return L_grid, A_grid, v_grid, R_t, p_acc_t


def run_cell(L_grid, A_grid, v_grid, R_t, p_acc_t, *, alpha, pi_min, delta):
    """One certify_grid call + held-out check + deployed-pair metrics."""
    n_cert = L_grid.shape[0]
    res = certify_grid(L_grid, A_grid, v_grid, alpha=alpha, pi_min=pi_min,
                       delta=delta, c=C0, V=V0, B=B0, check_sample_size=False)
    if res.is_infeasible:
        return {"feasible": False}
    k = int(res.selected)
    p_hat = res.s_count / n_cert
    width_r = float((res.eb_per_pair[k] - res.z_bar[k])
                    / max(p_hat[k], 1e-9))
    return {
        "feasible": True,
        "vio_joint": bool(R_t[k] > alpha) or bool(p_acc_t[k] < pi_min),
        "n_certified": int(res.n_certified),
        "u_lcb": float(res.u_lcb_per_pair[k]),
        "width_r": width_r,
        "selected_k": k,
    }


def aggregate(cells, n_seeds):
    feas = [c for c in cells if c["feasible"]]
    med = lambda key: (float(np.median([c[key] for c in feas])) if feas else None)
    return {
        "n_seeds": n_seeds,
        "n_feasible": len(feas),
        "n_vio_joint": sum(c["vio_joint"] for c in feas),
        "median_n_certified": med("n_certified"),
        "median_u_lcb": med("u_lcb"),
        "median_width_r": med("width_r"),
        "per_seed": cells,
    }


def main():
    logits_full = np.load(LOGITS).astype(np.float64)
    labels_full = np.load(LABELS).astype(np.int64)
    print(f"logits {logits_full.shape}, top-1 = "
          f"{(logits_full.argmax(1) == labels_full).mean():.4f}")

    out = {"axes": {}, "small_n": [], "asserts": {}}

    # ---- Protocol P1 (A1/A9-style): n_cert=25000, n_tune=n_test=12500 -----
    # Anchor-grid data, one build per seed, shared by the pi_min/alpha/delta axes.
    p1_data = []
    for i in range(N_SEEDS_STAR):
        p1_data.append(build_seed_data(
            logits_full, labels_full, 42 + i, 25000, 12500, 12500,
            np.array(LAMBDA0), np.array(TAU0)))
        print(f"[P1] seed {42+i} data built")

    # pi_min axis (= A1 replication)
    rows = []
    for pi_min in PIMIN_AXIS:
        cells = [run_cell(*d, alpha=ALPHA0, pi_min=pi_min, delta=DELTA0)
                 for d in p1_data]
        row = {"pi_min": pi_min, **aggregate(cells, N_SEEDS_STAR)}
        rows.append(row)
        print(f"[pi_min={pi_min}] feas={row['n_feasible']}/{N_SEEDS_STAR} "
              f"vio={row['n_vio_joint']} |G|={row['median_n_certified']} "
              f"U_LCB={row['median_u_lcb']} w_r={row['median_width_r']}")
    out["axes"]["pi_min"] = rows

    a1 = json.loads((OUT / "A1_imagenet_resnet50v2_pi_min_sweep.json").read_text())
    for row, ref in zip(rows, a1["sweep"]):
        assert row["pi_min"] == ref["pi_min"]
        assert row["n_feasible"] == ref["n_feasible"], (row, ref)
        ref_vio = 0 if ref["vio_joint_rate"] in (None, 0, 0.0) else round(
            ref["vio_joint_rate"] * ref["n_feasible"])
        assert row["n_vio_joint"] == ref_vio, (row, ref)
        # |G_hat| per seed must match A1's recorded n_certified
        for cell, ps in zip(row["per_seed"], ref["per_seed"]):
            if cell["feasible"]:
                assert cell["n_certified"] == ps["n_certified"], (cell, ps)
    out["asserts"]["A1_pi_min"] = "pass"
    print("[assert] pi_min axis == A1: pass")

    # alpha axis (NEW)
    rows = []
    for alpha in ALPHA_AXIS:
        cells = [run_cell(*d, alpha=alpha, pi_min=PIMIN0, delta=DELTA0)
                 for d in p1_data]
        row = {"alpha": alpha, **aggregate(cells, N_SEEDS_STAR)}
        rows.append(row)
        print(f"[alpha={alpha}] feas={row['n_feasible']}/{N_SEEDS_STAR} "
              f"vio={row['n_vio_joint']} |G|={row['median_n_certified']} "
              f"U_LCB={row['median_u_lcb']} w_r={row['median_width_r']}")
    out["axes"]["alpha"] = rows

    # delta axis (NEW)
    rows = []
    for delta in DELTA_AXIS:
        cells = [run_cell(*d, alpha=ALPHA0, pi_min=PIMIN0, delta=delta)
                 for d in p1_data]
        row = {"delta": delta, **aggregate(cells, N_SEEDS_STAR)}
        rows.append(row)
        print(f"[delta={delta}] feas={row['n_feasible']}/{N_SEEDS_STAR} "
              f"vio={row['n_vio_joint']} |G|={row['median_n_certified']} "
              f"U_LCB={row['median_u_lcb']} w_r={row['median_width_r']}")
    out["axes"]["delta"] = rows

    # Consistency: the anchor cell appears on all three P1 axes.
    anchor_pi = next(r for r in out["axes"]["pi_min"] if r["pi_min"] == PIMIN0)
    anchor_al = next(r for r in out["axes"]["alpha"] if r["alpha"] == ALPHA0)
    anchor_de = next(r for r in out["axes"]["delta"] if r["delta"] == DELTA0)
    for a, b in [(anchor_pi, anchor_al), (anchor_pi, anchor_de)]:
        assert (a["n_feasible"], a["n_vio_joint"], a["median_n_certified"],
                a["median_u_lcb"], a["median_width_r"]) == \
               (b["n_feasible"], b["n_vio_joint"], b["median_n_certified"],
                b["median_u_lcb"], b["median_width_r"])
    out["asserts"]["anchor_consistency"] = "pass"
    print("[assert] anchor cell identical across P1 axes: pass")
    del p1_data

    # m axis (= A9 replication, A9's own lambda grids)
    rows = []
    for m_l in MLAMBDA_AXIS:
        Lambda = np.array(sorted(np.linspace(
            0.001, A9_LAMBDA_POOL[min(m_l - 1, len(A9_LAMBDA_POOL) - 1)],
            m_l).tolist()))
        cells = []
        for i in range(N_SEEDS_STAR):
            d = build_seed_data(logits_full, labels_full, 42 + i,
                                25000, 12500, 12500, Lambda, np.array(TAU0))
            cells.append(run_cell(*d, alpha=ALPHA0, pi_min=PIMIN0, delta=DELTA0))
        row = {"m_lambda": m_l, "m_total": m_l * len(TAU0),
               **aggregate(cells, N_SEEDS_STAR)}
        rows.append(row)
        print(f"[m={row['m_total']}] feas={row['n_feasible']}/{N_SEEDS_STAR} "
              f"vio={row['n_vio_joint']} |G|={row['median_n_certified']} "
              f"U_LCB={row['median_u_lcb']} w_r={row['median_width_r']}")
    out["axes"]["m"] = rows

    a9 = json.loads((OUT / "A9_imagenet_resnet50v2_grid_size_sweep.json").read_text())
    for row, ref in zip(rows, a9["sweep"]):
        assert row["m_total"] == ref["m_total"]
        assert row["n_feasible"] == ref["n_feasible"], (row, ref)
        assert row["n_vio_joint"] == ref["n_vio_joint"], (row, ref)
    out["asserts"]["A9_m"] = "pass"
    print("[assert] m axis == A9: pass")

    # ---- Protocol P2 (A8-style): n_tune = n_test = 5000 -------------------
    # n_cert axis (= A8 replication)
    rows = []
    for n_cert in NCERT_AXIS:
        cells = []
        for i in range(N_SEEDS_STAR):
            d = build_seed_data(logits_full, labels_full, 42 + i,
                                n_cert, 5000, 5000,
                                np.array(LAMBDA0), np.array(TAU0))
            cells.append(run_cell(*d, alpha=ALPHA0, pi_min=PIMIN0, delta=DELTA0))
        row = {"n_cert": n_cert, **aggregate(cells, N_SEEDS_STAR)}
        rows.append(row)
        print(f"[n_cert={n_cert}] feas={row['n_feasible']}/{N_SEEDS_STAR} "
              f"vio={row['n_vio_joint']} |G|={row['median_n_certified']} "
              f"U_LCB={row['median_u_lcb']} w_r={row['median_width_r']}")
    out["axes"]["n_cert"] = rows

    a8 = json.loads((OUT / "A8_imagenet_resnet50v2_n_cert_sweep.json").read_text())
    for row, ref in zip(rows, a8["sweep"]):
        assert row["n_cert"] == ref["n_cert"]
        assert row["n_feasible"] == ref["n_feasible"], (row, ref)
        assert row["n_vio_joint"] == ref["n_vio_joint"], (row, ref)
    out["asserts"]["A8_n_cert"] = "pass"
    print("[assert] n_cert axis == A8: pass")

    # Small-calibration block (NEW): n_cert < 1001, 30 seeds, two budgets
    for n_cert in SMALL_N:
        seed_data = []
        for i in range(N_SEEDS_SMALL):
            seed_data.append(build_seed_data(
                logits_full, labels_full, 42 + i, n_cert, 5000, 5000,
                np.array(LAMBDA0), np.array(TAU0)))
        for alpha in SMALL_N_ALPHAS:
            cells = [run_cell(*d, alpha=alpha, pi_min=PIMIN0, delta=DELTA0)
                     for d in seed_data]
            row = {"n_cert": n_cert, "alpha": alpha,
                   **aggregate(cells, N_SEEDS_SMALL)}
            out["small_n"].append(row)
            print(f"[small n={n_cert} alpha={alpha}] "
                  f"feas={row['n_feasible']}/{N_SEEDS_SMALL} "
                  f"vio={row['n_vio_joint']} |G|={row['median_n_certified']} "
                  f"U_LCB={row['median_u_lcb']} w_r={row['median_width_r']}")

    out["config"] = {
        "dataset": "imagenet_resnet50v2",
        "anchor": {"alpha": ALPHA0, "delta": DELTA0, "pi_min": PIMIN0,
                   "m": len(LAMBDA0) * len(TAU0), "n_cert": 25000},
        "c": C0, "V": V0, "B": B0,
        "Lambda_grid": LAMBDA0, "T_grid": TAU0,
        "protocol_P1": {"n_cert": 25000, "n_tune": 12500, "n_test": 12500,
                        "n_seeds": N_SEEDS_STAR, "axes": ["pi_min", "alpha", "delta", "m"]},
        "protocol_P2": {"n_tune": 5000, "n_test": 5000,
                        "axes": ["n_cert", "small_n"],
                        "n_seeds_n_cert": N_SEEDS_STAR,
                        "n_seeds_small_n": N_SEEDS_SMALL},
        "seeds": "split_seed = 42 + i",
        "metric_width_r": "(EB - Z_bar) / p_hat at the deployed pair",
        "note": "pi_min/m/n_cert axes replicate A1/A9/A8 seed-for-seed; "
                "counts asserted equal to the committed JSONs above.",
    }

    out_file = OUT / "A18_imagenet_resnet50v2_sensitivity_star.json"
    out_file.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
