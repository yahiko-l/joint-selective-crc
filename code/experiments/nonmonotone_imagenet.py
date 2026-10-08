"""ImageNet with a confidence-weighted top-k miss loss that is not monotone in k.

Design split. The 8,500 ImageNet val images of the tuning split of split seed 42 in
the headline protocol are reserved as a fixed design split: they informed the design
of this configuration and fix the softmax temperature T (negative log-likelihood),
and they never enter certification or testing. The other 41,500 images are re-split
30 times into a certification split of 33,000 and a test split of 8,500.

With p the temperature-scaled class probabilities, the candidate output is the
top-k label set C_k (k in K_GRID), s_k is the probability mass of C_k, and the
acceptance score is the largest probability g = s_1. The loss

    L_k(x, y) = s_k(x) * 1{y not in C_k(x)}

charges a miss at the confidence the system placed on the returned set; an input
missed by every C_k has a loss that grows with k, so the loss is not monotone in k
for such inputs. Its conditional mean given the confidence g is recorded on the
test splits; the loss itself does not depend on tau. The deployment value is
v = 1{y in C_k} / k with abstention cost c = 0.1. The risk budget alpha bounds this
weighted miss cost, not a miscoverage rate.

Per split the script certifies the (k, tau) grid on the certification split and
evaluates the returned pair on the test split (weighted risk, acceptance,
ordinary miscoverage, utility); on the test split it also records the binned
conditional loss E[L_k | g] and the share of inputs whose loss is not monotone in k.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selective_crc import certify_grid, three_split_indices  # noqa: E402

# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nonmonotone"
LOGITS = f"{DATA_ROOT}/imagenet_data/val_logits.npy"
LABELS = f"{DATA_ROOT}/imagenet_data/val_labels.npy"

DESIGN_SEED, N_DESIGN = 42, 8500
N_CERT, N_TEST = 33000, 8500
SEEDS = range(42, 72)
K_GRID = np.array([1, 2, 3, 5])
T_GRID = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
PARAMS = dict(alpha=0.05, pi_min=0.01, delta=0.05, c=0.1, V=1.0, B=1.0)
BINS = np.linspace(0.0, 1.0, 11)


def design_indices(n_total):
    """The tuning split of split seed 42 in the headline ImageNet protocol."""
    sub = np.random.default_rng(DESIGN_SEED).choice(n_total, size=n_total, replace=False)
    tune, _, _ = three_split_indices(n_total, n_tune=N_DESIGN, n_cert=N_CERT, seed=DESIGN_SEED)
    return np.sort(sub[tune])


def fit_temperature(z, y):
    def nll(t):
        zt = z / t
        return float(np.mean(logsumexp(zt, axis=1) - zt[np.arange(len(y)), y]))
    return float(minimize_scalar(nll, bounds=(0.05, 5.0), method="bounded").x)


def surface(top_logits, lse, rank, temp):
    """Per-input s_k, miss_k and g at temperature `temp`."""
    p_top = np.exp(top_logits / temp - lse[:, None])
    s = np.cumsum(p_top, axis=1)[:, K_GRID - 1]
    miss = rank[:, None] >= K_GRID[None, :]
    return s, miss, s[:, 0]


def grid_arrays(s, miss, g):
    """(L, A, v, miss) with columns k-major over (k, tau)."""
    m_tau = len(T_GRID)
    L = np.repeat(s * miss, m_tau, axis=1)
    A = np.tile((g[:, None] > T_GRID[None, :]).astype(np.float64), (1, len(K_GRID)))
    v = np.repeat((~miss) / K_GRID[None, :], m_tau, axis=1)
    M = np.repeat(miss.astype(np.float64), m_tau, axis=1)
    return L, A, v, M


def loss_shape(s, miss, g):
    loss = s * miss
    b = np.clip(np.digitize(g, BINS) - 1, 0, len(BINS) - 2)
    counts = np.bincount(b, minlength=len(BINS) - 1)
    cond = [[float(loss[b == j, i].mean()) if counts[j] else None for j in range(len(BINS) - 1)]
            for i in range(len(K_GRID))]
    d = np.diff(loss, axis=1)
    return {"cond_loss_by_g_bin": cond, "bin_counts": counts.tolist(),
            "share_rise_and_fall_in_k": float(((d > 1e-12).any(axis=1) & (d < -1e-12).any(axis=1)).mean()),
            "share_not_nonincreasing_in_k": float((d > 1e-12).any(axis=1).mean())}


def cp_upper(k, n):
    return 1.0 if k >= n else float(beta_dist.ppf(0.95, k + 1, n - k))


def main():
    labels = np.load(LABELS).astype(np.int64)
    logits = np.load(LOGITS).astype(np.float64)
    n_total = len(labels)
    kmax = int(K_GRID.max())
    order = np.argsort(-logits, axis=1)
    rank_all = np.argsort(order, axis=1)[np.arange(n_total), labels]
    top_all = np.take_along_axis(logits, order[:, :kmax], axis=1)

    design = design_indices(n_total)
    pool = np.setdiff1d(np.arange(n_total), design)
    assert len(pool) == N_CERT + N_TEST
    temp = fit_temperature(logits[design], labels[design])
    lse_all = logsumexp(logits / temp, axis=1)
    surf = lambda idx: surface(top_all[idx], lse_all[idx], rank_all[idx], temp)  # noqa: E731
    design_shape = loss_shape(*surf(design))
    print(f"design split: {len(design)} images, T = {temp:.4f}", flush=True)

    per_split = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(pool)
        cert, test = perm[:N_CERT], perm[N_CERT:]
        L, A, v, _ = grid_arrays(*surf(cert))
        res = certify_grid(L, A, v, check_sample_size=True, **PARAMS)
        Lt, At, vt, Mt = grid_arrays(*surf(test))
        n_acc = At.sum(axis=0)
        p_test = At.mean(axis=0)
        R_test = np.divide((At * Lt).sum(axis=0), n_acc, out=np.full(At.shape[1], np.inf), where=n_acc > 0)
        miscov_test = np.divide((At * Mt).sum(axis=0), n_acc, out=np.full(At.shape[1], np.inf), where=n_acc > 0)
        U_test = (At * vt - PARAMS["c"] * (1.0 - At)).mean(axis=0)
        row = {"seed": seed, "feasible": not res.is_infeasible, "n_certified": int(res.n_certified),
               "test_loss_shape": loss_shape(*surf(test))}
        if not res.is_infeasible:
            k = int(res.selected)
            row.update({"k": int(K_GRID[k // len(T_GRID)]), "tau": float(T_GRID[k % len(T_GRID)]),
                        "p_lcb": float(res.p_lcb_per_pair[k]), "u_lcb": float(res.u_lcb_per_pair[k]),
                        "test_R_sel": float(R_test[k]), "test_p_acc": float(p_test[k]),
                        "test_miscoverage": float(miscov_test[k]), "test_U_dep": float(U_test[k]),
                        "test_exceedance": bool(R_test[k] > PARAMS["alpha"] or p_test[k] < PARAMS["pi_min"]),
                        "test_utility_below_lcb": bool(U_test[k] < res.u_lcb_per_pair[k])})
        per_split.append(row)
        print(f"seed {seed}: feasible={row['feasible']} pair=({row.get('k')},{row.get('tau')}) "
              f"R={row.get('test_R_sel')} p={row.get('test_p_acc')} miscov={row.get('test_miscoverage')} "
              f"U={row.get('test_U_dep')} ULCB={row.get('u_lcb')}", flush=True)

    feas = [r for r in per_split if r["feasible"]]
    n_exc = sum(r["test_exceedance"] for r in feas)
    med = lambda key: float(np.median([r[key] for r in feas])) if feas else None  # noqa: E731
    shapes = [r["test_loss_shape"] for r in per_split]
    summary = {
        "splits": len(per_split), "feasible": len(feas),
        "test_exceedances": n_exc, "test_exceedance_cp95_upper_all_splits": cp_upper(n_exc, len(per_split)),
        "test_utility_below_lcb": sum(r["test_utility_below_lcb"] for r in feas),
        "median_p_lcb": med("p_lcb"), "median_test_p_acc": med("test_p_acc"),
        "median_test_R_sel": med("test_R_sel"), "median_test_miscoverage": med("test_miscoverage"),
        "median_u_lcb": med("u_lcb"), "median_test_U_dep": med("test_U_dep"),
        "selected_pairs": {f"{k},{t}": sum(r["k"] == k and r["tau"] == t for r in feas)
                           for k in K_GRID.tolist() for t in T_GRID.tolist()
                           if any(r["k"] == k and r["tau"] == t for r in feas)},
        "temperature": temp,
        "mean_test_cond_loss_by_g_bin": [[float(np.mean([sh["cond_loss_by_g_bin"][i][j] for sh in shapes
                                                          if sh["cond_loss_by_g_bin"][i][j] is not None]))
                                          if any(sh["cond_loss_by_g_bin"][i][j] is not None for sh in shapes) else None
                                          for j in range(len(BINS) - 1)] for i in range(len(K_GRID))],
        "mean_test_bin_counts": np.mean([sh["bin_counts"] for sh in shapes], axis=0).tolist(),
        "median_test_share_rise_and_fall_in_k": float(np.median([sh["share_rise_and_fall_in_k"] for sh in shapes])),
        "median_test_share_not_nonincreasing_in_k": float(np.median([sh["share_not_nonincreasing_in_k"] for sh in shapes])),
    }
    out = {"design": {"design_split": "tuning split of split seed 42 in the headline protocol",
                      "n_design": len(design), "n_cert": N_CERT, "n_test": N_TEST,
                      "seeds": [SEEDS.start, SEEDS.stop - 1], "k_grid": K_GRID.tolist(),
                      "tau_grid": T_GRID.tolist(), "params": PARAMS,
                      "loss": "s_k * 1{y not in top_k}, s_k = temperature-scaled mass of top_k",
                      "value": "1{y in top_k} / k", "g_bins": BINS.tolist()},
           "design_split_loss_shape": design_shape, "summary": summary, "per_split": per_split}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "imagenet_nonmonotone.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
