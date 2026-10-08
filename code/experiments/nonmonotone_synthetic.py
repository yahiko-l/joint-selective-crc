"""Synthetic non-monotone losses: the certifier and three threshold rules under nested, mixed and non-nested couplings.

Acceptance score g ~ U(0, 1), acceptance A = 1{g > tau} on a fixed tau grid. At grid
value lambda_j an input's loss is Bernoulli(r_j h(g)) with h(g) = 4 g (1 - g), so the
risk contribution peaks at intermediate confidence and the selected risk
R_sel(lambda_j, tau) = r_j H(tau), H(tau) = E[h(g) | g > tau] = (2/3)(1 - tau)(1 + 2 tau),
first rises and then falls in tau. Three couplings share every per-candidate
conditional marginal, selected risk, acceptance probability and utility (only the
dependence across candidates differs):
  nested      L_i(lambda_j) = 1{U_i < r_j h(g_i)} with r_j non-increasing in j, so each
              input's loss is non-increasing in lambda (the monotone case);
  non-nested  L_i(lambda_j) = 1{U_ij < r_j h(g_i)} with U_ij independent across j given g,
              adapted from the construction CRC uses to show that its rule needs
              monotone losses;
  mixed       per input, the nested draw with probability 1/2 and the non-nested one
              otherwise.
At the reference threshold tau0 the aggressive half of the lambda grid has selected
risk alpha + eps, the conservative half RISK_GOOD, and the most conservative value
has zero loss. Population risk is therefore non-increasing in lambda; only the
samplewise nesting differs between couplings.

Per replication and cell:
  * the certifier on the full (lambda, tau) grid; its returned pair is scored on the
    exact population values (risk, acceptance, utility against U_LCB);
  * at tau0, on the accepted calibration points, three threshold rules:
      CRC            lambda_hat = inf{lambda : S(lambda) + 1 <= (s + 1) alpha}, lambda_max if
                     empty (Angelopoulos et al., eq. (4)); guarantee on the expectation;
      first crossing the first lambda whose pointwise Clopper-Pearson upper bound at
                     level delta is below alpha (the simplification of the RCPS rule that
                     holds for nested losses);
      RCPS scan      the smallest lambda such that every lambda' >= lambda has its upper
                     bound below alpha (Bates et al., eq. (4), binary-loss bound).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selective_crc import certify_grid  # noqa: E402
from selective_crc.bounds import sample_size_condition  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nonmonotone"

T = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
P_ACC = np.array([0.9, 0.7, 0.5, 0.3, 0.1])          # exact 1 - T
TAU0_IDX = 2                                          # tau0 = 0.5
ALPHA, DELTA, PI_MIN, C = 0.10, 0.10, 0.10, 0.05
RISK_GOOD = 0.03
MAIN_EPS = 0.005
N_CERTS = (2000, 5000, 20000)
M_LAMBDAS = (10, 20, 100)
COUPLINGS = ("nested", "mixed", "non-nested")
SENS_EPS = (0.0, 0.01, 0.02)                          # plus MAIN_EPS, at the sensitivity cell
SENS_CELL = (2000, 100)


def H(tau):
    """E[4 g (1 - g) | g > tau] for g ~ U(0, 1)."""
    tau = np.asarray(tau, dtype=float)
    return (2.0 / 3.0) * (1.0 - tau) * (1.0 + 2.0 * tau)


def profile(m_lambda, eps):
    half = m_lambda // 2
    r = np.where(np.arange(m_lambda) < half, ALPHA + eps, RISK_GOOD) / H(T[TAU0_IDX])
    r[-1] = 0.0
    return r


def weights(m_lambda):
    return 1.0 - 0.5 * np.arange(m_lambda) / max(1, m_lambda - 1)


def population(r):
    """Exact R_sel, p_acc and deployment utility per pair, lambda-major."""
    w = weights(len(r))
    R = np.outer(r, H(T))                                            # (m_lambda, m_tau)
    U = w[:, None] * P_ACC[None, :] * (1.0 - R) - C * (1.0 - P_ACC)[None, :]
    return R.reshape(-1), np.tile(P_ACC, len(r)), U.reshape(-1)


def oracle(R, p, U):
    ok = (R <= ALPHA) & (p >= PI_MIN)
    return float(U[ok].max()) if ok.any() else None


def draw_losses(n, r, coupling, g, rng):
    q = np.outer(4.0 * g * (1.0 - g), r)
    common = rng.uniform(size=(n, 1))
    indep = rng.uniform(size=(n, len(r)))
    if coupling == "nested":
        u = np.broadcast_to(common, q.shape)
    elif coupling == "non-nested":
        u = indep
    else:
        u = np.where(rng.uniform(size=(n, 1)) < 0.5, common, indep)
    return (u < q).astype(np.float64)


def cp_upper(k, n, level):
    return np.where(k >= n, 1.0, beta_dist.ppf(level, k + 1, np.maximum(n - k, 1)))


def one_replication(n, r, coupling, rng, pop):
    R_pop, p_pop, U_pop = pop
    m_lambda, m_tau = len(r), len(T)
    g = rng.uniform(size=n)
    L_lam = draw_losses(n, r, coupling, g, rng)
    A_tau = (g[:, None] > T[None, :]).astype(np.float64)

    L = np.repeat(L_lam, m_tau, axis=1)
    A = np.tile(A_tau, (1, m_lambda))
    v = (1.0 - L) * np.repeat(weights(m_lambda), m_tau)[None, :]
    res = certify_grid(L, A, v, alpha=ALPHA, pi_min=PI_MIN, delta=DELTA, c=C,
                       V=1.0, B=1.0, check_sample_size=False)
    if res.is_infeasible:
        cert = {"feasible": False}
    else:
        k = int(res.selected)
        j = k // m_tau
        risk_fail = bool(R_pop[k] > ALPHA)
        acc_fail = bool(p_pop[k] < PI_MIN)
        util_fail = bool(res.u_lcb_per_pair[k] > U_pop[k])
        cert = {"feasible": True, "risk_fail": risk_fail, "acc_fail": acc_fail,
                "util_fail": util_fail, "R_sel": float(R_pop[k]), "p_acc": float(p_pop[k]),
                "U_dep": float(U_pop[k]), "U_lcb": float(res.u_lcb_per_pair[k]),
                "kind": "terminal" if j == m_lambda - 1 else ("aggressive" if j < m_lambda // 2 else "conservative"),
                "tau": float(T[k % m_tau])}

    acc = A_tau[:, TAU0_IDX] > 0
    s = int(acc.sum())
    S = L_lam[acc].sum(axis=0)
    risk0 = r * H(T[TAU0_IDX])
    ok = S + 1.0 <= (s + 1.0) * ALPHA
    j_crc = int(np.argmax(ok)) if ok.any() else m_lambda - 1
    ucb = cp_upper(S, s, 1.0 - DELTA) if s > 0 else np.ones(m_lambda)
    below = ucb < ALPHA
    j_fc = int(np.argmax(below)) if below.any() else m_lambda - 1
    # suffix scan: smallest j with below[j:] all True (lambda_max if below[-1] is False)
    j_scan = m_lambda - 1
    while j_scan > 0 and below[j_scan - 1] and below[j_scan:].all():
        j_scan -= 1
    return cert, float(risk0[j_crc]), float(risk0[j_fc]), float(risk0[j_scan])


def cp_bounds(k, n):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.05, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.95, k + 1, n - k))
    return lo, hi


def run_cell(args):
    n, m_lambda, coupling, eps, reps, seed = args
    rng = np.random.default_rng(seed)
    r = profile(m_lambda, eps)
    pop = population(r)
    rows = [one_replication(n, r, coupling, rng, pop) for _ in range(reps)]
    cert = [x[0] for x in rows]
    feas = [c for c in cert if c["feasible"]]
    crc, fc, scan = (np.array([x[i] for x in rows]) for i in (1, 2, 3))

    def count(key):
        return sum(c[key] for c in feas)

    risk_or_acc = sum(c["risk_fail"] or c["acc_fail"] for c in feas)
    any_fail = sum(c["risk_fail"] or c["acc_fail"] or c["util_fail"] for c in feas)
    agg = ALPHA + eps
    n_agg = int(np.isclose(crc, agg).sum())
    crc_lo_bound = agg * cp_bounds(n_agg, reps)[0] if agg > ALPHA else None
    n_fc, n_scan = int((fc > ALPHA).sum()), int((scan > ALPHA).sum())
    orc = oracle(*pop)
    med = lambda key: float(np.median([c[key] for c in feas])) if feas else None  # noqa: E731
    return {
        "n_cert": n, "m_lambda": m_lambda, "m": m_lambda * len(T), "coupling": coupling, "eps": eps,
        "reps": reps, "seed": seed,
        "n_required_star": sample_size_condition(m_lambda * len(T), DELTA, PI_MIN),
        "star_satisfied": bool(n >= sample_size_condition(m_lambda * len(T), DELTA, PI_MIN)),
        "certifier": {
            "feasible": len(feas),
            "risk_or_acceptance_failures": risk_or_acc,
            "risk_or_acceptance_cp90": cp_bounds(risk_or_acc, reps),
            "utility_failures": count("util_fail"), "any_failures": any_fail,
            "any_cp90": cp_bounds(any_fail, reps),
            "median_R_sel": med("R_sel"), "median_p_acc": med("p_acc"),
            "median_U_dep": med("U_dep"), "median_U_lcb": med("U_lcb"),
            "oracle_U_dep": orc,
            "median_oracle_gap": (orc - med("U_dep")) if feas and orc is not None else None,
            "kind_counts": {k: sum(c["kind"] == k for c in feas) for k in ("aggressive", "conservative", "terminal")},
            "tau_counts": {str(t): sum(c["tau"] == t for c in feas) for t in T.tolist()},
        },
        "crc": {"mean_R_sel": float(crc.mean()), "mc_se": float(crc.std(ddof=1) / np.sqrt(reps)),
                "aggressive_selections": n_agg,
                "mean_R_sel_lower_95": crc_lo_bound},
        "first_crossing": {"failures": n_fc, "rate": n_fc / reps, "cp90": cp_bounds(n_fc, reps)},
        "rcps_scan": {"failures": n_scan, "rate": n_scan / reps, "cp90": cp_bounds(n_scan, reps)},
    }


def main(reps, workers):
    jobs, seq = [], np.random.SeedSequence(20260928)
    cells = [(n, m, cpl, MAIN_EPS) for n, m, cpl in product(N_CERTS, M_LAMBDAS, COUPLINGS)]
    cells += [(SENS_CELL[0], SENS_CELL[1], cpl, e) for e in SENS_EPS for cpl in COUPLINGS]
    for cell, ss in zip(cells, seq.spawn(len(cells))):
        jobs.append((*cell, reps, int(ss.generate_state(1)[0])))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_cell, jobs))
    for res in results:
        c = res["certifier"]
        print(f"n={res['n_cert']:6d} m_lam={res['m_lambda']:4d} eps={res['eps']:.3f} {res['coupling']:10s} "
              f"star={res['star_satisfied']!s:5} | cert feas {c['feasible']} R/A-fail {c['risk_or_acceptance_failures']} "
              f"any-fail {c['any_failures']} p_acc {c['median_p_acc']} U {c['median_U_dep']} gap {c['median_oracle_gap']} "
              f"| CRC {res['crc']['mean_R_sel']:.4f}+-{res['crc']['mc_se']:.4f} "
              f"| 1st-cross {res['first_crossing']['failures']} | scan {res['rcps_scan']['failures']}", flush=True)
    out = {"design": {"tau": T.tolist(), "p_acc": P_ACC.tolist(), "tau0": float(T[TAU0_IDX]),
                      "alpha": ALPHA, "delta": DELTA, "pi_min": PI_MIN, "c": C,
                      "risk_good": RISK_GOOD, "main_eps": MAIN_EPS, "sens_eps": SENS_EPS,
                      "sens_cell": SENS_CELL, "h": "4 g (1 - g)", "H_tau": H(T).tolist(),
                      "value": "(1 - L) w_j, w_j linear from 1 (most aggressive) to 0.5",
                      "reps": reps, "interval_level": "one-sided 95% (two-sided 90% CP)"},
           "cells": results}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "synthetic_nonmonotone.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=24)
    a = parser.parse_args()
    main(a.reps, a.workers)
