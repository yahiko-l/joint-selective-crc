"""Sign-aware valid-vs-valid per-pair risk-UCB comparison (revision).

Reformulates the matched per-pair tightness comparison so that BOTH methods are
standalone (1-delta) grid-valid upper bounds on R_sel, with sign-aware
denominator handling. For each grid pair, let U = z_bar + r be the method's
upper bound on E[Z] = p_acc * (R_sel - alpha), and [p-, p+] a TWO-SIDED
Clopper--Pearson confidence interval for p_acc. The standalone risk UCB is

    UCB_sel = alpha + U / p-   if U >= 0   (lower endpoint is conservative)
              alpha + U / p+   if U <  0   (upper endpoint is conservative)

with the loss-range bound UCB_sel = B when U >= 0 and p- = 0. Because the
branch U >= 0 vs U < 0 is data-dependent, the CP interval must cover BOTH
endpoints simultaneously; a union bound over the two tails suffices
(independence between numerator and denominator is not required).

The compared quantity is the UPPER-UCB EXCESS over the shared empirical anchor,
UCB_sel - Rhat_sel, a descriptive derived statistic; validity resides in
UCB_sel itself. PRIMARY convention: both methods' UCBs are intersected with the
deterministic loss-range bound R_sel <= B before the excess is formed (a UCB
above the loss range carries no information beyond B, and crediting either
method for the other's beyond-range values would distort the ratio); the
unclipped medians are reported as *_unclipped sensitivity fields.

Matched error allocation (per pair, both methods): the numerator-UCB event
receives delta/(2m); the two-sided CP interval receives total noncoverage
delta/(2m), split as delta/(4m) per tail. The two components cost at most
delta/m per pair and delta over the m-pair grid. Numerator radii keep each
method's published convention: Ours uses the Maurer--Pontil empirical-variance
radius with log(4/delta_event) (the paper-wide two-tail constant), the
comparator uses the range Hoeffding radius with log(2/delta_event) exactly as
`baseline_a_range_hoeffding_plcb`. A certificate-allocation sensitivity
(numerator at the certificate's own delta/(16m) ledger level, CP tails at
delta/(16m); total 3*delta/16 over the grid, still grid-valid) is reported in
the JSON only.

Three blocks, one run:
  1. HEADLINE  -- D_5baseline protocol (ImageNet RN50/101/152 V2, n_cert=33000,
     20 seeds 42..61, m=35): pooled per-(seed,pair) median excess ratios
     Ours / sign-aware Hoeffding--CP, negative-numerator fractions, edge-case
     ledger (min s, s=0/s=1 cells, p-=0 cells, clipping check), per-seed data.
  2. FIG 7     -- same runs, per-pair frontier dump in the exact schema of
     `analysis_cert_frontier.py`, with ucb_rsel replaced by the sign-aware
     valid UCBs (matched allocation) for Ours and the comparator. The textbook
     A(pi_min) expression and ALL certification flags are byte-identical to
     the original script (certification is decided by EB <= 0 and
     p_LCB >= pi_min, unchanged).
  3. FIG 8(b)  -- A1 sweep protocol (RN50, n_cert=25000, 10 seeds 42..51):
     per-pi_min low-acceptance-subset (p_hat <= 2*pi_min) medians of the
     sign-aware matched excess ratio.

Outputs (overwrite):
  results/ablation_supplement/E_signaware_valid_ratio.json          (1 + 3)
  results/analysis/certified_decision_frontier_signaware.json       (2)

Run:
    python -m experiments.analysis_signaware_valid_ratio
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

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

OUT_SUPP = ROOT / "results" / "ablation_supplement"
OUT_ANALYSIS = ROOT / "results" / "analysis"

ALPHA, PI_MIN, DELTA, B = 0.05, 0.01, 0.05, 1.0
C_COST, V_VAL = 0.1, 1.0
N_CLASSES = 1000
LAMBDA = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], dtype=np.float64)
T = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)

# Block 1 + 2: D_5baseline / frontier protocol.
D_NCERT, D_NTUNE, D_NTEST, D_SEEDS = 33000, 8500, 8500, [42 + i for i in range(20)]
# Block 3: A1 pi_min sweep protocol.
A1_NCERT, A1_NTUNE, A1_NTEST, A1_SEEDS = 25000, 12500, 12500, [42 + i for i in range(10)]
A1_PI_MIN_VALUES = [0.005, 0.01, 0.02, 0.05, 0.10]

MODELS = {
    "ResNet-50 V2": f"{DATA_ROOT}/imagenet_data/val_logits.npy",
    "ResNet-101 V2": f"{DATA_ROOT}/imagenet_data/val_logits_resnet101.npy",
    "ResNet-152 V2": f"{DATA_ROOT}/imagenet_data/val_logits_resnet152.npy",
}
LABELS = f"{DATA_ROOT}/imagenet_data/val_labels.npy"


def cp_lower(s: np.ndarray, n: int, tail: float) -> np.ndarray:
    """Vectorised Clopper--Pearson lower endpoint at one-tail level `tail`."""
    out = np.zeros(len(s), dtype=np.float64)
    pos = s > 0
    out[pos] = beta_dist.ppf(tail, s[pos], n - s[pos] + 1)
    return out


def cp_upper(s: np.ndarray, n: int, tail: float) -> np.ndarray:
    """Vectorised Clopper--Pearson upper endpoint at one-tail level `tail`."""
    out = np.ones(len(s), dtype=np.float64)
    lt = s < n
    out[lt] = beta_dist.ppf(1.0 - tail, s[lt] + 1, n - s[lt])
    return out


def build_grid(logits_full, labels_full, split_seed, n_cert, n_tune, n_test):
    """Byte-for-byte the D_5baseline / frontier grid construction."""
    rng = np.random.default_rng(split_seed)
    subset = rng.choice(len(labels_full), size=n_cert + n_tune + n_test, replace=False)
    Y_all = labels_full[subset]
    logits_used = logits_full[subset]
    _, cert_idx, _ = three_split_indices(
        len(Y_all), n_tune=n_tune, n_cert=n_cert, seed=split_seed)
    Y_cert = Y_all[cert_idx]
    logits_cert = logits_used[cert_idx]

    m_tau = len(T)
    contains_Y, set_size = cifar100._compute_contains_and_size(
        logits_cert, Y_cert, LAMBDA, N_CLASSES)
    L_lambda = (~contains_Y).astype(np.float64)
    L_grid = np.repeat(L_lambda, m_tau, axis=1)
    A_lt = cifar100.construct_acceptance(logits_cert, T)
    A_grid = np.tile(A_lt, (1, len(LAMBDA)))
    set_size_safe = np.maximum(set_size, 1)
    v_lambda = contains_Y.astype(np.float64) / set_size_safe.astype(np.float64)
    v_grid = np.repeat(v_lambda, m_tau, axis=1)
    return L_grid, A_grid, v_grid


def seed_quantities(L_grid, A_grid, v_grid, *, pi_min):
    """certify_grid + both budgets' sign-aware margins for one seed split."""
    n, m = L_grid.shape
    res = certify_grid(L_grid, A_grid, v_grid,
                       alpha=ALPHA, pi_min=pi_min, delta=DELTA,
                       c=C_COST, V=V_VAL, B=B, check_sample_size=False)
    z_bar = res.z_bar
    eb = res.eb_per_pair
    s = res.s_count.astype(int)
    p_hat = s / n
    p_hat_safe = np.maximum(p_hat, 1e-9)
    eta_native = eb - z_bar

    Z = A_grid * (L_grid - ALPHA)
    sigma2 = Z.var(axis=0, ddof=1)
    log_matched = np.log(4.0 / (DELTA / (2 * m)))          # log(8m/delta)
    eta_matched = (np.sqrt(2.0 * sigma2 * log_matched / n)
                   + 7.0 * B * log_matched / (3.0 * (n - 1)))
    r_hoeffding = B * np.sqrt(np.log(2.0 / (DELTA / (2 * m))) / (2.0 * n))

    # Two-sided CP intervals: matched delta/(4m) per tail (total delta/(2m));
    # native sensitivity delta/(16m) per tail.
    pl_m, pu_m = cp_lower(s, n, DELTA / (4 * m)), cp_upper(s, n, DELTA / (4 * m))
    pl_n, pu_n = cp_lower(s, n, DELTA / (16 * m)), cp_upper(s, n, DELTA / (16 * m))
    assert np.all(pl_m <= p_hat + 1e-12) and np.all(p_hat <= pu_m + 1e-12)
    assert np.all(pl_n <= p_hat + 1e-12) and np.all(p_hat <= pu_n + 1e-12)

    rhat_minus_alpha = np.where(s > 0, z_bar / p_hat_safe, np.nan)

    def sa_excess(U, pl, pu):
        """Sign-aware UCB excess over alpha: UCB_sel = ALPHA + excess."""
        return np.where(U >= 0,
                        np.where(pl > 0, U / pl, B - ALPHA),   # loss-range fallback
                        U / pu)

    def sa_margin(excess, *, clip):
        """Upper-UCB excess over Rhat_sel; NaN where s <= 1 (kept to the
        same gate as the released headline computation). With clip=True
        (the primary convention) the UCB is first intersected with the
        deterministic loss-range bound R_sel <= B."""
        if clip:
            excess = np.minimum(excess, B - ALPHA)      # UCB = ALPHA + excess <= B
        marg = np.where(s > 1, excess - rhat_minus_alpha, np.nan)
        ok = ~np.isnan(marg)
        assert np.all(marg[ok] >= -1e-9), float(np.nanmin(marg))
        return marg

    exc_ours_m = sa_excess(z_bar + eta_matched, pl_m, pu_m)
    exc_comp_m = sa_excess(z_bar + r_hoeffding, pl_m, pu_m)
    exc_ours_n = sa_excess(z_bar + eta_native, pl_n, pu_n)

    # Display-normalised reproductions, byte-identical to the published pipeline.
    ours_display = eta_native / p_hat_safe                  # ours_r_margin
    comp_display = baseline_a_range_hoeffding_plcb(
        L_grid, A_grid, alpha=ALPHA, delta=DELTA, B=B)

    return {
        "res": res,
        "n": n, "m": m,
        "z_bar": z_bar, "eb": eb, "s": s, "p_hat": p_hat,
        "p_lcb_native": res.p_lcb_per_pair,
        "eta_matched": eta_matched, "eta_native": eta_native,
        "r_hoeffding": r_hoeffding,
        "pl_m": pl_m, "pu_m": pu_m,
        "margin_ours_matched": sa_margin(exc_ours_m, clip=True),
        "margin_comp_matched": sa_margin(exc_comp_m, clip=True),
        "margin_ours_native": sa_margin(exc_ours_n, clip=True),
        "margin_ours_matched_unclipped": sa_margin(exc_ours_m, clip=False),
        "margin_comp_matched_unclipped": sa_margin(exc_comp_m, clip=False),
        "margin_ours_native_unclipped": sa_margin(exc_ours_n, clip=False),
        "ucb_ours_matched": np.minimum(ALPHA + exc_ours_m, B),
        "ucb_comp_matched": np.minimum(ALPHA + exc_comp_m, B),
        "rhat": ALPHA + rhat_minus_alpha,
        "U_matched_ours": z_bar + eta_matched,
        "U_native_ours": z_bar + eta_native,
        "ours_display": ours_display,
        "comp_display": comp_display,
    }


def pooled_median_ratio(num_rows, den_rows):
    nums, dens = np.concatenate(num_rows), np.concatenate(den_rows)
    ok = np.isfinite(nums) & np.isfinite(dens) & (dens > 0)
    return float(np.median(nums[ok] / dens[ok])), int(ok.sum())


def headline_block(per_seed):
    """Block 1 summary for one backbone from the per-seed quantity dicts."""
    r_matched, k = pooled_median_ratio(
        [q["margin_ours_matched"] for q in per_seed],
        [q["margin_comp_matched"] for q in per_seed])
    r_native, _ = pooled_median_ratio(
        [q["margin_ours_native"] for q in per_seed],
        [q["margin_comp_matched"] for q in per_seed])
    # Display reproduction under the D-script inclusion rule (ours > 0, comp > 0).
    disp_pairs = []
    for q in per_seed:
        o, c = q["ours_display"], q["comp_display"]
        ok = np.isfinite(o) & (o > 0) & np.isfinite(c) & (c > 0)
        disp_pairs.append((o[ok], c[ok]))
    r_display = float(np.median(np.concatenate([o / c for o, c in disp_pairs])))

    # Clipping sensitivity: the unclipped (as-constructed) matched median.
    r_matched_unclipped, _ = pooled_median_ratio(
        [q["margin_ours_matched_unclipped"] for q in per_seed],
        [q["margin_comp_matched_unclipped"] for q in per_seed])

    # Low-acceptance subset (p_hat <= 2*pi_min) at the headline operating point.
    low_nums, low_dens, n_low = [], [], 0
    for q in per_seed:
        low = q["p_hat"] <= 2.0 * PI_MIN
        n_low += int(low.sum())
        low_nums.append(np.where(low, q["margin_ours_matched"], np.nan))
        low_dens.append(np.where(low, q["margin_comp_matched"], np.nan))
    r_low, k_low = (pooled_median_ratio(low_nums, low_dens)
                    if n_low > 0 else (None, 0))

    s_all = np.concatenate([q["s"] for q in per_seed])
    return {
        "median_excess_ratio_matched": round(r_matched, 4),
        "median_excess_ratio_matched_LOW": (
            None if r_low is None else round(r_low, 4)),
        "n_low_cells": n_low,
        "median_excess_ratio_native_sensitivity": round(r_native, 4),
        "median_ratio_display_form_reproduction": round(r_display, 4),
        "frac_cells_negative_numerator_matched": round(
            float(np.mean(np.concatenate(
                [q["U_matched_ours"] < 0 for q in per_seed]))), 4),
        "frac_cells_negative_numerator_native": round(
            float(np.mean(np.concatenate(
                [q["U_native_ours"] < 0 for q in per_seed]))), 4),
        "n_ratio_samples": k,
        "min_s": int(s_all.min()),
        "n_cells_s0": int((s_all == 0).sum()),
        "n_cells_s1": int((s_all == 1).sum()),
        "n_cells_plcb_zero_matched": int(
            (np.concatenate([q["pl_m"] for q in per_seed]) == 0).sum()),
        "median_excess_ratio_matched_unclipped": round(r_matched_unclipped, 4),
        "clipping_changes_matched_median": bool(
            abs(r_matched_unclipped - r_matched) > 5e-5),
        "per_seed": [{
            "seed": int(seed),
            "s": q["s"].tolist(),
            "p_hat": [round(float(x), 6) for x in q["p_hat"]],
            "margin_ours_matched": [round(float(x), 8) for x in q["margin_ours_matched"]],
            "margin_comp_matched": [round(float(x), 8) for x in q["margin_comp_matched"]],
            "margin_ours_native": [round(float(x), 8) for x in q["margin_ours_native"]],
        } for seed, q in zip(D_SEEDS, per_seed)],
    }


def frontier_pair_records(per_seed, pi_min):
    """Block 2: per-pair records in the analysis_cert_frontier.py schema,
    with sign-aware matched valid UCBs for ours / a_plcb and byte-identical
    certification flags."""
    m_tau = len(T)
    seeds_out = []
    for q in per_seed:
        n, m = q["n"], q["m"]
        z_bar, p_hat, s = q["z_bar"], q["p_hat"], q["s"]
        p_hat_safe = np.maximum(p_hat, 1e-12)
        acc_ok = q["p_lcb_native"] >= pi_min

        ours_cert = (q["eb"] <= 0.0) & acc_ok

        a_widths = baseline_a_range_hoeffding(
            (q["_L"]), (q["_A"]), alpha=ALPHA, pi_min=pi_min, delta=DELTA, B=B)
        a_ucb_z = z_bar + p_hat * a_widths
        a_cert = (a_ucb_z <= 0.0) & acc_ok
        ucb_rsel_a = ALPHA + a_ucb_z / p_hat_safe            # unchanged textbook display

        aplcb_widths = q["comp_display"]
        with np.errstate(invalid="ignore"):
            aplcb_ucb_z = z_bar + p_hat * aplcb_widths
            aplcb_cert = np.where(np.isnan(aplcb_widths), False,
                                  (aplcb_ucb_z <= 0.0) & acc_ok)

        def nan_to_none(arr):
            return [None if not np.isfinite(x) else float(x) for x in arr]

        seeds_out.append({
            "p_hat": [float(x) for x in p_hat],
            "p_lcb": [float(x) for x in q["p_lcb_native"]],
            "ucb_rsel": {
                "ours": nan_to_none(q["ucb_ours_matched"]),
                "a_pi_min": nan_to_none(ucb_rsel_a),
                "a_plcb": nan_to_none(q["ucb_comp_matched"]),
            },
            "cert": {
                "ours": [bool(x) for x in ours_cert],
                "a_pi_min": [bool(x) for x in a_cert],
                "a_plcb": [bool(x) for x in aplcb_cert],
            },
            "n_certified": {
                "ours": int(ours_cert.sum()),
                "a_pi_min": int(a_cert.sum()),
                "a_plcb": int(np.asarray(aplcb_cert).sum()),
            },
        })

    methods = ["ours", "a_pi_min", "a_plcb"]
    p_hat_med = np.median(np.array([t["p_hat"] for t in seeds_out]), axis=0)
    p_lcb_med = np.median(np.array([t["p_lcb"] for t in seeds_out]), axis=0)

    def med_col(vals):
        arr = np.array([[np.nan if v is None else v for v in row] for row in vals])
        out = []
        for j in range(arr.shape[1]):
            col = arr[:, j][np.isfinite(arr[:, j])]
            out.append(None if col.size == 0 else float(np.median(col)))
        return out

    ucb_med = {mth: med_col([t["ucb_rsel"][mth] for t in seeds_out]) for mth in methods}
    cert_frac = {mth: np.mean(np.array([t["cert"][mth] for t in seeds_out]), axis=0)
                 for mth in methods}
    m = len(LAMBDA) * m_tau
    pairs = []
    for k in range(m):
        pairs.append({
            "k": k,
            "lambda_idx": k // m_tau, "tau_idx": k % m_tau,
            "lambda": float(LAMBDA[k // m_tau]), "tau": float(T[k % m_tau]),
            "p_acc": float(p_hat_med[k]),
            "p_lcb": float(p_lcb_med[k]),
            "ucb_rsel": {mth: ucb_med[mth][k] for mth in methods},
            "cert_frac": {mth: float(cert_frac[mth][k]) for mth in methods},
            "cert": {mth: bool(cert_frac[mth][k] >= 0.5) for mth in methods},
        })

    # Caption-claim checks on the SIGN-AWARE values, every seed.
    apim_zero = all(t["n_certified"]["a_pi_min"] == 0 for t in seeds_out)
    ours_ge = all(t["n_certified"]["ours"] >= t["n_certified"]["a_plcb"]
                  for t in seeds_out)
    strict = True
    for t in seeds_out:
        uo, ua = t["ucb_rsel"]["ours"], t["ucb_rsel"]["a_plcb"]
        for k in range(len(uo)):
            if (t["cert"]["ours"][k] and t["cert"]["a_plcb"][k]
                    and uo[k] is not None and ua[k] is not None
                    and uo[k] > ua[k] + 1e-12):
                strict = False
    checks = {
        "n_seeds": len(seeds_out),
        "all_seeds_a_pi_min_certifies_zero": bool(apim_zero),
        "all_seeds_ours_count_ge_a_plcb": bool(ours_ge),
        "all_seeds_signaware_ucb_dominance_on_shared_pairs": bool(strict),
    }

    # Tier-level prose numbers: top-acceptance tier ratio (level ratio), as the
    # figure annotation computes it.
    by_tau = {}
    for p in pairs:
        by_tau.setdefault(p["tau_idx"], []).append(p)
    tiers = []
    for ti, pts in by_tau.items():
        rec = {"p_acc": float(np.median([p["p_acc"] for p in pts]))}
        for mth in ("ours", "a_plcb"):
            u = [p["ucb_rsel"][mth] for p in pts if p["ucb_rsel"][mth] is not None]
            rec[mth] = {"best": (min(u) if u else None),
                        "cert": any(p["cert"][mth] for p in pts)}
        tiers.append(rec)
    tiers.sort(key=lambda r: r["p_acc"])
    shared_dom = all(
        t["ours"]["best"] <= t["a_plcb"]["best"] + 1e-12
        for t in tiers if t["ours"]["cert"] and t["a_plcb"]["cert"]
        and t["ours"]["best"] is not None and t["a_plcb"]["best"] is not None)
    top = tiers[-1]
    prose = {
        "top_tier_ucb_ours": top["ours"]["best"],
        "top_tier_ucb_a_plcb": top["a_plcb"]["best"],
        "top_tier_level_ratio": (
            None if not (top["ours"]["best"] and top["a_plcb"]["best"])
            else round(top["a_plcb"]["best"] / top["ours"]["best"], 3)),
        "ours_ucb_le_a_plcb_on_all_shared_certified_tiers": bool(shared_dom),
    }
    return pairs, checks, prose, seeds_out


def main():
    t_start = time.time()
    labels_full = np.load(LABELS).astype(np.int64)

    e_out = {
        "experiment": "analysis_signaware_valid_ratio",
        "purpose": ("Sign-aware valid-vs-valid per-pair risk-UCB excess "
                    "comparison introduced in the revision; replaces the "
                    "display-normalised matched-valid width ratio."),
        "definition": {
            "object": "UCB_sel - Rhat_sel (upper-UCB excess), both methods",
            "sign_rule": "U>=0 -> divide by CP lower endpoint; U<0 -> divide by "
                         "CP upper endpoint; loss-range bound B if U>=0 and p-=0",
            "matched_allocation": "numerator-UCB event delta/(2m); two-sided CP "
                                  "interval total delta/(2m), delta/(4m) per tail; "
                                  "cost <= delta/m per pair, <= delta over the grid",
            "native_sensitivity": "numerator at the certificate ledger level "
                                  "delta/(16m); CP tails delta/(16m) each; "
                                  "total 3*delta/16 over the grid",
            "numerator_radii": "Ours: Maurer-Pontil log(4/delta_event); "
                               "comparator: range Hoeffding log(2/delta_event) "
                               "(baseline_a_range_hoeffding_plcb convention)",
        },
        "config": {
            "alpha": ALPHA, "pi_min": PI_MIN, "delta": DELTA, "B": B,
            "c": C_COST, "V": V_VAL, "n_classes": N_CLASSES,
            "Lambda": LAMBDA.tolist(), "T": T.tolist(),
            "headline": {"n_cert": D_NCERT, "n_tune": D_NTUNE, "n_test": D_NTEST,
                         "seeds": D_SEEDS},
            "fig8b_sweep": {"n_cert": A1_NCERT, "n_tune": A1_NTUNE,
                            "n_test": A1_NTEST, "seeds": A1_SEEDS,
                            "pi_min_values": A1_PI_MIN_VALUES},
        },
        "models": {},
    }

    frontier_models = {}
    for name, path in MODELS.items():
        logits_full = np.load(path).astype(np.float64)
        top1 = float((logits_full.argmax(axis=1) == labels_full).mean())
        print(f"[signaware] {name}: top-1 {top1:.4f}", flush=True)
        per_seed = []
        for seed in D_SEEDS:
            t0 = time.time()
            L_grid, A_grid, v_grid = build_grid(
                logits_full, labels_full, seed, D_NCERT, D_NTUNE, D_NTEST)
            q = seed_quantities(L_grid, A_grid, v_grid, pi_min=PI_MIN)
            q["_L"], q["_A"] = L_grid, A_grid
            per_seed.append(q)
            print(f"  seed {seed} in {time.time()-t0:.1f}s", flush=True)

        e_out["models"][name] = headline_block(per_seed)
        pairs, checks, prose, _ = frontier_pair_records(per_seed, PI_MIN)
        frontier_models[name] = {
            "top1": top1, "pairs": pairs,
            "per_seed_checks": checks, "prose_numbers": prose,
        }
        h = e_out["models"][name]
        print(f"  matched={h['median_excess_ratio_matched']} "
              f"native={h['median_excess_ratio_native_sensitivity']} "
              f"display={h['median_ratio_display_form_reproduction']} "
              f"negfrac={h['frac_cells_negative_numerator_matched']} "
              f"top-tier ratio={prose['top_tier_level_ratio']}", flush=True)
        for q in per_seed:
            del q["_L"], q["_A"], q["res"]
        del logits_full

    # ---- Block 3: Fig 8(b) sweep (A1 protocol, RN50 only) ----
    print("[signaware] Fig 8(b) sweep (A1 protocol)", flush=True)
    logits_full = np.load(MODELS["ResNet-50 V2"]).astype(np.float64)
    sweep_seed_q = []
    for seed in A1_SEEDS:
        t0 = time.time()
        L_grid, A_grid, v_grid = build_grid(
            logits_full, labels_full, seed, A1_NCERT, A1_NTUNE, A1_NTEST)
        # pi_min only gates feasibility, not any width; margins are pi_min-free.
        q = seed_quantities(L_grid, A_grid, v_grid, pi_min=A1_PI_MIN_VALUES[0])
        sweep_seed_q.append(q)
        print(f"  seed {seed} in {time.time()-t0:.1f}s", flush=True)
    del logits_full

    rows = []
    for pi in A1_PI_MIN_VALUES:
        nums, dens = [], []
        n_low = 0
        for q in sweep_seed_q:
            low = q["p_hat"] <= 2.0 * pi
            n_low += int(low.sum())
            nums.append(np.where(low, q["margin_ours_matched"], np.nan))
            dens.append(np.where(low, q["margin_comp_matched"], np.nan))
        med, k = pooled_median_ratio(nums, dens)
        rows.append({"pi_min": pi,
                     "median_signaware_excess_ratio_LOW": round(med, 4),
                     "n_low_cells": n_low, "n_ratio_samples": k})
        print(f"  pi_min={pi}: median={med:.4f} over {k} low cells", flush=True)
    e_out["fig8b_sweep_signaware"] = {"rows": rows}

    OUT_SUPP.mkdir(parents=True, exist_ok=True)
    OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    e_path = OUT_SUPP / "E_signaware_valid_ratio.json"
    e_path.write_text(json.dumps(e_out, indent=2))
    print(f"[signaware] wrote {e_path}")

    frontier_out = {
        "experiment": "analysis_signaware_valid_ratio (frontier block)",
        "purpose": ("Per-grid-pair SIGN-AWARE VALID risk UCBs behind the "
                    "frontier figure; certification flags byte-identical to "
                    "analysis_cert_frontier.py, y-values replaced by the "
                    "matched-allocation standalone valid UCBs."),
        "config": {
            "n_seeds": len(D_SEEDS), "n_cert": D_NCERT, "n_tune": D_NTUNE,
            "n_test": D_NTEST, "alpha": ALPHA, "pi_min": PI_MIN, "delta": DELTA,
            "c": C_COST, "V": V_VAL, "B": B, "n_classes": N_CLASSES,
            "Lambda": LAMBDA.tolist(), "T": T.tolist(),
            "ucb_definition": "sign-aware matched allocation; see "
                              "E_signaware_valid_ratio.json definition block",
        },
        "models": frontier_models,
    }
    f_path = OUT_ANALYSIS / "certified_decision_frontier_signaware.json"
    f_path.write_text(json.dumps(frontier_out, indent=2))
    print(f"[signaware] wrote {f_path}")
    print(f"[signaware] total {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
