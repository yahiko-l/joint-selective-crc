"""Re-split effect sizes and distribution-free CIs for the headline comparisons (revision).

Replaces the resplit Wilcoxon significance table (former Table 8 content, B14): the N
random calibration/test re-splits re-partition a single fixed evaluation pool, so they
are randomization replicates conditional on that pool, not independent samples from the
data distribution, and with saturated sign patterns an exact signed-rank p-value is a
function of the re-split count alone. This analysis therefore reports, per comparison,

  * the seed-level median (within-re-split median over grid pairs, then the median
    across re-splits),
  * the Hodges--Lehmann (HL) pseudo-median: the median of Walsh averages
    (x_i + x_j)/2, i <= j, of the seed-level statistics,
  * a distribution-free order-statistic confidence interval for the MEDIAN of the
    split-randomization distribution conditional on the pool: [x_(k), x_(n+1-k)] with
    the largest k such that the binomial coverage 1 - 2 BinomCDF(k-1; n, 1/2) is at
    least 0.95 (a finite-sample lower bound on coverage; ties can only increase it),
  * the sign-consistency count,

and attaches NO p-values. All inputs are existing artifacts; no new experiments.

Comparisons:
  1. ImageNet sign-aware valid UCB-excess ratio Ours / A(CP+-), matched allocation
     (sign-aware reformulation), RN50/101/152 V2, 20 re-splits x 35 pairs, from
     E_signaware_valid_ratio.json; all-pairs rows plus the RN50 low-acceptance subset
     (p_hat <= 2 pi_min). Difference scale (comp - ours) reported alongside for the
     Hodges--Lehmann paired-difference request (revision).
  2. COCO certified-acceptance gap over the Hoeffding--CRC selective baseline
     (delta_p_ours_hoeff, percentage points), g = softmax (30 re-splits) and
     g = entropy (20 re-splits), from G_coco_pixacc_*_robust.json. PRIMARY convention
     (all-split, operational): a re-split on which the certifier abstains contributes a
     zero gap; this is an evaluation convention, not a certified value. The
     feasibility-conditional summaries are recorded as secondary fields.
  3. Section 6 comparator disclosure Ours/B (nominal per-pair radii), RN50/101/152 V2,
     20 re-splits, from D_5baseline_multimodel.json; seed-level ratio statistics
     accompanying the published pooled per-pair medians.

Reproduction anchors (asserted): pooled medians 0.1106 / 0.1461 / 0.1446 (all pairs),
0.132 (RN50 low-acceptance), COCO softmax median gap +22.1 pp with 29/30 feasible,
entropy 12/20 feasible, Ours/B pooled 1.61 / 1.54 / 1.51.

Output (overwrite): results/ablation_supplement/H_resplit_effectsize.json

Run:
    python -m experiments.analysis_resplit_effectsize
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "results" / "ablation_supplement"
OUT_JSON = SUPP / "H_resplit_effectsize.json"

PI_MIN_IMAGENET = 0.01
CONF = 0.95


def hodges_lehmann(x: np.ndarray) -> float:
    """One-sample HL pseudo-median: median of Walsh averages (x_i + x_j)/2, i <= j."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    walsh = [(x[i] + x[j]) / 2.0 for i in range(n) for j in range(i, n)]
    return float(np.median(walsh))


def order_stat_ci(x: np.ndarray, conf: float = CONF) -> dict:
    """Distribution-free CI for the median of the split-randomization distribution.

    [x_(k), x_(n+1-k)] with the largest k such that the binomial coverage
    1 - 2 BinomCDF(k-1; n, 1/2) >= conf. The stated level is a finite-sample lower
    bound on coverage (exact for continuous distributions; ties only increase it).
    """
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = xs.size
    k_sel, cov_sel = None, None
    for k in range(1, n // 2 + 1):
        cov = 1.0 - 2.0 * float(binom.cdf(k - 1, n, 0.5))
        if cov >= conf:
            k_sel, cov_sel = k, cov
    assert k_sel is not None, f"no valid order-statistic CI at n={n}, conf={conf}"
    return {
        "lo": float(xs[k_sel - 1]),
        "hi": float(xs[n - k_sel]),
        "k": int(k_sel),
        "coverage_lower_bound": float(cov_sel),
    }


def summarize(x: np.ndarray, sign: str) -> dict:
    """Full row summary for seed-level statistics x; sign in {'below_1','above_1','above_0'}."""
    x = np.asarray(x, dtype=np.float64)
    thr, op = (1.0, np.less) if sign == "below_1" else (1.0, np.greater) if sign == "above_1" else (0.0, np.greater)
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "hodges_lehmann": hodges_lehmann(x),
        "ci95": order_stat_ci(x),
        "min": float(x.min()),
        "max": float(x.max()),
        "sign_consistent": int(op(x, thr).sum()),
        "sign_convention": sign,
        "values": [float(v) for v in x],
    }


def main() -> int:
    out = {
        "experiment": "H_resplit_effectsize",
        "purpose": "Re-split effect sizes + distribution-free CIs replacing resplit Wilcoxon p-values (revision; R3 m5)",
        "date": "2026-08-12",
        "estimand": (
            "median of the split-randomization distribution of the seed-level statistic, "
            "conditional on the fixed evaluation pool; NOT fresh-data inference"
        ),
        "ci_construction": (
            "order-statistic [x_(k), x_(n+1-k)], largest k with 1 - 2 BinomCDF(k-1; n, 1/2) >= 0.95; "
            "stated level is a finite-sample lower bound on coverage"
        ),
        "no_p_values": True,
        "imagenet_signaware": {},
        "coco_gap": {},
        "ours_over_b": {},
    }

    # ------------------------------------------------------------------ ImageNet
    E = json.load((SUPP / "E_signaware_valid_ratio.json").open())
    pooled_anchor = {"ResNet-50 V2": 0.1106, "ResNet-101 V2": 0.1461, "ResNet-152 V2": 0.1441}
    for name, blk in E["models"].items():
        ratio_seed, diff_seed, low_seed = [], [], []
        pooled_cells = []
        for row in blk["per_seed"]:
            mo = np.asarray(row["margin_ours_matched"], dtype=np.float64)
            mc = np.asarray(row["margin_comp_matched"], dtype=np.float64)
            ph = np.asarray(row["p_hat"], dtype=np.float64)
            ratio = mo / mc
            ratio_seed.append(float(np.median(ratio)))
            diff_seed.append(float(np.median(mc - mo)))
            pooled_cells.append(ratio)
            low = ph <= 2.0 * PI_MIN_IMAGENET
            if low.sum() > 0:
                low_seed.append(float(np.median(ratio[low])))
        pooled = float(np.median(np.concatenate(pooled_cells)))
        # anchor: pooled per-(seed,pair)-cell median must reproduce the published value
        assert abs(round(pooled, 4) - blk["median_excess_ratio_matched"]) < 5e-5 or \
            abs(pooled - blk["median_excess_ratio_matched"]) < 5e-4, (name, pooled)
        entry = {
            "ratio_all_pairs": summarize(np.array(ratio_seed), "below_1"),
            "diff_comp_minus_ours": summarize(np.array(diff_seed), "above_0"),
            "pooled_median_reproduction": pooled,
            "published_pooled_median": blk["median_excess_ratio_matched"],
        }
        if low_seed:
            entry["ratio_low_acceptance"] = summarize(np.array(low_seed), "below_1")
            lo_pool = np.concatenate([
                (np.asarray(r["margin_ours_matched"], float) / np.asarray(r["margin_comp_matched"], float))[
                    np.asarray(r["p_hat"], float) <= 2.0 * PI_MIN_IMAGENET]
                for r in blk["per_seed"]])
            entry["pooled_low_acceptance_reproduction"] = float(np.median(lo_pool))
        out["imagenet_signaware"][name] = entry
        print(f"{name}: seed-level median {entry['ratio_all_pairs']['median']:.4f} "
              f"HL {entry['ratio_all_pairs']['hodges_lehmann']:.4f} "
              f"CI [{entry['ratio_all_pairs']['ci95']['lo']:.4f}, {entry['ratio_all_pairs']['ci95']['hi']:.4f}] "
              f"sign {entry['ratio_all_pairs']['sign_consistent']}/{entry['ratio_all_pairs']['n']} "
              f"(pooled {pooled:.4f})")

    rn50 = out["imagenet_signaware"]["ResNet-50 V2"]
    assert "ratio_low_acceptance" in rn50 and round(rn50["pooled_low_acceptance_reproduction"], 3) == 0.132
    assert all(v["ratio_all_pairs"]["sign_consistent"] == v["ratio_all_pairs"]["n"]
               for v in out["imagenet_signaware"].values())

    # ---------------------------------------------------------------------- COCO
    for key, path, n_expect, feas_expect in (
        ("softmax", "G_coco_pixacc_g_softmax_a0.10_pi0.10_robust.json", 30, 29),
        ("entropy", "G_coco_pixacc_g_entropy_a0.10_pi0.10_robust.json", 20, 12),
    ):
        G = json.load((SUPP / path).open())
        gap_pp = np.asarray([s["delta_p_ours_hoeff"] for s in G["per_seed"]], dtype=np.float64) * 100.0
        feas = gap_pp > 0
        assert gap_pp.size == n_expect and int(feas.sum()) == feas_expect, (key, gap_pp.size, int(feas.sum()))
        out["coco_gap"][key] = {
            "unit": "percentage points",
            "primary_all_splits_operational": summarize(gap_pp, "above_0"),
            "primary_convention": "abstaining re-split contributes zero gap (evaluation convention, not a certified value)",
            "secondary_feasible_conditional": summarize(gap_pp[feas], "above_0"),
            "n_feasible": int(feas.sum()),
            "n_splits": int(gap_pp.size),
        }
        p = out["coco_gap"][key]["primary_all_splits_operational"]
        print(f"COCO {key}: median {p['median']:.1f} pp HL {p['hodges_lehmann']:.1f} "
              f"CI [{p['ci95']['lo']:.1f}, {p['ci95']['hi']:.1f}] sign {p['sign_consistent']}/{p['n']}")
    assert round(out["coco_gap"]["softmax"]["primary_all_splits_operational"]["median"], 1) == 22.1

    # ------------------------------------------------------------------- Ours/B
    D5 = json.load((SUPP / "D_5baseline_multimodel.json").open())
    pooled_b_anchor = {"ResNet-50 V2": 1.61, "ResNet-101 V2": 1.54, "ResNet-152 V2": 1.51}
    for name, blk in D5["models"].items():
        seed_ratio, pooled_cells = [], []
        for row in blk["per_seed"]:
            o = np.asarray(row["ours_widths"], dtype=np.float64)
            b = np.asarray(row["b_widths"], dtype=np.float64)
            fin = np.isfinite(o) & np.isfinite(b)
            r = o[fin] / b[fin]
            seed_ratio.append(float(np.median(r)))
            pooled_cells.append(r)
        pooled = float(np.median(np.concatenate(pooled_cells)))
        assert round(pooled, 2) == pooled_b_anchor[name], (name, pooled)
        out["ours_over_b"][name] = {
            "ratio_seed_level": summarize(np.array(seed_ratio), "above_1"),
            "pooled_median_reproduction": pooled,
            "published_pooled_median": pooled_b_anchor[name],
        }
        s = out["ours_over_b"][name]["ratio_seed_level"]
        print(f"Ours/B {name}: seed-level median {s['median']:.3f} "
              f"CI [{s['ci95']['lo']:.3f}, {s['ci95']['hi']:.3f}] sign {s['sign_consistent']}/{s['n']} "
              f"(pooled {pooled:.4f})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
