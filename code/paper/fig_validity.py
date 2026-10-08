"""Render the per-seed realized test-risk margin on six evaluation surfaces (fig:validity-dist).

The margin is 100 (R_sel^test - alpha) in percentage points, so all surfaces share the zero
threshold, and a point right of zero is a test exceedance: the certificate bounds the
population selected risk, and the realized test risk can exceed alpha by finite-test-sample
fluctuation. Inputs: results/imagenet_primary_real/results.json,
results/cifar100_full/results.json and, in results/ablation_supplement/,
B12_label_noise_robustness.json, G_coco_pixacc_g_softmax_a0.10_pi0.10_robust.json,
G_ade20k_mask2former_g_softmax.json and G_ade20k_segformer_g_softmax.json.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from style import COLORS, WIDTH_SINGLE_COL, RESULTS_DIR, load_json, save_fig


def cp_upper_one_sided(k: int, n: int, conf: float = 0.95) -> float:
    """Clopper-Pearson one-sided upper bound on the rate for k events of n."""
    if n == 0:
        return float("nan")
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    return float(beta.ppf(conf, k + 1, n - k))


def _per_seed(d):
    if isinstance(d, dict):
        if d.get("per_seed"):
            return d["per_seed"]
        for v in d.values():
            r = _per_seed(v)
            if r:
                return r
    if isinstance(d, list):
        for v in d[:3]:
            r = _per_seed(v)
            if r:
                return r
    return None


def imagenet_R(path, field):
    d = load_json(path)
    ps = d["per_seed"] if "per_seed" in d else _per_seed(d)
    return [s[field] for s in ps
            if s.get(field) is not None and not s.get("is_infeasible", False)]


def label_noise_R(path):
    d = load_json(path)
    out = []
    for row in d["rows"]:
        out += [s["R_test"] for s in row["per_seed"] if s.get("feasible")]
    return out


def field_R(path, field):
    """Realized risk over feasible seeds (skips INFEASIBLE splits with no value)."""
    return [s[field] for s in load_json(path)["per_seed"] if s.get(field) is not None]


def main() -> None:
    # (label, alpha, color, R-list)
    surfaces = [
        ("ImageNet\nRN50", 0.05, COLORS["ours"],
         imagenet_R("imagenet_primary_real/results.json", "test_R_sel_selected")),
        ("ImageNet\n+noise", 0.05, COLORS["wsr"],
         label_noise_R(str(RESULTS_DIR / "ablation_supplement/B12_label_noise_robustness.json"))),
        ("CIFAR-100", 0.15, COLORS["scrct"],
         imagenet_R("cifar100_full/results.json", "test_R_sel_selected")),
        ("COCO\npanoptic", 0.10, COLORS["bernstein"],
         field_R("ablation_supplement/G_coco_pixacc_g_softmax_a0.10_pi0.10_robust.json", "R_test_at_k")),
        ("ADE20K\nM2F", 0.20, COLORS["score"],
         field_R("ablation_supplement/G_ade20k_mask2former_g_softmax.json", "R_test_at_k")),
        ("ADE20K\nSegF", 0.20, COLORS["hoeffding"],
         field_R("ablation_supplement/G_ade20k_segformer_g_softmax.json", "R_test_at_k")),
    ]

    # aggregate exceedance check (per-surface n is too small for a per-surface CP)
    deltas = [0.05, 0.05, 0.10, 0.10, 0.10, 0.10]
    tot_n = sum(len(R) for _, _, _, R in surfaces)
    tot_exc = sum(int((np.array(R) > a).sum()) for (_, a, _, R) in surfaces)
    worst_exc_pp = max(
        ((np.array(R) - a).max() * 100.0
         for (_, a, _, R) in surfaces if (np.array(R) > a).any()),
        default=float("nan"))
    agg_ub = cp_upper_one_sided(tot_exc, tot_n)
    strict_delta = min(deltas)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_COL, WIDTH_SINGLE_COL * 0.92))
    n = len(surfaces)

    EXC = "#444444"
    # exceedance zone (margin > 0, in pp) + zero (= alpha) threshold
    ax.axvspan(0, 4, color=COLORS["hoeffding"], alpha=0.08, zorder=0)
    ax.axvline(0, color=COLORS["neutral"], lw=1.0, ls="--", zorder=1)

    for i, (lab, a, col, R) in enumerate(surfaces):
        y = n - 1 - i                             # first surface on top
        m = (np.array(R) - a) * 100.0             # percentage points
        jit = rng.uniform(-0.15, 0.15, size=len(m))
        within = m <= 0
        ax.scatter(m[within], y + jit[within], s=8, color=col, edgecolor="white",
                   linewidth=0.3, zorder=3, alpha=0.9)
        over = ~within
        if over.any():
            ax.scatter(m[over], y + jit[over], s=54, facecolors=col,
                       edgecolors="#111111", marker="o", linewidth=1.3, zorder=5)
        q1, med, q3 = np.percentile(m, [25, 50, 75])
        ax.add_patch(plt.Rectangle((q1, y - 0.26), q3 - q1, 0.52, fill=False,
                                   edgecolor=col, linewidth=1.1, zorder=2))
        ax.plot([med, med], [y - 0.26, y + 0.26], color=col, lw=1.8, zorder=2)

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [rf"{lab}  $n{{=}}{len(R)},\,\alpha{{=}}{a:.2f}$"
         for lab, a, _, R in reversed(surfaces)],
        fontsize=6.0)
    ax.set_xlabel(r"realized test-risk margin  $100{\cdot}(R_{\mathrm{sel}}^{\mathrm{test}} - \alpha)$  (pp)")
    ax.set_xticks([-15, -10, -5, 0])
    ax.set_xlim(-16, 4)
    ax.set_ylim(-0.95, n - 0.4)
    ax.tick_params(axis="x", labelsize=7)

    ax.text(-15.5, n - 0.62, r"test risk $\leq\alpha$", fontsize=6.2,
            color=COLORS["neutral"], ha="left", va="top")
    ax.text(2.0, n - 0.62, "test\nexceedance", fontsize=6.0, color=COLORS["hoeffding"],
            ha="center", va="top", linespacing=0.9)
    ax.annotate(rf"$1/20$, ${worst_exc_pp:+.2f}$ pp", xy=(0.25, 0.16), xytext=(2.6, 0.95),
                fontsize=6.2, color=EXC, ha="center",
                arrowprops=dict(arrowstyle="->", color=EXC, lw=0.8))

    ax.text(-15.5, -0.80,
            rf"{tot_n} feasible risk evaluations $\cdot$ {tot_exc} test exceedance (${worst_exc_pp:+.2f}$ pp)",
            fontsize=6.0, color=COLORS["neutral"], ha="left", va="center", style="italic")

    save_fig(fig, "validity")
    print(f"[check] aggregate {tot_exc}/{tot_n} exceedances, CP-95 one-sided UB={agg_ub:.4f}; "
          f"strictest delta={strict_delta}; per-surface n="
          f"{[len(R) for _,_,_,R in surfaces]}")


if __name__ == "__main__":
    main()
