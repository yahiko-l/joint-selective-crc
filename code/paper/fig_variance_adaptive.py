"""Render the variance-adaptive payoff figure (fig:variance-adaptive).

Panel (a) shows the per-pair width ratio Ours / A(pi_min) on three ImageNet ResNet V2
backbones at pi_min = 0.01, and panel (b) sweeps pi_min on ResNet-50 V2, plotting over the
low-acceptance pairs the median nominal ratio against A(pi_min) and the median sign-aware
valid UCB-excess ratio against sign-aware Hoeffding-CP. Inputs, all in
results/ablation_supplement/: D_5baseline_multimodel.json,
A1_imagenet_resnet50v2_pi_min_sweep.json and E_signaware_valid_ratio.json.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import (
    COLORS,
    MARKERS,
    WIDTH_TWO_COL,
    load_json,
    save_fig,
)


def main() -> None:
    # This figure uses its own teal and coral palette instead of the shared COLORS.
    C_CLOUD  = "#2196a0"
    C_BOX    = "#08474d"
    C_R_OURS = "#138d96"
    C_R_PLCB = "#e8714a"
    d = load_json("ablation_supplement/D_5baseline_multimodel.json")
    rn50_sweep = load_json("ablation_supplement/A1_imagenet_resnet50v2_pi_min_sweep.json")["sweep"]
    signaware_rows = load_json(
        "ablation_supplement/E_signaware_valid_ratio.json")["fig8b_sweep_signaware"]["rows"]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(WIDTH_TWO_COL, WIDTH_TWO_COL * 0.36),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.24},
    )

    models = ["ResNet-50 V2", "ResNet-101 V2", "ResNet-152 V2"]
    short = ["RN50", "RN101", "RN152"]
    top1 = [d["models"][m]["top1"] for m in models]
    rng = np.random.default_rng(0)
    x = np.arange(len(models))

    ax_left.axhspan(1.0, 6.0, color=COLORS["neutral"], alpha=0.05, zorder=0)
    ax_left.axhline(1.0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, zorder=1)

    for xi, name in zip(x, models):
        m = d["models"][name]
        r = np.concatenate([np.asarray(s["ours_widths"]) / np.asarray(s["a_widths"])
                            for s in m["per_seed"]])
        jit = rng.uniform(-0.20, 0.20, size=r.size)
        ax_left.scatter(xi + jit, r, s=4, color=C_CLOUD, alpha=0.16,
                       edgecolor="none", zorder=2)
        q1, med, q3 = np.percentile(r, [25, 50, 75])
        ax_left.add_patch(plt.Rectangle((xi - 0.27, q1), 0.54, q3 - q1, fill=False,
                                       edgecolor=C_BOX, linewidth=1.1, zorder=4))
        ax_left.plot([xi - 0.27, xi + 0.27], [med, med], color=C_BOX, lw=2.0, zorder=5)
        fold = 1.0 / m["paired_ratios_all"]["ours_over_a"]
        ax_left.text(xi, r.max() * 1.6, rf"${fold:.0f}{{\times}}$", ha="center",
                    va="bottom", fontsize=7.4, color=C_BOX, fontweight="bold")

    # A(pi_min) has constant width 1.05 >= B = 1 at this operating point, hence the
    # "vacuous" label on the ratio-1 line.
    ax_left.text(0.985, 1.07, r"$=A(\pi_{\min})$ (vacuous, $\geq B{=}1$)",
                transform=ax_left.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=6.0, color=COLORS["neutral"])

    ax_left.set_yscale("log")
    ax_left.set_ylim(1e-3, 6.0)
    ax_left.set_xlim(-0.5, len(models) - 0.5)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"{s}\n(top-1 {t:.3f})" for s, t in zip(short, top1)], fontsize=7)
    ax_left.set_ylabel(r"per-pair width ratio  Ours$/A$")
    ax_left.set_yticks([0.001, 0.01, 0.1, 1.0])
    ax_left.set_yticklabels(["0.001", "0.01", "0.1", "1.0"])
    ax_left.set_title(r"(a) Per-pair $R_{\mathrm{sel}}$-width ratio at $\pi_{\min}{=}0.01$, all 35 grid pairs",
                     fontsize=7.5, pad=4)

    pis = [r["pi_min"] for r in rn50_sweep]
    a_low = [r["median_ratio_ours_A_LOW"] for r in rn50_sweep]
    sa_pis = [r["pi_min"] for r in signaware_rows]
    sa_valid = [r["median_signaware_excess_ratio_LOW"] for r in signaware_rows]

    ax_right.plot(pis, a_low,
                marker=MARKERS["ours"], color=C_R_OURS, markersize=4.5,
                linewidth=1.2, label=r"vs $A$ (Hoeffding$/\pi_{\min}$, nominal)",
                markerfacecolor="white", markeredgewidth=1.0)
    ax_right.plot(sa_pis, sa_valid,
                marker=MARKERS["bernstein"], color=C_R_PLCB, markersize=4.5,
                linewidth=1.2,
                label=r"vs sign-aware Hoeffding--CP (valid excess)",
                markerfacecolor="white", markeredgewidth=1.0)
    ax_right.axhline(1.0, color=COLORS["neutral"], linestyle="--", linewidth=0.7)

    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.set_xlabel(r"acceptance floor $\pi_{\min}$")
    ax_right.set_ylabel(r"Ours / baseline ratio")
    ax_right.set_xticks([0.005, 0.01, 0.02, 0.05, 0.10])
    ax_right.set_xticklabels(["0.005", "0.01", "0.02", "0.05", "0.10"], fontsize=6.5)
    ax_right.set_yticks([0.05, 0.1, 0.3, 1.0])
    ax_right.set_yticklabels(["0.05", "0.1", "0.3", "1.0"])
    ax_right.set_ylim(0.04, 1.5)
    ax_right.legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=1.8)
    ax_right.set_title(r"(b) RN50 V2 $\pi_{\min}$ sweep on low-acceptance subset",
                     fontsize=7.5, pad=4)

    save_fig(fig, "variance_adaptive")


if __name__ == "__main__":
    main()
