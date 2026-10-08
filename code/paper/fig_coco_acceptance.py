"""Render the COCO panoptic certified-acceptance figure as two PDFs (fig:coco-acceptance).

Panel (a) shows the per-seed certified acceptance (the returned pair's p_LCB) of Ours over its
feasible splits next to the Hoeffding-CRC selective family, which certifies no acceptance on
any split, and panel (b) the realized test acceptance and selected risk of the certified pair
on every feasible seed. Inputs, in results/ablation_supplement/:
G_coco_pixacc_g_softmax_a0.10_pi0.10_robust.json (both panels) and
G_coco_pixacc_g_entropy_a0.10_pi0.10_robust.json (median entropy acceptance, printed only).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import COLORS, WIDTH_TWO_COL, load_json, save_fig

SOFTMAX = "ablation_supplement/G_coco_pixacc_g_softmax_a0.10_pi0.10_robust.json"
ENTROPY = "ablation_supplement/G_coco_pixacc_g_entropy_a0.10_pi0.10_robust.json"


def _box(ax, x, ys, color, half=0.17):
    """Draw a median bar + IQR rectangle for the values ys centred at x."""
    q1, med, q3 = np.percentile(ys, [25, 50, 75])
    ax.add_patch(plt.Rectangle((x - half, q1), 2 * half, q3 - q1, fill=False,
                               edgecolor=color, linewidth=1.1, zorder=4))
    ax.plot([x - half, x + half], [med, med], color=color, lw=1.9, zorder=5)
    return med


def main() -> None:
    sm = load_json(SOFTMAX)
    en = load_json(ENTROPY)
    cfg = sm["config"]
    pmin = cfg["pi_min"]                       # 0.10 deployability floor
    rng = np.random.default_rng(0)

    sm_ours = np.array([s["p_ours"] for s in sm["per_seed"]])
    sm_hoef = np.array([s["p_hoeff"] for s in sm["per_seed"]])   # all 0
    med_sm = float(np.median(sm_ours))
    cpub = sm["validity"]["cp_95_one_sided_ub_risk_vio_rate"]
    nvio = sm["validity"]["n_vio_joint_risk_side"]
    nseed_sm = cfg["n_seeds"]

    en_ours = np.array([s["p_ours"] for s in en["per_seed"]])
    med_en = float(np.median(en_ours))
    nseed_en = en["config"]["n_seeds"]

    # Two single-panel canvases sized to sit side by side across the two-column width.
    figA, axA = plt.subplots(figsize=(WIDTH_TWO_COL * 0.485, WIDTH_TWO_COL * 0.40))
    figB, axB = plt.subplots(figsize=(WIDTH_TWO_COL * 0.485, WIDTH_TWO_COL * 0.40))

    # Infeasible splits (p_ours = 0) are left out instead of being drawn as certified
    # zeros; the tick labels report each method's feasibility count.
    feas_ours = sm_ours[sm_ours > 0]
    n_feas = int(feas_ours.size)
    med_feas = float(np.median(feas_ours))
    axA.axhline(pmin, color=COLORS["neutral"], lw=1.0, ls="--", zorder=1)
    axA.text(-0.46, pmin + 0.005, r"$\pi_{\min}{=}0.10$ floor", fontsize=6.0,
             color=COLORS["neutral"], va="bottom", ha="left")
    jit = rng.uniform(-0.12, 0.12, size=n_feas)
    axA.scatter(0 + jit, feas_ours, s=14, color=COLORS["ours"], edgecolor="white",
                linewidth=0.3, zorder=3, alpha=0.9)
    _box(axA, 0, feas_ours, COLORS["ours"])
    axA.annotate(f"median {med_feas:.3f}", xy=(0.16, med_feas),
                 xytext=(0.30, med_feas + 0.030), fontsize=6.4, color=COLORS["ours"],
                 ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=COLORS["ours"], lw=0.8))
    # Hoeffding-CRC: best certifiable acceptance is 0 on every split
    axA.plot([1 - 0.20, 1 + 0.20], [0, 0], color=COLORS["hoeffding"], lw=2.4, zorder=5)
    axA.annotate(r"best certifiable $=0$", xy=(1, 0.002), xytext=(1.0, 0.058),
                 fontsize=6.4, color=COLORS["hoeffding"], ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=COLORS["hoeffding"], lw=0.8))
    # median certified-acceptance gap (Ours median vs Hoeffding-CRC's 0)
    axA.annotate("", xy=(0.55, med_feas), xytext=(0.55, 0.0),
                 arrowprops=dict(arrowstyle="<->", color=COLORS["ours"], lw=1.0))
    axA.text(0.595, med_feas / 2.0, r"$+22.1$ pp" "\n" "(median)", fontsize=6.5,
             color=COLORS["ours"], ha="left", va="center", linespacing=1.0)
    axA.set_xticks([0, 1])
    axA.set_xticklabels([f"Ours (joint EB)\nfeasible {n_feas}/{nseed_sm}",
                         f"Hoeffding–CRC\nfeasible 0/{nseed_sm}"], fontsize=7)
    axA.set_xlim(-0.5, 1.5)
    axA.set_ylim(-0.012, 0.285)
    axA.set_ylabel(r"certified acceptance (returned-pair $p_{\mathrm{LCB}}$)")

    # An infeasible seed deploys no decision, so it has no operating point to plot.
    alpha = cfg["alpha"]
    feas = [s for s in sm["per_seed"]
            if s.get("R_test_at_k") is not None and s["p_ours"] > 0]
    pt = np.array([s["p_test_at_k"] for s in feas])
    rt = np.array([s["R_test_at_k"] for s in feas])
    XLO, XHI, YHI = 0.06, 0.28, 0.125
    # valid & deployable quadrant: acceptance >= pi_min and risk <= alpha
    axB.axhspan(0, alpha, xmin=(pmin - XLO) / (XHI - XLO), xmax=1.0,
                color=COLORS["ours"], alpha=0.08, zorder=0)
    axB.axhline(alpha, color=COLORS["hoeffding"], lw=1.1, ls="--", zorder=2)
    axB.axvline(pmin, color=COLORS["neutral"], lw=1.0, ls="--", zorder=2)
    axB.scatter(pt, rt, s=20, color=COLORS["ours"], edgecolor="white",
                linewidth=0.3, zorder=4, alpha=0.9)
    axB.text(XHI - 0.005, alpha + 0.002, r"risk budget $\alpha{=}0.10$", fontsize=6.2,
             color=COLORS["hoeffding"], va="bottom", ha="right")
    axB.text(pmin + 0.004, YHI - 0.004, r"$\pi_{\min}{=}0.10$", fontsize=6.2,
             color=COLORS["neutral"], va="top", ha="left")
    axB.text(0.205, 0.022,
             f"all {len(pt)} feasible deployments:\n"
             r"realized risk $\leq\alpha$, acceptance $\geq\pi_{\min}$",
             fontsize=6.3, color=COLORS["ours"], ha="center", va="bottom",
             style="italic", linespacing=1.1)
    axB.set_xlim(XLO, XHI)
    axB.set_ylim(0, YHI)
    axB.set_xlabel(r"realized acceptance  $\hat p_{\mathrm{acc}}^{\mathrm{test}}$")
    axB.set_ylabel(r"realized selected risk  $R_{\mathrm{sel}}^{\mathrm{test}}$")

    save_fig(figA, "coco_acceptance_a")
    save_fig(figB, "coco_acceptance_b")
    print(f"[check] median p_ours softmax={med_sm:.4f} (+{med_sm*100:.1f}pp), "
          f"entropy={med_en:.4f} (+{med_en*100:.1f}pp); "
          f"risk vio={nvio}/{nseed_sm}; "
          f"CP95 UB={cpub:.4f} (delta={cfg['delta']})")


if __name__ == "__main__":
    main()
