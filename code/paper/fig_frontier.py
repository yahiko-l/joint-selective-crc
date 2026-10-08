"""Render the certified operating frontier on three ImageNet backbones (fig:cert-frontier).

Each point is one acceptance tier (one tau), plotted at its acceptance against the smallest
sign-aware valid UCB on R_sel over the seven lambda values: filled where the method certifies
the tier, a hollow triangle where it is infeasible, capped at the top edge when off-scale.
Input: results/analysis/certified_decision_frontier_signaware.json, written by
experiments/analysis_signaware_valid_ratio.py.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import COLORS, WIDTH_TWO_COL, load_json, save_fig

YMAX = 0.065          # top of the visible risk-UCB axis (alpha=.05 sits at ~77%)
CARET_Y = YMAX * 0.955


def tiers_of(model):
    """Group the 35 pairs by tau (acceptance tier); return list sorted by p_acc.

    Each tier: dict(p_acc, best, cert, lo, hi) per method, where best = min UCB
    over the 7 lambda, cert = any lambda certified, [lo,hi] = lambda UCB spread.
    """
    by_tau = {}
    for p in model["pairs"]:
        by_tau.setdefault(p["tau_idx"], []).append(p)
    out = []
    for ti, pts in by_tau.items():
        p_acc = float(np.median([p["p_acc"] for p in pts]))
        rec = {"p_acc": p_acc}
        for mth in ("ours", "a_pi_min", "a_plcb"):
            u = [p["ucb_rsel"][mth] for p in pts if p["ucb_rsel"][mth] is not None]
            cu = [p["ucb_rsel"][mth] for p in pts
                  if p["cert"][mth] and p["ucb_rsel"][mth] is not None]
            cert = any(p["cert"][mth] for p in pts)
            rec[mth] = {
                "best": (min(u) if u else None),
                # lambda spread over certified lambda only (bounded by alpha)
                "lo": (min(cu) if cu else None),
                "hi": (max(cu) if cu else None),
                "cert": cert,
            }
        out.append(rec)
    return sorted(out, key=lambda r: r["p_acc"])


def draw_method(ax, tiers, mth, color, ls, marker, alpha_line):
    """Frontier line through certified tiers + open/caret markers for infeasible."""
    xs_c, ys_c, xs_o, ys_o, xs_off = [], [], [], [], []
    for t in tiers:
        m = t[mth]
        if m["best"] is None:
            continue
        x, y = t["p_acc"], m["best"]
        if m["cert"]:
            xs_c.append(x); ys_c.append(y)
        elif y <= YMAX:
            xs_o.append(x); ys_o.append(y)
        else:
            xs_off.append(x)
    if xs_c:
        ax.plot(xs_c, ys_c, color=color, ls=ls, lw=1.5, marker=marker,
                mfc=color, mec="white", mew=0.4, ms=5, zorder=5, alpha=alpha_line)
    # infeasible tiers: hollow triangles at the true UCB, capped at the top edge when
    # off-scale (colour still encodes method)
    if xs_o:
        ax.scatter(xs_o, ys_o, s=30, facecolors="none", edgecolors=color,
                   linewidths=1.0, marker="^", zorder=4, alpha=0.9)
    if xs_off:
        ax.scatter(xs_off, [CARET_Y] * len(xs_off), s=34, marker="^",
                   facecolors="none", edgecolors=color, linewidths=1.0, zorder=4)


def main() -> None:
    d = load_json("analysis/certified_decision_frontier_signaware.json")
    alpha = d["config"]["alpha"]
    models = [("ResNet-50 V2", "ResNet-50", 0.32),
              ("ResNet-101 V2", "ResNet-101", 0.88),
              ("ResNet-152 V2", "ResNet-152", 0.88)]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_TWO_COL, 2.75), sharey=True)

    for ax, (key, short, xmax) in zip(axes, models):
        model = d["models"][key]
        tiers = tiers_of(model)

        ax.axhline(alpha, color=COLORS["neutral"], lw=1.0, ls="--", zorder=1)

        only = [t["p_acc"] for t in tiers if t["ours"]["cert"] and not t["a_plcb"]["cert"]]
        apl = [t["p_acc"] for t in tiers if t["a_plcb"]["cert"]]
        if only:
            lo = max(0.0, min(only) - 0.025)
            hi = max(only) + 0.035          # tight bracket around the ours-only tiers
            if apl:
                hi = min(hi, (max(only) + min(apl)) / 2)
            ax.axvspan(lo, hi, color=COLORS["ours"], alpha=0.09, zorder=0)
            ax.text((lo + hi) / 2, YMAX * 0.6, "feasible only w/ Ours",
                    fontsize=5.8, color=COLORS["ours"], ha="center", va="center",
                    rotation=90, style="italic", zorder=6)

        # A(pi_min) is about 1.0 at every tier, so it appears only as grey carets along
        # the top edge.
        draw_method(ax, tiers, "a_pi_min", COLORS["neutral"], ":", "^", 0.6)
        draw_method(ax, tiers, "a_plcb", COLORS["hoeffding"], (0, (4, 2)), "s", 0.95)
        draw_method(ax, tiers, "ours", COLORS["ours"], "-", "o", 1.0)

        top = tiers[-1]
        uo, ua = top["ours"]["best"], top["a_plcb"]["best"]
        if uo and ua:
            ax.annotate(f"same max acc.;\n{ua / uo:.1f}$\\times$ lower valid UCB",
                        xy=(top["p_acc"], uo),
                        xytext=(top["p_acc"] * 0.5, YMAX * 0.62),
                        fontsize=5.5, color=COLORS["ours"], ha="center", va="center",
                        arrowprops=dict(arrowstyle="->", color=COLORS["ours"], lw=0.8))

        pa_max = max(t["p_acc"] for t in tiers if t["ours"]["cert"])
        ax.text(0.5, -0.0052, rf"max certified acceptance ${pa_max * 100:.1f}\%$",
                transform=ax.get_yaxis_transform(), fontsize=6.2,
                color=COLORS["ours"], ha="center", va="center")

        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.0085, YMAX)
        ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        ax.set_title(f"{short}  (top-1 {model['top1']:.3f})", fontsize=7.5)
        ax.set_xlabel(r"acceptance  $\hat p_{\mathrm{acc}}$  (filled $=$ certified)")
        ax.tick_params(labelsize=6.6)

    axes[0].set_ylabel(r"valid UCB on $R_{\mathrm{sel}}$ (sign-aware)  ($\downarrow$ better)")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLORS["ours"], marker="o", mfc=COLORS["ours"],
               mec="white", ls="-", lw=1.5, ms=5, label="Ours (joint EB)"),
        Line2D([0], [0], color=COLORS["hoeffding"], marker="s", mfc=COLORS["hoeffding"],
               mec="white", ls=(0, (4, 2)), lw=1.5, ms=5,
               label=r"$A(\mathrm{CP}^{\pm})$ (sign-aware Hoeffding--CP)"),
        Line2D([0], [0], color=COLORS["neutral"], ls="--", lw=1.0, label=r"$\alpha$ certify ceiling"),
        Line2D([0], [0], color="0.35", marker="^", mfc="none", mec="0.35",
               ls="none", ms=6, label="hollow $\\triangle$: infeasible (capped if off-scale)"),
        Line2D([0], [0], color=COLORS["neutral"], marker="^", mfc="none",
               mec=COLORS["neutral"], ls="none", ms=6,
               label=r"$A(\pi_{\min})$: off-scale (0/35 certified)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=6.2, bbox_to_anchor=(0.5, 1.12), handletextpad=0.4,
               columnspacing=1.4)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_fig(fig, "frontier")


if __name__ == "__main__":
    main()
