"""Render the closed-form width-scaling figure (fig:width-scaling) as three single-panel PDFs.

Panel (a) compares the certified R_sel margin of Ours with the textbook Hoeffding-CRC margin
A(pi_min) across pi_min, panel (b) the per-pair half-widths of Ours and Hoeffding against
n_cert at the COCO accepted-sample variance, and panel (c) the same half-widths against the
accepted-sample variance T_obs, which cross at sigma^2_*(s). Every curve is a closed-form
function of (n, pi_min, m, delta, B), so no result file is read.
"""

from __future__ import annotations

import numpy as np

from style import (
    COLORS,
    WIDTH_TWO_COL,
    save_fig,
)
import matplotlib.pyplot as plt


# Closed-form half-width / margin expressions (B = 1 throughout the paper).
def L_O(m: float, delta: float) -> float:
    """log(64 m / delta), the two-tailed Maurer-Pontil and Chernoff union log term (U1-U4)."""
    return np.log(64.0 * m / delta)


def L_H(m: float, delta: float) -> float:
    """log(m / delta), the per-pair Hoeffding log term of Corollary 9."""
    return np.log(m / delta)


def gamma_r(n, pi_min, m, delta, B=1.0):
    """Ours certified Rsel-margin, Theorem 1."""
    Lo = L_O(m, delta)
    return 4.0 * B * np.sqrt(Lo / (n * pi_min)) + (14.0 * B / 3.0) * Lo / (pi_min * (n - 1.0))


def hoeff_textbook(n, pi_min, m, delta, B=1.0):
    """Textbook pi_min-saturated Hoeffding-CRC selective margin
    A(pi_min) = (B/pi_min) sqrt(log(2 m / delta) / (2 n))."""
    return (B / pi_min) * np.sqrt(np.log(2.0 * m / delta) / (2.0 * n))


def ours_perpair_width(T_obs, n, p_hat, m, delta, B=1.0):
    """Ours per-pair half-width UCB_ours - Rhat = eta_Z / p_hat on the Rsel scale
    (Corollary 9), with Sigma_Z ~ p_hat * T_obs to leading order."""
    Lo = L_O(m, delta)
    sigma_z = p_hat * T_obs
    eta_z = np.sqrt(2.0 * sigma_z * Lo / n) + 7.0 * B * Lo / (3.0 * (n - 1.0))
    return eta_z / p_hat


def hoeff_perpair_width(s, m, delta, B=1.0):
    """Per-pair Hoeffding half-width B sqrt(log(m/delta)/(2 s)), variance-blind
    (UCB_Hoeff - Rhat)."""
    return B * np.sqrt(L_H(m, delta) / (2.0 * s))


def sigma_star(s, m, delta, B=1.0):
    """Crossing threshold sigma^2_*(s) of Corollary 9, a variance on the T_obs scale."""
    Lo = L_O(m, delta)
    Lh = L_H(m, delta)
    inner = np.maximum(np.sqrt(Lh / 2.0) - 7.0 * Lo / (3.0 * np.sqrt(s)), 0.0)
    return (B ** 2) / (2.0 * Lo) * inner ** 2


def n_star(pi_min, m, delta):
    """Sample-size condition (star): n_cert >= 32 log(32 m / delta) / pi_min
    (Assumption 5). Below this the inclusion and external-oracle tier is not asserted."""
    return 32.0 * np.log(32.0 * m / delta) / pi_min


PANEL_W = 0.33          # per-subfigure width as a fraction of WIDTH_TWO_COL
UNIT_SUFFIX = r"(loss units, $B{=}1$)"


def _new_ax():
    fig, ax = plt.subplots(figsize=(WIDTH_TWO_COL * PANEL_W, WIDTH_TWO_COL * 0.34))
    return fig, ax


def _unit_note(ax):
    """Unified-unit suffix under the y-label."""
    ax.text(-0.315, 0.5, UNIT_SUFFIX, transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontsize=5.2)


def main() -> None:
    B = 1.0
    # ImageNet headline operating point.
    M_CLS, D_CLS, NCERT_CLS, PMIN_CLS = 35, 0.05, 33_000, 0.01
    # Segmentation operating point (COCO certifier-selected pair).
    M_SEG, D_SEG, NCERT_SEG, PMIN_SEG = 15, 0.10, 4_000, 0.10
    S_COCO, PHAT_COCO = 880.0, 0.22          # certifier-selected COCO pair
    TOBS_COCO, TOBS_ADE = 0.007, 0.092       # observed on COCO and ADE20K

    dash = (0, (2, 1.5))
    LDASH = (0, (5, 2))   # long-dash for Hoeffding (grayscale-distinct from solid Ours)

    figA, axA = _new_ax()
    pmin = np.logspace(np.log10(2e-3), np.log10(1e-1), 200)
    g = gamma_r(NCERT_CLS, pmin, M_CLS, D_CLS, B)
    a = hoeff_textbook(NCERT_CLS, pmin, M_CLS, D_CLS, B)
    pstar = n_star(1.0, M_CLS, D_CLS) / NCERT_CLS   # pi_min where (star) binds at n=33k
    valid = pmin >= pstar
    inval = pmin <= pstar

    axA.loglog(pmin[inval], a[inval], color=COLORS["hoeffding"], lw=1.1, ls=dash, alpha=0.40)
    axA.loglog(pmin[inval], g[inval], color=COLORS["ours"], lw=1.1, ls=dash, alpha=0.40)
    axA.loglog(pmin[valid], a[valid], color=COLORS["hoeffding"], lw=1.5, ls=LDASH,
               label=r"Hoeffding margin $A\!\propto\!\pi_{\min}^{-1}$")
    axA.loglog(pmin[valid], g[valid], color=COLORS["ours"], lw=1.5,
               label=r"Ours margin $\gamma_r\!\propto\!\pi_{\min}^{-1/2}$")
    axA.loglog(pmin, 0.0105 / pmin, color=COLORS["hoeffding"], lw=0.6, ls=":", alpha=0.5)
    axA.loglog(pmin, 0.072 / np.sqrt(pmin), color=COLORS["ours"], lw=0.6, ls=":", alpha=0.5)
    axA.axhline(B, color=COLORS["neutral"], lw=0.7, ls="--")
    axA.text(9.5e-2, B * 1.18, r"$B$ (saturated)", fontsize=5.6,
             color=COLORS["neutral"], va="bottom", ha="right")
    axA.axvspan(2e-3, pstar, color=COLORS["neutral"], alpha=0.12)
    axA.axvline(pstar, color=COLORS["neutral"], lw=0.9, ls="-", alpha=0.7)
    axA.text(2.25e-3, 7.5, "no stronger tier" "\n" r"below ($\star$)", fontsize=5.2,
             color=COLORS["neutral"], rotation=90, va="top", linespacing=0.9)
    axA.text(pstar * 1.12, 9e-2, r"$\pi_\star{\approx}0.0097$", fontsize=5.2,
             color=COLORS["neutral"], va="bottom", ha="left")
    axA.set_xlabel(r"acceptance floor $\pi_{\min}$")
    axA.set_ylabel(r"certified $R_{\mathrm{sel}}$ margin")
    _unit_note(axA)
    axA.set_xlim(2e-3, 1e-1)
    axA.set_ylim(8e-2, 1e1)
    axA.legend(loc="upper right", fontsize=5.6, frameon=False, handlelength=1.4)
    save_fig(figA, "width_scaling_a")

    figB, axB = _new_ax()
    n = np.logspace(np.log10(3e3), 5, 200)
    s = PHAT_COCO * n
    wb_ours = ours_perpair_width(TOBS_COCO, n, PHAT_COCO, M_SEG, D_SEG, B)
    wb_hoeff = hoeff_perpair_width(s, M_SEG, D_SEG, B)
    axB.loglog(n, wb_hoeff, color=COLORS["hoeffding"], lw=1.5, ls=LDASH,
               label=r"per-pair Hoeffding $B\sqrt{L_H/2s}$")
    axB.loglog(n, wb_ours, color=COLORS["ours"], lw=1.5,
               label=r"per-pair Ours $\eta_Z/\hat p$")
    axB.set_xlabel(r"certification samples $n_{\mathrm{cert}}$")
    axB.set_ylabel(r"per-pair $R_{\mathrm{sel}}$ half-width")
    _unit_note(axB)
    axB.set_xlim(3e3, 1e5)
    axB.set_ylim(2e-3, 1e-1)
    axB.legend(loc="upper right", fontsize=6.0, frameon=False, handlelength=1.4)
    save_fig(figB, "width_scaling_b")

    figC, axC = _new_ax()
    T = np.linspace(0.0, 0.15, 200)
    w_ours = ours_perpair_width(T, NCERT_SEG, PHAT_COCO, M_SEG, D_SEG, B)
    w_hoeff = hoeff_perpair_width(S_COCO, M_SEG, D_SEG, B)
    sstar = sigma_star(S_COCO, M_SEG, D_SEG, B)

    axC.fill_between(T, 0, 0.09, where=(T < sstar), color=COLORS["ours"], alpha=0.07)
    axC.text(sstar * 0.5, 0.071, "ours\ntighter", fontsize=5.0, color=COLORS["ours"],
             ha="center", va="center", alpha=0.8, linespacing=0.9)
    axC.axhline(w_hoeff, color=COLORS["hoeffding"], lw=1.5, ls=LDASH,
                label=r"per-pair Hoeffding (blind)")
    axC.plot(T, w_ours, color=COLORS["ours"], lw=1.5,
             label=r"per-pair Ours (adaptive)")
    axC.axvline(sstar, color=COLORS["neutral"], lw=0.8, ls="--")
    axC.text(sstar + 0.006, 0.012, r"$\sigma^2_{*}(s){\approx}0.041$", fontsize=5.6,
             rotation=90, va="bottom", ha="left", color=COLORS["neutral"],
             bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
    for tv, lab, col, ls_, xoff in [
        (TOBS_COCO, "COCO", COLORS["bernstein"], ":", 0.004),
        (TOBS_ADE, "ADE20K", COLORS["score"], (0, (4, 1, 1, 1)), 0.004),
    ]:
        axC.axvline(tv, color=col, lw=0.9, ls=ls_)
        axC.text(tv + xoff, 0.083, lab, fontsize=5.8, color=col, ha="center", va="top",
                 rotation=90)
    axC.set_xlabel(r"accepted-sample variance $T_{\mathrm{obs}}$")
    axC.set_ylabel(r"per-pair $R_{\mathrm{sel}}$ half-width")
    _unit_note(axC)
    axC.set_xlim(0, 0.15)
    axC.set_ylim(0, 0.09)
    axC.legend(loc="lower right", fontsize=5.8, frameon=False, handlelength=1.4)
    save_fig(figC, "width_scaling_c")


if __name__ == "__main__":
    main()
