"""Render the eight-ingredient stress matrix (tab:ingredient).

Each row removes or weakens one ingredient of the certifier in a regime chosen to expose
its failure mode. Inputs, all in results/ablation_supplement/: A16_no_cp_isolation.json,
A17_imagenet_end_to_end_no_mp_var.json, wave2_adversarial.json, wave_p3_misc.json and
B5_extended_3_new_ablations.json.
"""

from __future__ import annotations

from style import load_json, write_tex


def feas_fraction(counts: set, n: int) -> str:
    return f"{counts.pop()}/{n}" if len(counts) == 1 else str(sorted(counts))


def main() -> None:
    cp_only = load_json("ablation_supplement/A16_no_cp_isolation.json")
    no_mp_var = load_json("ablation_supplement/A17_imagenet_end_to_end_no_mp_var.json")
    adversarial = load_json("ablation_supplement/wave2_adversarial.json")
    misc = load_json("ablation_supplement/wave_p3_misc.json")
    extended = load_json("ablation_supplement/B5_extended_3_new_ablations.json")
    split = adversarial["A5_no3split_leakage"]
    margin = adversarial["A11_no_margin_adversarial"]

    mp_ratios = [misc["A12"][k]["ratio_range_over_mp"] for k in ("bimodal", "uniform", "beta")]

    low_pi = [r for r in cp_only["sweep"] if r["pi_min"] <= 0.005]
    cp_lo = min(r["cp_pass_rate"] for r in low_pi)
    cp_hi = max(r["cp_pass_rate"] for r in low_pi)
    pi_lo = min(r["pi_min"] for r in low_pi)
    pi_hi = max(r["pi_min"] for r in low_pi)

    ours_feas = feas_fraction({r["ours_n_feasible"] for r in no_mp_var["sweep"]}, 10)
    nmpv_feas = feas_fraction({r["no_mp_var_n_feasible"] for r in no_mp_var["sweep"]}, 10)

    s = extended["summary"]
    two_sided_fold = s["NO_TWO_SIDED_MP"]["median_width_r"] / extended["ours_median_width_r"]

    n_trials = margin["n_trials"]
    vio_no_margin = margin["vio_rate_no_margin"]
    vio_margin = margin["vio_rate_with_margin"]

    rows = [
        ("1", r"CP-LCB on $p_{\mathrm{acc}}$",
         rf"$\pi_{{\min}} \in [{pi_lo},{pi_hi}]$ synthetic binomial",
         rf"CP pass-rate {cp_lo * 100:.0f}--{cp_hi * 100:.0f}\% \emph{{vs}} Hoeffding \textbf{{0\%}}",
         "empirical"),
        ("2", r"MP-variance (real ImageNet end-to-end)",
         r"low-$\pi_{\min}$ feasibility, $5$ settings",
         rf"\textbf{{{ours_feas}}} feas \emph{{vs}} \textbf{{{nmpv_feas}}} without MP-variance",
         "empirical"),
        ("3", r"Margin in oracle",
         r"boundary-acceptance adversarial pairs",
         rf"vio-rate ${vio_no_margin:.3f}$~(${round(vio_no_margin * n_trials)}/{n_trials}$) without margin "
         rf"\emph{{vs}} ${vio_margin:.3f}$~(${round(vio_margin * n_trials)}/{n_trials}$) with it",
         "empirical"),
        ("4", r"Three-split protocol",
         r"adversarial cert-data leakage, $1000$ trials",
         rf"vio-rate ${split['vio_rate_no3split']:.3f}$ without three-split \emph{{vs}} "
         rf"$\delta{{=}}{split['delta_target']}$ (fails); ${split['vio_rate_ours']:.3f}$ with it",
         "empirical"),
        ("5", r"MP-utility",
         r"$3$ loss distributions ($\beta$, bimodal, uniform)",
         rf"range-only Hoeffding ${min(mp_ratios):.2f}$--${max(mp_ratios):.2f}{{\times}}$ wider than MP",
         "empirical"),
        ("6", r"Chernoff variance bridge",
         r"required for $\hat\sigma^2 {\le} 2B^2 p_{\mathrm{acc}}$, synthetic",
         rf"\textbf{{{s['NO_CHERNOFF_VARBRIDGE']['n_feasible']}/10}} feas without the variance bridge",
         "empirical"),
        ("7", r"Ratio reformulation $Z{=}A(L{-}\alpha)$",
         r"denominator handling, synthetic",
         rf"\textbf{{{s['NO_RATIO']['n_feasible']}/10}} feas without reformulation (confounded$^\ast$)",
         "empirical (confounded)"),
        ("8", r"Two-sided MP",
         r"Lemma~\ref{lem:inclusion} inclusion direction",
         rf"one-sided MP \textbf{{{s['NO_TWO_SIDED_MP']['n_feasible']}/10}} feas at "
         rf"${two_sided_fold:.3f}{{\times}}$ our median width (empirically tighter)",
         r"\textbf{proof-only}"),
    ]
    tex_rows = "\n".join(
        f"        {num} & {ing} & {regime} & {result} & {kind} \\\\"
        for num, ing, regime, result, kind in rows
    )

    tex = rf"""\begin{{table*}}[t]
    \centering
    \caption{{Full $8$-ingredient stress matrix.
    Each row uses a regime designed to expose its ingredient's failure mode, not vanilla
    benign settings. Of the $8$ ingredients, $7$ produce a named empirical degradation
    (rows~1--7); for the $8$th (two-sided MP, row~8) the one-sided variant is
    empirically tighter (${two_sided_fold:.3f}\times$ the median width), and the two-sided form
    is required by Lemma~\ref{{lem:inclusion}}'s inclusion-direction
    analysis, a proof-only necessity, not an empirical performance claim.
    $^\ast$Row~7 is confounded: removing the ratio reformulation also forces removing the CP-LCB,
    so its $0/10$ infeasibility cannot be cleanly attributed to ratio removal alone.}}
    \label{{tab:ingredient}}
    \footnotesize
    \setlength{{\tabcolsep}}{{3pt}}
    \begin{{tabular}}{{c p{{0.20\linewidth}} p{{0.22\linewidth}} p{{0.33\linewidth}} l}}
        \toprule
        \# & Ingredient & Failure-mode regime & Result & Type \\
        \midrule
{tex_rows}
        \bottomrule
    \end{{tabular}}
\end{{table*}}
"""
    write_tex("table_ingredients", tex)


if __name__ == "__main__":
    main()
