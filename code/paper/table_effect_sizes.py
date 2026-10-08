"""Render the re-split effect-size table for the headline comparisons (tab:b14).

Input: results/ablation_supplement/H_resplit_effectsize.json, written by
experiments/analysis_resplit_effectsize.py.
"""

from __future__ import annotations

from style import load_json, write_tex

CAPTION = (
    "Re-split effect sizes for the headline comparisons. Each row summarises a per-re-split "
    "statistic over $N$ independently drawn random calibration/test re-splits of the fixed "
    "evaluation pool: the median across re-splits, the Hodges--Lehmann (HL) "
    "pseudo-median~\\cite{hodges1963_location} (the median of Walsh averages), a "
    "distribution-free order-statistic confidence interval for the median of the "
    "split-randomization distribution conditional on the pool (per-comparison, finite-sample "
    "coverage at least $95\\%$), and the sign-consistency count. ImageNet rows summarise the "
    "within-re-split median (over the $35$ grid pairs) of the sign-aware valid UCB-excess ratio "
    "Ours$/A(\\mathrm{CP}^{\\pm})$ of \\cref{sec:cert-decision}; values below $1$ favour ours. "
    "COCO rows summarise the per-re-split certified-acceptance gap over the Hoeffding--CRC "
    "selective baseline of \\cref{sec:segmentation}, in percentage points, under the "
    "operational convention that a re-split on which the certifier abstains contributes a zero "
    "gap: an evaluation convention, not a certified value. No re-split $p$-values are attached "
    "(see the accompanying text)."
)


def f4(x: float) -> str:
    return f"{x:.4f}"


def pp1(x: float) -> str:
    if x == 0.0:
        return "0.0"
    return f"{'+' if x > 0 else ''}{x:.1f}"


def ratio_row(label: str, s: dict) -> tuple:
    return (label, s["n"], f4(s["median"]), f4(s["hodges_lehmann"]),
            f"$[{f4(s['ci95']['lo'])},\\, {f4(s['ci95']['hi'])}]$",
            f"${s['sign_consistent']}/{s['n']} < 1$")


def main() -> None:
    d = load_json("ablation_supplement/H_resplit_effectsize.json")
    im = d["imagenet_signaware"]

    rows = [
        ratio_row(f"Excess ratio Ours$/A(\\mathrm{{CP}}^{{\\pm}})$, {short}, all pairs",
                  im[name]["ratio_all_pairs"])
        for name, short in (("ResNet-50 V2", "RN50 V2"), ("ResNet-101 V2", "RN101 V2"),
                            ("ResNet-152 V2", "RN152 V2"))
    ]
    rows.append(ratio_row("Excess ratio Ours$/A(\\mathrm{CP}^{\\pm})$, RN50 V2, low-acceptance",
                          im["ResNet-50 V2"]["ratio_low_acceptance"]))
    for score in ("softmax", "entropy"):
        s = d["coco_gap"][score]["primary_all_splits_operational"]
        rows.append((
            f"Certified-acceptance gap (pp), COCO, $g{{=}}${score}",
            s["n"], pp1(s["median"]), pp1(s["hodges_lehmann"]),
            f"$[{pp1(s['ci95']['lo'])},\\, {pp1(s['ci95']['hi'])}]$",
            f"${s['sign_consistent']}/{s['n']} > 0$",
        ))

    body = "\n".join(
        f"        {c} & ${n}$ & ${med}$ & ${hl}$ & {ci} & {sg} \\\\"
        for c, n, med, hl, ci, sg in rows
    )
    tex = f"""\\begin{{table}}[t]
    \\centering
    \\caption{{{CAPTION}}}
    \\label{{tab:b14}}
    \\footnotesize
    \\setlength{{\\tabcolsep}}{{4pt}}
    {{%
    \\begin{{tabular}}{{lccccc}}
        \\toprule
        Comparison & $N$ & median & HL & $95\\%$ re-split CI & sign \\\\
        \\midrule
{body}
        \\bottomrule
    \\end{{tabular}}%
    }}
\\end{{table}}
"""
    write_tex("table_effect_sizes", tex)


if __name__ == "__main__":
    main()
