"""Render the certified-decision payoff table on three ImageNet backbones (tab:cert-decision).

Input: results/analysis/certified_decision_payoff.json (per-seed records; medians are taken here).
"""

from __future__ import annotations

import statistics

from style import load_json, write_tex


def seed_median(per_seed: list, field: str, method: str):
    vals = [s[field][method] for s in per_seed if s[field][method] is not None]
    return statistics.median(vals) if vals else None


def main() -> None:
    d = load_json("analysis/certified_decision_payoff.json")
    cfg = d["config"]
    grid_size = len(cfg["Lambda"]) * len(cfg["T"])

    body_rows = []
    for name, payload in d["models"].items():
        per_seed = payload["per_seed"]
        ours_p = seed_median(per_seed, "p_acc_cert", "ours")
        apm_p = seed_median(per_seed, "p_acc_cert", "a_pi_min")
        aplcb_p = seed_median(per_seed, "p_acc_cert", "a_plcb")
        ours_n = seed_median(per_seed, "n_certified", "ours")
        aplcb_n = seed_median(per_seed, "n_certified", "a_plcb")
        body_rows.append(
            f"        {name} & {payload['top1']:.4f} & {ours_p:.3f} & {apm_p:.3f} & "
            f"\\textbf{{${(ours_p - apm_p) * 100:+.1f}$}} & {aplcb_p:.3f} & ${(ours_p - aplcb_p) * 100:+.2f}$ & "
            f"{int(ours_n)} vs {int(aplcb_n)} (${ours_n / aplcb_n:.1f}\\times$) \\\\"
        )
    tex_rows = "\n".join(body_rows)

    tex = rf"""\begin{{table*}}[t]
    \centering
    \caption{{Certified-decision payoff on three ImageNet backbones at
    $\alpha=${cfg['alpha']}, $\pi_{{\min}}=${cfg['pi_min']}, $\delta=${cfg['delta']},
    $n_{{\mathrm{{cert}}}}=${cfg['n_cert']}, $m=${grid_size}, {cfg['n_seeds']} seeds.
    $p_{{\mathrm{{acc}}}}^{{\mathrm{{cert}}}}$: maximum acceptance the method can certify on $D_{{\mathrm{{cert}}}}$
    (median across seeds).
    $\Delta$ vs $A(\pi_{{\min}})$: textbook $\pi_{{\min}}$-saturated Hoeffding--CRC selective bound certifies nothing
    in this regime; column 5 reports the operational acceptance window Ours opens.
    $\Delta$ vs $A(p_{{\mathrm{{LCB}}}})$: the empirical Clopper--Pearson denominator variant
    coincides with Ours at the saturated maximum but is dominated by Ours at every low-acceptance operating point
    (last column counts grid pairs each method certifies; Fig.~\ref{{fig:variance-adaptive}} shows the per-pair width gap).}}
    \label{{tab:cert-decision}}
    \small
    \begin{{tabular}}{{lccccccc}}
        \toprule
        \multirow{{2}}{{*}}{{Model}} & \multirow{{2}}{{*}}{{top-1}} &
        Ours & $A(\pi_{{\min}})$ & $\Delta$ vs & $A(p_{{\mathrm{{LCB}}}})$ & $\Delta$ vs & \#cert pairs \\
        & & $p_{{\mathrm{{acc}}}}^{{\mathrm{{cert}}}}$ & $p_{{\mathrm{{acc}}}}^{{\mathrm{{cert}}}}$ & $A(\pi_{{\min}})$ [pp] &
        $p_{{\mathrm{{acc}}}}^{{\mathrm{{cert}}}}$ & $A(p_{{\mathrm{{LCB}}}})$ [pp] & Ours vs $A(p_{{\mathrm{{LCB}}}})$ \\
        \midrule
{tex_rows}
        \bottomrule
    \end{{tabular}}
\end{{table*}}
"""
    write_tex("table_cert_decision", tex)


if __name__ == "__main__":
    main()
