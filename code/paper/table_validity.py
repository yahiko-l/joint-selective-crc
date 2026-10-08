"""Render the held-out validity table across four evaluation surfaces (tab:validity).

Inputs: results/synthetic_full/pc1_summary.json and results/ablation_supplement/
F1_b1_pc1_30seeds.json, B12_label_noise_robustness.json, B11_distribution_shift.json.
"""

from __future__ import annotations

from scipy.stats import beta

from style import load_json, write_tex

IMAGENET_GRID_SIZE = 35


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}") if n >= 10_000 else str(n)


def fmt_set(values) -> str:
    return r"$\{" + ", ".join(fmt_int(v) for v in sorted(values)) + r"\}$"


def cp_upper(n_vio: int, n_feas: int) -> str:
    """One-sided 95% Clopper-Pearson upper bound on the violation rate."""
    return f"{beta.ppf(0.95, n_vio + 1, n_feas - n_vio):.4f}"


def main() -> None:
    sweep = load_json("synthetic_full/pc1_summary.json")["configs"]
    stress = load_json("ablation_supplement/F1_b1_pc1_30seeds.json")
    noise = load_json("ablation_supplement/B12_label_noise_robustness.json")
    shift = load_json("ablation_supplement/B11_distribution_shift.json")

    sweep_feas = sum(c["n_feasible"] for c in sweep)
    sweep_runs = sum(c["n_runs"] for c in sweep)
    sweep_vio = int(round(sum(c["vio_joint_rate"] * c["n_feasible"]
                              for c in sweep if c["vio_joint_rate"] is not None)))

    stress_cfg = [r["config"] for r in stress["rows"]]
    noise_feas = sum(r["n_feasible"] for r in noise["rows"])
    noise_vio = sum(r["n_vio_joint"] for r in noise["rows"])
    shift_sum = shift["shift_summary"]
    top1_drop = (shift["val_top1"] - shift["v2_top1"]) * 100

    rows = [
        (r"Synthetic Beta$(2,5)$, $27$ configs $\times$ $20$ seeds",
         fmt_set({c["n_cert"] for c in sweep}), fmt_set({c["m"] for c in sweep}),
         f"{sweep_feas} / {sweep_runs}", sweep_vio, cp_upper(sweep_vio, sweep_feas)),
        (r"Synthetic Beta$(2,5)$ stress sweep, $16$ configs $\times$ $30$ seeds",
         fmt_set({c["n_cert"] for c in stress_cfg}), fmt_set({c["m"] for c in stress_cfg}),
         str(stress["total_feasible"]), stress["total_vio_joint"],
         cp_upper(stress["total_vio_joint"], stress["total_feasible"])),
        (r"ImageNet RN50 V2, symmetric label-flip $\{0,5,10,20\}\%$",
         f"${fmt_int(noise['config']['n_cert'])}$", f"${IMAGENET_GRID_SIZE}$",
         f"{noise_feas} ({len(noise['rows'])} rates)", noise_vio, cp_upper(noise_vio, noise_feas)),
        (rf"ImageNet$\to$ImageNet-V2 shift, $\Delta$top-1 $={top1_drop:.1f}$ pp",
         f"${fmt_int(shift['config']['n_cert'])}$", f"${IMAGENET_GRID_SIZE}$",
         str(shift_sum["n_feasible"]), shift_sum["n_vio_joint"],
         cp_upper(shift_sum["n_vio_joint"], shift_sum["n_feasible"]) + r" $^\dagger$"),
    ]
    tex_rows = "\n".join(
        f"        {s} & {n} & {m} & {feas} & {vio} & {ub} \\\\" for s, n, m, feas, vio, ub in rows
    )

    tex = rf"""\begin{{table*}}[t]
    \centering
    \caption{{Joint-certificate held-out validity across four evaluation surfaces.
    ``\#feas'' is the number of feasible runs (Algorithm~1 did not return INFEASIBLE);
    ``\#vio'' counts joint violations ($R_{{\mathrm{{test}}}} > \alpha$ or $p_{{\mathrm{{test}}}} < \pi_{{\min}}$);
    ``CP 95\% UB'' is the Clopper--Pearson 95\% one-sided upper bound on the violation rate
    (avoids the invalid $[0/n,0/n]$ artefact at zero observed events).
    The $n_{{\mathrm{{cert}}}}$ and $m$ columns give the certification sample sizes and grid sizes represented in each row; the remaining certification parameters, and the configuration grids of the two synthetic sweeps, are in \cref{{tab:expconfig}} (table note d).
    $^\dagger$This row is reported descriptively: the sample-size condition (\ref{{eq:sample-size}}) is not met at the configured $(\pi_{{\min}}, m)$;
    we make no theorem-backed shift-robustness claim.}}
    \label{{tab:validity}}
    \small
    \begin{{tabular}}{{lccrrl}}
        \toprule
        Surface & $n_{{\mathrm{{cert}}}}$ & $m$ & \#feas & \#vio & CP 95\% UB \\
        \midrule
{tex_rows}
        \bottomrule
    \end{{tabular}}
\end{{table*}}
"""
    write_tex("table_validity", tex)


if __name__ == "__main__":
    main()
