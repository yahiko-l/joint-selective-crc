#!/usr/bin/env python
"""Infeasibility census across the evaluation surfaces of Table 4 (tab:expconfig).

Recomputes, from the released per-seed artifacts, how often Algorithm 1 returned
INFEASIBLE on every evaluation surface, together with the two diagnostics cited
alongside the census (the Fig. 8(b) pi_min floor sweep and the A8
calibration-budget sweep). Nothing is re-run: this is bookkeeping over the
committed artifacts, and every count is asserted against the totals quoted in
the manuscript's Sec. 5.4 census display.

Output: results/analysis/infeasibility_census.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUPP = ROOT / "results" / "ablation_supplement"
OUT = ROOT / "results" / "analysis" / "infeasibility_census.json"


def load(path):
    with open(path) as f:
        return json.load(f)


def frac(feasible, runs):
    return {"runs": runs, "feasible": feasible, "infeasible": runs - feasible}


def main():
    surfaces = []

    # --- ImageNet val, frontier re-splits (F_joint artifact; ours rows) ---
    fj = load(SUPP / "F_joint_hoeffding_baseline.json")
    for bk, v in fj["imagenet"].items():
        feas, runs = (int(x) for x in v["ours"]["feasible"].split("/"))
        surfaces.append({"surface": f"ImageNet val ({bk}; frontier re-splits)",
                         "source": "F_joint_hoeffding_baseline.json",
                         **frac(feas, runs)})
        assert (feas, runs) == (20, 20), (bk, feas, runs)

    # --- ImageNet val RN50 V2, held-out validity re-splits ---
    prim = load(ROOT / "results" / "imagenet_primary_real" / "results.json")

    def find_feasibility(node):
        if isinstance(node, dict):
            if "n_feasible" in node and "n_seeds" in node:
                return int(node["n_feasible"]), int(node["n_seeds"])
            for v in node.values():
                got = find_feasibility(v)
                if got:
                    return got
        return None

    feas, runs = find_feasibility(prim)
    surfaces.append({"surface": "ImageNet val (RN50 V2; validity re-splits)",
                     "source": "imagenet_primary_real/results.json",
                     **frac(feas, runs)})
    assert (feas, runs) == (30, 30)

    # --- COCO val 2017, softmax and entropy scores (robust artifacts) ---
    for tag, label, expect in [
        ("G_coco_pixacc_g_softmax_a0.10_pi0.10_robust", "COCO val 2017 (softmax g)", (29, 30)),
        ("G_coco_pixacc_g_entropy_a0.10_pi0.10_robust", "COCO val 2017 (entropy g)", (12, 20)),
    ]:
        per = load(SUPP / f"{tag}.json")["per_seed"]
        feas = sum(1 for s in per if (s.get("p_ours") or 0) > 0)
        infeasible_seeds = [s["seed"] for s in per if (s.get("p_ours") or 0) == 0]
        surfaces.append({"surface": label, "source": f"{tag}.json",
                         "infeasible_seeds": infeasible_seeds,
                         **frac(feas, len(per))})
        assert (feas, len(per)) == expect, (tag, feas, len(per))

    # --- CIFAR-100 pi_min sweep ---
    cif = load(SUPP / "A1_cifar100_resnet56_pi_min_sweep.json")["sweep"]
    feas = sum(int(r["n_feasible"]) for r in cif)
    runs = sum(int(r["n_seeds"]) for r in cif)
    surfaces.append({"surface": "CIFAR-100 (RN56; four floors x 10 seeds)",
                     "source": "A1_cifar100_resnet56_pi_min_sweep.json",
                     **frac(feas, runs)})
    assert (feas, runs) == (40, 40)

    # --- ADE20K, both backbones (softmax score, the manuscript's variant) ---
    for tag, label in [("G_ade20k_mask2former_g_softmax", "ADE20K (Mask2Former-Swin-B)"),
                       ("G_ade20k_segformer_g_softmax", "ADE20K (SegFormer-MiT-B2)")]:
        val = load(SUPP / f"{tag}.json")["validity"]
        surfaces.append({"surface": label, "source": f"{tag}.json",
                         **frac(int(val["n_seeds_with_decision"]), int(val["n_seeds"]))})
        assert val["n_seeds_with_decision"] == val["n_seeds"] == 20

    # --- ImageNet-V2 block ---
    b11 = load(SUPP / "B11_distribution_shift.json")
    n_seeds = int(b11["config"]["n_seeds"])
    for key, label in [("iid_summary", "ImageNet-V2 block (i.i.d. test)"),
                       ("shift_summary", "ImageNet-V2 block (shift test)")]:
        surfaces.append({"surface": label, "source": "B11_distribution_shift.json",
                         **frac(int(b11[key]["n_feasible"]), n_seeds)})
        assert b11[key]["n_feasible"] == n_seeds == 10

    # --- Synthetic: 27-config calibration sanity + F.1 stress sweep, merged ---
    sanity = load(ROOT / "results" / "synthetic_full" / "pc1_summary.json")["configs"]
    f1 = load(SUPP / "F1_b1_pc1_30seeds.json")["rows"]
    merged = {}
    for r in sanity:
        key = (r["alpha"], r["n_cert"])
        cell = merged.setdefault(key, {"runs": 0, "feasible": 0})
        cell["runs"] += int(r["n_runs"])
        cell["feasible"] += int(r["n_feasible"])
        assert r["alpha"] == 0.3
    for r in f1:
        c = r["config"]
        key = (c["alpha"], c["n_cert"])
        cell = merged.setdefault(key, {"runs": 0, "feasible": 0})
        cell["runs"] += int(r["n_seeds"])
        cell["feasible"] += int(r["n_feasible"])
    alpha010 = {"runs": 0, "feasible": 0}
    synthetic_rows = []
    for (alpha, n_cert), cell in sorted(merged.items()):
        if alpha == 0.1:
            alpha010["runs"] += cell["runs"]
            alpha010["feasible"] += cell["feasible"]
        else:
            synthetic_rows.append({"surface": f"Synthetic, alpha=0.30, n_cert={n_cert}",
                                   "source": "pc1_summary.json + F1_b1_pc1_30seeds.json",
                                   **frac(cell["feasible"], cell["runs"])})
    synthetic_rows.insert(0, {"surface": "Synthetic, alpha=0.10 (budget below E[L]=2/7)",
                              "source": "F1_b1_pc1_30seeds.json",
                              **frac(alpha010["feasible"], alpha010["runs"])})
    surfaces.extend(synthetic_rows)
    by = {r["surface"]: r for r in synthetic_rows}
    assert by["Synthetic, alpha=0.10 (budget below E[L]=2/7)"]["infeasible"] == 240
    assert by["Synthetic, alpha=0.30, n_cert=1000"]["infeasible"] == 180
    assert by["Synthetic, alpha=0.30, n_cert=5000"]["infeasible"] == 213
    assert by["Synthetic, alpha=0.30, n_cert=25000"]["infeasible"] == 0
    syn_runs = sum(r["runs"] for r in synthetic_rows)
    syn_feas = sum(r["feasible"] for r in synthetic_rows)
    assert (syn_runs, syn_feas) == (1020, 387)

    # --- Diagnostics cited alongside the census ---
    diagnostics = {}
    floors = {}
    for tag in ["A1_imagenet_resnet50v2_pi_min_sweep", "A1_imagenet_resnet101v2_pi_min_sweep"]:
        rows = load(SUPP / f"{tag}.json")["sweep"]
        floors[tag] = {str(r["pi_min"]): f"{r['n_feasible']}/{r['n_seeds']}" for r in rows}
        assert all(r["n_feasible"] == r["n_seeds"] == 10 for r in rows)
    diagnostics["fig8b_floor_sweeps"] = floors
    a8 = load(SUPP / "A8_imagenet_resnet50v2_n_cert_sweep.json")["sweep"]
    diagnostics["a8_n_cert_sweep"] = {str(r["n_cert"]): f"{r['n_feasible']}/{r['n_seeds']}" for r in a8}
    assert diagnostics["a8_n_cert_sweep"]["2000"] == "0/10"
    for n in ("5000", "10000", "15000", "25000"):
        assert diagnostics["a8_n_cert_sweep"][n] == "10/10"

    out = {
        "purpose": "Infeasibility census over the evaluation surfaces of Table 4 "
                   "(manuscript Sec. 5.4), recomputed from released per-seed artifacts.",
        "generated_by": "experiments/analysis_infeasibility_census.py",
        "surfaces": surfaces,
        "diagnostics": diagnostics,
        "totals": {
            "synthetic": {"runs": syn_runs, "feasible": syn_feas, "infeasible": syn_runs - syn_feas},
            "real_data_infeasible": {"COCO softmax": "1/30", "COCO entropy": "8/20",
                                     "all other real surfaces": "0"},
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")
    for r in surfaces:
        print(f"  {r['surface']:55s} INFEASIBLE {r['infeasible']}/{r['runs']}")


if __name__ == "__main__":
    main()
