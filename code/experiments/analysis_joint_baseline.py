"""Matched joint Hoeffding--CRC ablation, "joint-H" (revision).

Construction: Algorithm 1 with exactly ONE substitution. The line-5
empirical-Bernstein UCB on E[Z] is replaced by the one-sided range-Hoeffding
UCB at the same U1 allocation delta/(16m),

    EB_H := z_bar + r_H,    r_H = B * sqrt(log(16 m / delta) / (2 n_cert)),

a choice deliberately favourable to the baseline (one-sided, since only the
upper tail is needed for the returned-pair validity tier). The Clopper-Pearson
acceptance leg (delta/(16m)), the Maurer-Pontil utility leg (delta/(2m)), the
membership rule (numerator UCB <= 0 AND p_LCB >= pi_min), the argmax-U_LCB
selection rule, the grid, and the total failure budget are byte-identical to
`certify_grid`. Both certifiers therefore use the same nominal failure budget
delta and the same basic-tier allocations, with union-bound failure at most
5*delta/8; the basic-tier proof is unchanged except that the U1 event family
is swapped. No inclusion or external-oracle guarantee is asserted for joint-H,
regardless of the surface's (star) status.

Surfaces:
  - ImageNet RN50/101/152 V2, D protocol (n_cert=33000, 20 seeds 42..61, m=35,
    alpha=0.05, pi_min=0.01, delta=0.05, c=0.1, V=1).
  - COCO val 2017 panoptic, canonical re-grid path (30 splits 42..71, m=15,
    alpha=pi_min=delta=0.10, c=0, V=1); the ours-side values reproduce the
    published headline (29/30 feasible, U_LCB = 0.199 at the returned pair).

Reported per surface and method: feasibility, certified-set size, maximum
certified acceptance (Table 7 convention), returned-pair empirical acceptance
p_hat(lambda_hat, tau_hat), returned-pair U_LCB; returned-pair statistics are
aggregated over feasible runs only (never zero-imputed). The returned-pair
p_LCB is retained in this artifact for audit. Per-seed records, the
Ghat_H-subset-of-Ghat assertion, and the same-returned-pair indicator are
stored for every run.

Output (overwrite): results/ablation_supplement/F_joint_hoeffding_baseline.json

Run:
    python -m experiments.analysis_joint_baseline
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from selective_crc import certify_grid, three_split_indices  # noqa: E402
from experiments import cifar100  # noqa: E402

import os
# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

OUT = ROOT / "results" / "ablation_supplement" / "F_joint_hoeffding_baseline.json"

IMAGENET = dict(alpha=0.05, pi_min=0.01, delta=0.05, B=1.0, c=0.1, V=1.0,
                n_cert=33000, n_tune=8500, n_test=8500, n_classes=1000,
                seeds=[42 + i for i in range(20)])
LAMBDA = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
T = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
MODELS = {
    "ResNet-50 V2": f"{DATA_ROOT}/imagenet_data/val_logits.npy",
    "ResNet-101 V2": f"{DATA_ROOT}/imagenet_data/val_logits_resnet101.npy",
    "ResNet-152 V2": f"{DATA_ROOT}/imagenet_data/val_logits_resnet152.npy",
}
LABELS = f"{DATA_ROOT}/imagenet_data/val_labels.npy"
COCO = dict(alpha=0.10, pi_min=0.10, delta=0.10, B=1.0, c=0.0, V=1.0,
            m=15, n_cal=4000, seeds=list(range(42, 72)),
            dir=f"{DATA_ROOT}/coco_data")


def one_run(L_grid, A_grid, v_grid, *, alpha, pi_min, delta, c, V, B):
    """Ours (certify_grid) vs joint-H on one calibration draw."""
    n, m = L_grid.shape
    res = certify_grid(L_grid, A_grid, v_grid, alpha=alpha, pi_min=pi_min,
                       delta=delta, c=c, V=V, B=B, check_sample_size=False)
    p_hat = res.s_count / n
    r_H = B * np.sqrt(np.log(16.0 * m / delta) / (2.0 * n))   # one-sided UCB
    eb_H = res.z_bar + r_H
    inG_H = (eb_H <= 0.0) & (res.p_lcb_per_pair >= pi_min)
    inG = res.in_g_hat_mask
    assert bool(np.all(inG_H <= inG)), "Ghat_H must be a subset of Ghat"

    def summarize(mask):
        if not mask.any():
            return dict(feasible=False, n_certified=0, max_cert_p_hat=0.0,
                        sel=None, p_hat_sel=None, p_lcb_sel=None, u_lcb_sel=None)
        k = int(np.flatnonzero(mask)[np.argmax(res.u_lcb_per_pair[mask])])
        return dict(feasible=True, n_certified=int(mask.sum()),
                    max_cert_p_hat=float(p_hat[mask].max()), sel=k,
                    p_hat_sel=float(p_hat[k]),
                    p_lcb_sel=float(res.p_lcb_per_pair[k]),
                    u_lcb_sel=float(res.u_lcb_per_pair[k]))

    ours, jh = summarize(inG), summarize(inG_H)
    both_feasible = ours["feasible"] and jh["feasible"]
    return dict(ours=ours, joint_h=jh, r_H=float(r_H),
                # None unless both certifiers returned a pair (a run where both
                # are infeasible must not count as an agreement).
                same_returned_pair=(ours["sel"] == jh["sel"]) if both_feasible else None,
                ghat_strict_superset=bool(inG.sum() > inG_H.sum()))


def aggregate(rows):
    """Per-method summary; returned-pair stats over feasible runs only."""
    out = {}
    for side in ("ours", "joint_h"):
        feas = [r[side] for r in rows if r[side]["feasible"]]
        med = lambda key: (round(float(np.median([f[key] for f in feas])), 4)  # noqa: E731
                           if feas else None)
        out[side] = {
            "feasible": f"{len(feas)}/{len(rows)}",
            "median_n_certified": med("n_certified"),
            "median_max_cert_p_hat": med("max_cert_p_hat"),
            "median_p_hat_at_returned_pair": med("p_hat_sel"),
            "median_p_lcb_at_returned_pair": med("p_lcb_sel"),
            "median_u_lcb_at_returned_pair": med("u_lcb_sel"),
        }
    both = [r for r in rows if r["ours"]["feasible"] and r["joint_h"]["feasible"]]
    out["same_returned_pair_rate_when_both_feasible"] = (
        round(sum(r["same_returned_pair"] for r in both) / len(both), 4)
        if both else None)
    out["ghatH_subset_of_ghat_all_runs"] = True   # asserted per run
    strict = sum(r["ghat_strict_superset"] for r in rows)
    out["ghatH_strict_subset_runs"] = f"{strict}/{len(rows)}"
    return out


def imagenet_block():
    cfg = IMAGENET
    labels_full = np.load(LABELS).astype(np.int64)
    out = {}
    for name, path in MODELS.items():
        logits_full = np.load(path).astype(np.float64)
        rows = []
        t0 = time.time()
        for seed in cfg["seeds"]:
            rng = np.random.default_rng(seed)
            subset = rng.choice(len(labels_full),
                                size=cfg["n_cert"] + cfg["n_tune"] + cfg["n_test"],
                                replace=False)
            Y_all, logit_all = labels_full[subset], logits_full[subset]
            _, cert_idx, _ = three_split_indices(
                len(Y_all), n_tune=cfg["n_tune"], n_cert=cfg["n_cert"], seed=seed)
            Yc, Lc = Y_all[cert_idx], logit_all[cert_idx]
            contains, size = cifar100._compute_contains_and_size(
                Lc, Yc, LAMBDA, cfg["n_classes"])
            L_l = (~contains).astype(np.float64)
            L_grid = np.repeat(L_l, len(T), axis=1)
            A_lt = cifar100.construct_acceptance(Lc, T)
            A_grid = np.tile(A_lt, (1, len(LAMBDA)))
            v_l = contains.astype(np.float64) / np.maximum(size, 1)
            v_grid = np.repeat(v_l, len(T), axis=1)
            rows.append(one_run(L_grid, A_grid, v_grid, alpha=cfg["alpha"],
                                pi_min=cfg["pi_min"], delta=cfg["delta"],
                                c=cfg["c"], V=cfg["V"], B=cfg["B"]))
        agg = aggregate(rows)
        agg["per_seed"] = [
            {"seed": s, "ours": r["ours"], "joint_h": r["joint_h"],
             "same_returned_pair": r["same_returned_pair"]}
            for s, r in zip(cfg["seeds"], rows)]
        agg["seconds"] = round(time.time() - t0, 1)
        out[name] = agg
        print(name, json.dumps({k: v for k, v in agg.items() if k != "per_seed"}),
              flush=True)
        del logits_full
    return out


def coco_block():
    cfg = COCO
    d = Path(cfg["dir"])
    L_raw = np.clip(np.load(d / "val_mask2former_coco_loss_pixacc.npy").astype(np.float64), 0, 1)
    g = np.load(d / "val_mask2former_coco_g_softmax.npy").astype(np.float64)
    rows = []
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        cal = rng.permutation(len(g))[:cfg["n_cal"]]
        g_cal, L_cal = g[cal], L_raw[cal]
        tau = np.quantile(g_cal, np.linspace(0.5, 0.95, cfg["m"]))
        A = (g_cal.reshape(-1, 1) > tau.reshape(1, cfg["m"])).astype(np.float64)
        L = np.broadcast_to(L_cal.reshape(-1, 1), (cfg["n_cal"], cfg["m"])).astype(np.float64)
        v = np.clip(1.0 - L, 0.0, 1.0)
        rows.append(one_run(L, A, v, alpha=cfg["alpha"], pi_min=cfg["pi_min"],
                            delta=cfg["delta"], c=cfg["c"], V=cfg["V"], B=cfg["B"]))
    agg = aggregate(rows)
    agg["per_seed"] = [
        {"seed": s, "ours": r["ours"], "joint_h": r["joint_h"],
         "same_returned_pair": r["same_returned_pair"]}
        for s, r in zip(cfg["seeds"], rows)]
    print("COCO", json.dumps({k: v for k, v in agg.items() if k != "per_seed"}),
          flush=True)
    return agg


def main():
    out = {
        "experiment": "analysis_joint_baseline",
        "purpose": ("Matched joint Hoeffding--CRC ablation (joint-H) vs "
                    "Algorithm 1: same CP acceptance leg, MP utility leg, "
                    "membership and argmax-U_LCB selection rules, and "
                    "basic-tier failure allocations; only the U1 numerator "
                    "bound family is swapped."),
        "definition": {
            "numerator": "one-sided range-Hoeffding UCB at the U1 allocation: "
                         "r_H = B*sqrt(log(16m/delta)/(2 n_cert)); deliberately "
                         "favourable to the baseline",
            "budget": "same nominal failure budget delta; basic-tier "
                      "allocations delta/(16m) numerator family, delta/(16m) "
                      "Clopper-Pearson acceptance, delta/(2m) utility; "
                      "union-bound failure at most 5*delta/8 for both",
            "claim_scope": "basic returned-pair validity tier only; no "
                           "inclusion or external-oracle guarantee is asserted "
                           "for joint-H, regardless of (star) status",
            "aggregation": "returned-pair statistics over feasible runs only; "
                           "infeasible runs are never zero-imputed",
        },
        "config": {"imagenet": {k: v for k, v in IMAGENET.items()},
                   "imagenet_grid": {"Lambda": LAMBDA.tolist(), "T": T.tolist()},
                   "coco": {k: v for k, v in COCO.items() if k != "dir"}},
        "imagenet": imagenet_block(),
        "coco": coco_block(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
