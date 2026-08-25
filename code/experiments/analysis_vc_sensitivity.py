"""Utility-parameter (v, c) sensitivity analysis (revision).

Blocks (all on the frozen headline protocols; loaders and splits are
byte-identical to analysis_joint_baseline.py):

  A. Well-specified c-sweep: certify AND evaluate at the same c in
     C_SWEEP = {0, 0.05, 0.1, 0.2, 0.5, 1.0}. Reported per c: returned-pair
     drift rate vs the surface default, median returned-pair U_LCB, median
     2*gamma_u (gamma_u = grid max of the MP radius eta_U = u_bar - U_LCB),
     median realized test utility at the same c, vacuity margin
     U_LCB - (-c) = U_LCB + c, and 2*gamma_u / (c + V).
  B. Value-family sweep: ImageNet {set-size-discounted (default), plain
     correctness 1{Y in C_lambda}} at c = 0.1; COCO {1 - L = pixel accuracy
     (default), mIoU} at c = 0. Units checks (artifact-level): joint
     rescaling (v, V, c) -> kappa*(v, V, c) with kappa = 0.5 must leave the
     argmax invariant and rescale U_LCB and gamma_u exactly (max float
     residual recorded); equivariance holds for any c.
  C. Cost-misspecification sweep at reference cost c_ref = 0.1 (on ImageNet
     c_ref equals the paper's operating specification; on COCO the paper's
     operating specification is c = 0, so c_ref = 0.1 is an instantiated
     counterfactual reference, needed for the ratio rho = c_cert/c_ref to be
     well-defined). For rho in {0, 0.25, 0.5, 1, 2, 4}: certify at
     c_cert = rho*c_ref with the default v, then evaluate the returned
     pair's utility at c_ref on the disjoint test split. Reported per rho:
     median raw U_LCB (a bound for the c_cert functional), median
     transferred lower bound at c_ref
     (U_LCB - max(c_ref - c_cert, 0) * (1 - p_LCB) at the returned pair;
     valid on the grid-uniform CP and utility events, no extra budget),
     median held-out utility gap (test utility at c_ref minus raw U_LCB),
     test-bound exceedance rate (descriptive; the rho = 1 row is the
     test-noise floor), median 2*gamma_u, drift vs rho = 1, and the max
     float residual of the per-pair linearity identity
     u_bar(c_ref) - u_bar(c_cert) = (c_cert - c_ref) * (1 - p_hat).

Structural invariance: Ghat and feasibility depend only on (L, A, alpha,
pi_min, delta), never on (v, c); asserted exactly (np.array_equal) for every
setting against the surface-default run.

Consistency anchors: the COCO default run must reproduce the published
returned-pair U_LCB = 0.199, and its 2*gamma_u must reconcile with the
published 0.062.

Output (overwrite): results/ablation_supplement/G_vc_sensitivity.json

Run:
    python -m experiments.analysis_vc_sensitivity
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

OUT = ROOT / "results" / "ablation_supplement" / "G_vc_sensitivity.json"

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

C_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
C_REF = 0.1
RHOS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
KAPPA = 0.5


def run_setting(L, A, v, *, alpha, pi_min, delta, c, V, B):
    """One certify_grid call plus the derived per-run quantities."""
    n, m = L.shape
    res = certify_grid(L, A, v, alpha=alpha, pi_min=pi_min, delta=delta,
                       c=c, V=V, B=B, check_sample_size=False)
    eta_u = res.u_bar - res.u_lcb_per_pair          # exact: no clipping anywhere
    two_gamma_u = 2.0 * float(eta_u.max())          # gamma_u = grid max of eta_U
    mask = res.in_g_hat_mask
    sel = None if not mask.any() else int(
        np.flatnonzero(mask)[np.argmax(res.u_lcb_per_pair[mask])])
    return dict(res=res, mask=mask, sel=sel, two_gamma_u=two_gamma_u,
                p_hat=res.s_count / n)


def test_utility(A_t, v_t, c, k):
    """Realized test utility of pair k at cost c."""
    a = A_t[:, k]
    return float((a * v_t[:, k]).mean() - c * (1.0 - a).mean())


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 4) if xs else None


def rate(num, den):
    return None if den == 0 else round(num / den, 4)


def surface_records(cal_data, test_data, cfg, v_families, default_family):
    """All blocks for one surface; cal_data/test_data map seed -> (L, A, {fam: v})."""
    base = dict(alpha=cfg["alpha"], pi_min=cfg["pi_min"], delta=cfg["delta"],
                B=cfg["B"])
    default_c, V = cfg["c"], cfg["V"]
    seeds = sorted(cal_data)
    per_seed = {s: {} for s in seeds}

    # --- per-seed runs, cached by (family, c, V, kappa-tag) --------------------
    for s in seeds:
        L, A, vmap = cal_data[s]
        cache = {}

        def run(fam, c, V_=V, scale=None):
            key = (fam, round(c, 6), round(V_, 6), scale)
            if key not in cache:
                v = vmap[fam] if scale is None else scale * vmap[fam]
                cache[key] = run_setting(L, A, v, c=c, V=V_, **base)
            return cache[key]

        ref = run(default_family, default_c)
        per_seed[s]["ref"] = ref
        # Ghat/feasibility invariance: every other setting must reproduce ref's mask
        per_seed[s]["runs"] = {}
        for c in sorted(set(C_SWEEP) | {rho * C_REF for rho in RHOS}):
            r = run(default_family, c)
            assert np.array_equal(r["mask"], ref["mask"]), "Ghat must be (v,c)-invariant"
            per_seed[s]["runs"][round(c, 6)] = r
        for fam in v_families:
            if fam == default_family:
                continue
            r = run(fam, default_c)
            assert np.array_equal(r["mask"], ref["mask"]), "Ghat must be (v,c)-invariant"
            per_seed[s][fam] = r
        # units check: joint rescaling (v, V, c) -> kappa*(v, V, c)
        rs = run(default_family, KAPPA * default_c, V_=KAPPA * V, scale=KAPPA)
        assert np.array_equal(rs["mask"], ref["mask"])
        per_seed[s]["units"] = dict(
            argmax_invariant=(rs["sel"] == ref["sel"]),
            max_residual_ulcb=float(np.abs(
                rs["res"].u_lcb_per_pair - KAPPA * ref["res"].u_lcb_per_pair).max()),
            max_residual_two_gamma=abs(rs["two_gamma_u"] - KAPPA * ref["two_gamma_u"]),
        )

    feas = [s for s in seeds if per_seed[s]["ref"]["sel"] is not None]
    infeas = [s for s in seeds if s not in feas]

    # --- Block A: well-specified c-sweep --------------------------------------
    block_a = {}
    for c in C_SWEEP:
        rows = [per_seed[s]["runs"][round(c, 6)] for s in feas]
        A_ts = [test_data[s] for s in feas]
        ulcb = [r["res"].u_lcb_per_pair[r["sel"]] for r in rows]
        block_a[f"c={c}"] = {
            "ghat_identical_runs": f"{len(seeds)}/{len(seeds)}",   # asserted above
            "drift_rate_vs_default": rate(
                sum(r["sel"] != per_seed[s]["ref"]["sel"] for s, r in zip(feas, rows)),
                len(feas)),
            "median_u_lcb_sel": med(ulcb),
            "median_two_gamma_u": med([r["two_gamma_u"] for r in rows]),
            "median_u_test_same_c": med([
                test_utility(At, vt[default_family], c, r["sel"])
                for (At, vt), r in zip(A_ts, rows)]),
            "median_vacuity_margin": med([u + c for u in ulcb]),
            "median_two_gamma_over_range": med(
                [r["two_gamma_u"] / (c + V) for r in rows]),
        }

    # --- Block B: value families + units --------------------------------------
    block_b = {}
    for fam in v_families:
        rows = [per_seed[s]["ref"] if fam == default_family else per_seed[s][fam]
                for s in feas]
        block_b[fam] = {
            "c": default_c,
            "drift_rate_vs_default": rate(
                sum(r["sel"] != per_seed[s]["ref"]["sel"] for s, r in zip(feas, rows)),
                len(feas)),
            "median_u_lcb_sel": med([r["res"].u_lcb_per_pair[r["sel"]] for r in rows]),
            "median_two_gamma_u": med([r["two_gamma_u"] for r in rows]),
        }
    units = [per_seed[s]["units"] for s in seeds]
    block_b["units_joint_scaling"] = {
        "kappa": KAPPA,
        "argmax_invariant": f"{sum(u['argmax_invariant'] for u in units)}/{len(units)}",
        "max_residual_ulcb": float(max(u["max_residual_ulcb"] for u in units)),
        "max_residual_two_gamma": float(max(u["max_residual_two_gamma"] for u in units)),
    }

    # --- Block C: cost misspecification at c_ref ------------------------------
    block_c = {}
    ref_key = round(C_REF, 6)
    for rho in RHOS:
        c_cert = round(rho * C_REF, 6)
        rows = [per_seed[s]["runs"][c_cert] for s in feas]
        rows_ref = [per_seed[s]["runs"][ref_key] for s in feas]
        raw, transferred, u_true, gap, exceed, resid = [], [], [], [], [], []
        for s, r, rr in zip(feas, rows, rows_ref):
            k = r["sel"]
            u_lcb = float(r["res"].u_lcb_per_pair[k])
            p_lcb = float(r["res"].p_lcb_per_pair[k])
            At, vt = test_data[s]
            ut = test_utility(At, vt[default_family], C_REF, k)
            raw.append(u_lcb)
            transferred.append(u_lcb - max(C_REF - c_cert, 0.0) * (1.0 - p_lcb))
            u_true.append(ut)
            gap.append(ut - u_lcb)
            exceed.append(u_lcb > ut)
            # per-pair linearity identity across the full grid (float residual)
            resid.append(float(np.abs(
                (rr["res"].u_bar - r["res"].u_bar)
                - (c_cert - C_REF) * (1.0 - r["p_hat"])).max()))
        block_c[f"rho={rho}"] = {
            "c_cert": c_cert,
            "drift_rate_vs_rho1": rate(
                sum(r["sel"] != rr["sel"] for r, rr in zip(rows, rows_ref)),
                len(feas)),
            "median_u_lcb_raw": med(raw),
            "median_transferred_lcb_at_c_ref": med(transferred),
            "median_u_test_at_c_ref": med(u_true),
            "median_heldout_gap": med(gap),
            "test_bound_exceedance_rate": rate(sum(exceed), len(feas)),
            "median_two_gamma_u": med([r["two_gamma_u"] for r in rows]),
            "median_two_gamma_over_range": med(
                [r["two_gamma_u"] / (c_cert + V) for r in rows]),
            "linearity_max_residual": float(max(resid)) if resid else None,
        }

    audit_rows = [
        {"seed": s,
         "feasible": per_seed[s]["ref"]["sel"] is not None,
         "sel_default": per_seed[s]["ref"]["sel"],
         "sel_by_c": {str(c): per_seed[s]["runs"][round(c, 6)]["sel"]
                      for c in sorted(set(C_SWEEP) | {r * C_REF for r in RHOS})},
         "sel_by_family": {f: per_seed[s][f]["sel"] for f in v_families
                           if f != default_family}}
        for s in seeds]
    return {"feasible": f"{len(feas)}/{len(seeds)}",
            "infeasible_seeds": infeas,
            "block_a_c_sweep": block_a,
            "block_b_v_families": block_b,
            "block_c_misspec": block_c,
            "per_seed": audit_rows}


def imagenet_block():
    cfg = IMAGENET
    labels_full = np.load(LABELS).astype(np.int64)
    out = {}
    for name, path in MODELS.items():
        logits_full = np.load(path).astype(np.float64)
        cal_data, test_data = {}, {}
        t0 = time.time()
        for seed in cfg["seeds"]:
            rng = np.random.default_rng(seed)
            subset = rng.choice(len(labels_full),
                                size=cfg["n_cert"] + cfg["n_tune"] + cfg["n_test"],
                                replace=False)
            Y_all, logit_all = labels_full[subset], logits_full[subset]
            _, cert_idx, test_idx = three_split_indices(
                len(Y_all), n_tune=cfg["n_tune"], n_cert=cfg["n_cert"], seed=seed)

            def build(idx):
                Yc, Lc = Y_all[idx], logit_all[idx]
                contains, size = cifar100._compute_contains_and_size(
                    Lc, Yc, LAMBDA, cfg["n_classes"])
                L_grid = np.repeat((~contains).astype(np.float64), len(T), axis=1)
                A_grid = np.tile(cifar100.construct_acceptance(Lc, T),
                                 (1, len(LAMBDA)))
                v_disc = np.repeat(
                    contains.astype(np.float64) / np.maximum(size, 1), len(T), axis=1)
                v_plain = np.repeat(contains.astype(np.float64), len(T), axis=1)
                return L_grid, A_grid, {"set_size_discounted": v_disc,
                                        "plain_correctness": v_plain}

            L_grid, A_grid, vmap = build(cert_idx)
            Lt, At, vtmap = build(test_idx)
            cal_data[seed] = (L_grid, A_grid, vmap)
            test_data[seed] = (At, vtmap)
        rec = surface_records(cal_data, test_data, cfg,
                              ["set_size_discounted", "plain_correctness"],
                              "set_size_discounted")
        rec["seconds"] = round(time.time() - t0, 1)
        out[name] = rec
        print(name, json.dumps({k: v for k, v in rec.items() if k != "per_seed"},
                               indent=None), flush=True)
        del logits_full
    return out


def coco_block():
    cfg = COCO
    d = Path(cfg["dir"])
    L_raw = np.clip(np.load(d / "val_mask2former_coco_loss_pixacc.npy").astype(np.float64), 0, 1)
    miou_raw = np.clip(np.load(d / "val_mask2former_coco_miou.npy").astype(np.float64), 0, 1)
    g = np.load(d / "val_mask2former_coco_g_softmax.npy").astype(np.float64)
    cal_data, test_data = {}, {}
    t0 = time.time()
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(g))
        cal, test = perm[:cfg["n_cal"]], perm[cfg["n_cal"]:]
        tau = np.quantile(g[cal], np.linspace(0.5, 0.95, cfg["m"]))

        def build(idx):
            A = (g[idx].reshape(-1, 1) > tau.reshape(1, cfg["m"])).astype(np.float64)
            L = np.broadcast_to(L_raw[idx].reshape(-1, 1),
                                (len(idx), cfg["m"])).astype(np.float64)
            v_pix = np.clip(1.0 - L, 0.0, 1.0)
            v_miou = np.broadcast_to(miou_raw[idx].reshape(-1, 1),
                                     (len(idx), cfg["m"])).astype(np.float64)
            return L, A, {"pixel_accuracy": v_pix, "miou": v_miou}

        L, A, vmap = build(cal)
        Lt, At, vtmap = build(test)
        cal_data[seed] = (L, A, vmap)
        test_data[seed] = (At, vtmap)
    rec = surface_records(cal_data, test_data, cfg,
                          ["pixel_accuracy", "miou"], "pixel_accuracy")
    rec["seconds"] = round(time.time() - t0, 1)
    print("COCO", json.dumps({k: v for k, v in rec.items() if k != "per_seed"},
                             indent=None), flush=True)
    return rec


def main():
    imagenet = imagenet_block()
    coco = coco_block()
    anchors = {
        "coco_default_median_u_lcb": coco["block_a_c_sweep"]["c=0.0"]["median_u_lcb_sel"],
        "coco_default_median_two_gamma_u":
            coco["block_a_c_sweep"]["c=0.0"]["median_two_gamma_u"],
        "published_u_lcb": 0.199,
        "published_two_gamma_u": 0.062,
    }
    out = {
        "experiment": "analysis_vc_sensitivity",
        "purpose": ("Two-parameter (v, c) sensitivity of the utility leg: "
                    "well-specified c-sweep, value-family sweep with joint "
                    "unit-scaling equivariance checks, and cost "
                    "misspecification at a reference cost, with Ghat/"
                    "feasibility invariance asserted per run."),
        "definition": {
            "invariance": "Ghat = {EB <= 0 and p_LCB >= pi_min} involves "
                          "neither v nor c; asserted np.array_equal per "
                          "setting against the surface-default run",
            "two_gamma_u": "2 * grid max of eta_U = u_bar - U_LCB (exact; "
                           "no clipping in certify_grid); the certified-set "
                           "optimality tolerance of the corollary",
            "transfer": "U_LCB(c_cert) - max(c_ref - c_cert, 0) * "
                        "(1 - p_LCB) at the returned pair; valid for the "
                        "c_ref functional on the grid-uniform CP and "
                        "utility events, no additional failure budget",
            "c_ref": "0.1 on both surfaces; equals the ImageNet operating "
                     "specification, and is an instantiated counterfactual "
                     "reference on COCO (operating specification c = 0)",
            "test_side": "test-split utilities are descriptive supporting "
                         "evidence (paper convention), not the guarantee; "
                         "the rho = 1 exceedance row is the test-noise floor",
            "aggregation": "returned-pair statistics over feasible runs "
                           "only; the COCO infeasible split stays "
                           "infeasible at every setting (Ghat invariance)",
        },
        "config": {"imagenet": {k: v for k, v in IMAGENET.items()},
                   "imagenet_grid": {"Lambda": LAMBDA.tolist(), "T": T.tolist()},
                   "coco": {k: v for k, v in COCO.items() if k != "dir"},
                   "c_sweep": C_SWEEP, "c_ref": C_REF, "rhos": RHOS,
                   "kappa": KAPPA},
        "imagenet": imagenet,
        "coco": coco,
        "anchors": anchors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("anchors", json.dumps(anchors))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
