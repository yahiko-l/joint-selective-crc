"""Constant-cost learned abstention on ImageNet: a (K+1)-class affine head on frozen features.

Features are the cached ResNet-50 (IMAGENET1K_V2) penultimate activations phi. The
head has K = 1000 class logits W phi + b and one abstention logit w phi + b_abs, is
initialised at the pretrained classifier (W_fc, b_fc) with a zero abstention row, and
is trained with the constant-cost surrogate of learning to defer,

    loss = -log s_y - (1 - c_rej) log s_abs,    s = softmax over the K+1 logits,

plus rho/2 (||W - W_fc||^2 + ||b - b_fc||^2 + ||w||^2) (mean loss over examples).
In the unrestricted population limit its own rule, abstain when the abstention logit
is the largest, is Chow's rule at rejection cost c_rej; the rule is fixed at training
time, without a separate post-training threshold-calibration step.

Protocol. The design split is the tuning split of split seed 42 in the headline
ImageNet protocol (8,500 images, reserved): it trains the head (rho chosen from
RHO_GRID by the held-out surrogate loss on a fixed 20% of the design split, then
refitted on all of it), fits the temperature of the scaled-softmax control, and fixes
every threshold grid. The other 41,500 images are re-split 30 times into a
certification split (33,000) and a test split (8,500).

Scores certified on the same footing (top-1 loss, value 1{correct}, abstention cost
c = 0.1), each with its own predictor, over thresholds at the design-split quantiles
for acceptance levels ACC_LEVELS plus one threshold at the design-split acceptance of
the head's own rule (for the learned score that threshold is 0):
  learned        g = max_k z_k - z_abs            (head predictor)
  head_softmax   max_k softmax(z_1..z_K)          (head predictor; isolates the abstention row)
  softmax        max softmax of the pretrained logits
  softmax_T      max softmax of the pretrained logits at the design-split NLL temperature
Also reported: the head's own rule on the test split, and an empirical selection
comparator (the learned score's grid, choosing on the certification split the
threshold with the largest empirical utility among those with empirical risk <= alpha
and empirical acceptance >= pi_min, without confidence corrections).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tvm
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selective_crc import certify_grid, three_split_indices  # noqa: E402

# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "learned_deferral"
DATA = Path(DATA_ROOT) / "imagenet_data"

DESIGN_SEED, N_DESIGN = 42, 8500
N_CERT, N_TEST = 33000, 8500
SEEDS = range(42, 72)
ACC_LEVELS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
PARAMS = dict(alpha=0.05, pi_min=0.01, delta=0.05, c=0.1, V=1.0, B=1.0)
RHO_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
LBFGS = dict(max_iter=1000, history_size=20, tolerance_grad=1e-8, tolerance_change=1e-12,
             line_search_fn="strong_wolfe")


def design_indices(n_total):
    """The tuning split of split seed 42 in the headline ImageNet protocol."""
    sub = np.random.default_rng(DESIGN_SEED).choice(n_total, size=n_total, replace=False)
    tune, _, _ = three_split_indices(n_total, n_tune=N_DESIGN, n_cert=N_CERT, seed=DESIGN_SEED)
    return np.sort(sub[tune])


def surrogate(Z, z_abs, y, c_rej):
    lse = torch.logsumexp(torch.cat([Z, z_abs[:, None]], dim=1), dim=1)
    return torch.mean((2.0 - c_rej) * lse - Z[torch.arange(len(y)), y] - (1.0 - c_rej) * z_abs)


def fit_head(phi, y, W0, b0, c_rej, rho, device):
    phi_t = torch.as_tensor(phi, dtype=torch.float64, device=device)
    y_t = torch.as_tensor(y, device=device)
    W0_t = torch.as_tensor(W0, dtype=torch.float64, device=device)
    b0_t = torch.as_tensor(b0, dtype=torch.float64, device=device)
    W = W0_t.clone().requires_grad_(True)
    b = b0_t.clone().requires_grad_(True)
    w = torch.zeros(phi.shape[1], dtype=torch.float64, device=device, requires_grad=True)
    b_abs = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b, w, b_abs], **LBFGS)

    def closure():
        opt.zero_grad()
        loss = surrogate(phi_t @ W.T + b, phi_t @ w + b_abs, y_t, c_rej)
        loss = loss + 0.5 * rho * ((W - W0_t).pow(2).sum() + (b - b0_t).pow(2).sum() + w.pow(2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    state = opt.state[opt._params[0]]
    final = float(closure().detach())            # objective and gradients at the returned parameters
    grad_max = float(max(p.grad.abs().max() for p in (W, b, w, b_abs)))
    diag = {"objective": final, "grad_max_abs": grad_max, "n_iter": int(state.get("n_iter", -1)),
            "func_evals": int(state.get("func_evals", -1)),
            "grad_tolerance_met": grad_max <= LBFGS["tolerance_grad"],
            "hit_iteration_budget": int(state.get("n_iter", 0)) >= LBFGS["max_iter"]}
    return ({"W": W.detach().cpu().numpy(), "b": b.detach().cpu().numpy(),
             "w": w.detach().cpu().numpy(), "b_abs": float(b_abs.detach())}, diag)


def held_out_surrogate(head, phi, y, c_rej):
    Z = phi @ head["W"].T + head["b"]
    z_abs = phi @ head["w"] + head["b_abs"]
    lse = logsumexp(np.concatenate([Z, z_abs[:, None]], axis=1), axis=1)
    return float(np.mean((2.0 - c_rej) * lse - Z[np.arange(len(y)), y] - (1.0 - c_rej) * z_abs))


def fit_temperature(z, y):
    def nll(t):
        zt = z / t
        return float(np.mean(logsumexp(zt, axis=1) - zt[np.arange(len(y)), y]))
    return float(minimize_scalar(nll, bounds=(0.05, 5.0), method="bounded").x)


def threshold_for(score_design, acc):
    """Threshold whose strict-inequality design acceptance is as close to acc as ties allow."""
    if acc >= 1.0:
        return float(np.nextafter(score_design.min(), -np.inf))
    if acc <= 0.0:
        return float(score_design.max())
    return float(np.quantile(score_design, 1.0 - acc))


def grid(score_design, native_acc, native_tau=None):
    """ACC_LEVELS thresholds plus the native-acceptance slot; duplicates are kept so that every
    system certifies over the same number of candidates."""
    taus = [threshold_for(score_design, a) for a in ACC_LEVELS]
    taus.append(native_tau if native_tau is not None else threshold_for(score_design, native_acc))
    return np.array(taus)


def evaluate_pair(accept_test, err_test):
    p = float(accept_test.mean())
    R = float(err_test[accept_test].mean()) if accept_test.any() else None
    U = float(np.mean(accept_test * (1.0 - err_test) - PARAMS["c"] * (1.0 - accept_test)))
    return {"test_p_acc": p, "test_R_sel": R, "test_U_dep": U, "test_accepted": int(accept_test.sum()),
            "test_risk_exceedance": bool(R is not None and R > PARAMS["alpha"]),
            "test_acceptance_shortfall": bool(p < PARAMS["pi_min"])}


def curve(score, err, taus):
    """Acceptance and selected top-1 error at every threshold (descriptive)."""
    A = score[:, None] > taus[None, :]
    n_acc = A.sum(axis=0)
    risk = [float((A[:, j] * err).sum() / n_acc[j]) if n_acc[j] else None for j in range(len(taus))]
    return {"p_acc": (n_acc / len(score)).tolist(), "R_sel": risk}


def certify_system(score_c, err_c, score_t, err_t, taus):
    A = (score_c[:, None] > taus[None, :]).astype(np.float64)
    L = np.repeat(err_c[:, None], len(taus), axis=1)
    res = certify_grid(L, A, 1.0 - L, check_sample_size=True, **PARAMS)
    out = {"m": len(taus), "feasible": not res.is_infeasible, "n_certified": int(res.n_certified),
           "cert_curve": curve(score_c, err_c, taus), "test_curve": curve(score_t, err_t, taus)}
    if not res.is_infeasible:
        k = int(res.selected)
        out.update({"tau": float(taus[k]), "p_lcb": float(res.p_lcb_per_pair[k]),
                    "u_lcb": float(res.u_lcb_per_pair[k]), **evaluate_pair(score_t > taus[k], err_t)})
        out["test_utility_below_lcb"] = bool(out["test_U_dep"] < out["u_lcb"])
    out["taus"] = taus.tolist()
    return out


def empirical_selection(score_c, err_c, score_t, err_t, taus):
    A = score_c[:, None] > taus[None, :]
    p_hat = A.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        R_hat = (A * err_c[:, None]).sum(axis=0) / A.sum(axis=0)
    U_hat = (A * (1.0 - err_c[:, None]) - PARAMS["c"] * (1.0 - A)).mean(axis=0)
    ok = (A.sum(axis=0) > 0) & (R_hat <= PARAMS["alpha"]) & (p_hat >= PARAMS["pi_min"])
    if not ok.any():
        return {"selected": False}
    k = int(np.flatnonzero(ok)[np.argmax(U_hat[ok])])
    return {"selected": True, "tau": float(taus[k]), **evaluate_pair(score_t > taus[k], err_t)}


def median_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def summarise(runs, key):
    rows = [r[key] for r in runs]
    deployed = [x for x in rows if x.get("feasible", x.get("selected"))]
    out = {"splits": len(rows), "deployed": len(deployed),
           "test_risk_exceedances": sum(x["test_risk_exceedance"] for x in deployed),
           "test_acceptance_shortfalls": sum(x["test_acceptance_shortfall"] for x in deployed),
           "deployed_without_test_acceptance": sum(x["test_R_sel"] is None for x in deployed),
           "median_test_p_acc": median_or_none([x["test_p_acc"] for x in deployed]),
           "median_test_R_sel": median_or_none([x["test_R_sel"] for x in deployed]),
           "median_test_U_dep": median_or_none([x["test_U_dep"] for x in deployed])}
    if deployed and "p_lcb" in deployed[0]:
        out.update({"median_p_lcb": median_or_none([x["p_lcb"] for x in deployed]),
                    "median_u_lcb": median_or_none([x["u_lcb"] for x in deployed]),
                    "test_utility_below_lcb": sum(x["test_utility_below_lcb"] for x in deployed)})
    return out


def main(c_rejs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = np.load(DATA / "val_labels.npy").astype(np.int64)
    phi_all = np.load(DATA / "val_features_resnet50_v2.npy").astype(np.float64)
    state = tvm.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(map_location="cpu")
    W_fc = state["fc.weight"].double().numpy()
    b_fc = state["fc.bias"].double().numpy()
    n_total = len(labels)

    design = design_indices(n_total)
    pool = np.setdiff1d(np.arange(n_total), design)
    assert len(pool) == N_CERT + N_TEST
    splits = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(pool)
        splits.append((seed, perm[:N_CERT], perm[N_CERT:]))

    # pretrained predictor and its two softmax scores (independent of c_rej)
    logits_pre = phi_all @ W_fc.T + b_fc
    err_pre = (logits_pre.argmax(axis=1) != labels).astype(np.float64)
    temp = fit_temperature(logits_pre[design], labels[design])
    s_pre = softmax(logits_pre, axis=1).max(axis=1)
    s_pre_T = softmax(logits_pre / temp, axis=1).max(axis=1)

    # fixed internal split of the design split for choosing rho
    perm_d = np.random.default_rng(0).permutation(design)
    d_fit, d_val = perm_d[: int(0.8 * len(design))], perm_d[int(0.8 * len(design)):]

    report = {"design": {"c_rej": list(c_rejs), "acc_levels": ACC_LEVELS, "rho_grid": RHO_GRID,
                         "lbfgs": {k: v for k, v in LBFGS.items()}, "params": PARAMS,
                         "n_design": len(design), "n_cert": N_CERT, "n_test": N_TEST,
                         "seeds": [SEEDS.start, SEEDS.stop - 1], "device": device,
                         "design_split": "tuning split of split seed 42 in the headline protocol"},
              "pretrained": {"design_temperature": temp, "top1_error_pool": float(err_pre[pool].mean())},
              "by_c_rej": {}}
    for c_rej in c_rejs:
        val_loss, cand_diag = {}, {}
        for rho in RHO_GRID:
            head, cand_diag[rho] = fit_head(phi_all[d_fit], labels[d_fit], W_fc, b_fc, c_rej, rho, device)
            val_loss[rho] = held_out_surrogate(head, phi_all[d_val], labels[d_val], c_rej)
        rho = min(val_loss, key=val_loss.get)
        head, diag = fit_head(phi_all[design], labels[design], W_fc, b_fc, c_rej, rho, device)
        Z = phi_all @ head["W"].T + head["b"]
        z_abs = phi_all @ head["w"] + head["b_abs"]
        err_h = (Z.argmax(axis=1) != labels).astype(np.float64)       # prediction among the K classes
        g_learn = Z.max(axis=1) - z_abs
        s_head = softmax(Z, axis=1).max(axis=1)
        native_acc = float((g_learn[design] > 0).mean())
        systems = {
            "learned": (g_learn, err_h, grid(g_learn[design], native_acc, native_tau=0.0)),
            "head_softmax": (s_head, err_h, grid(s_head[design], native_acc)),
            "softmax": (s_pre, err_pre, grid(s_pre[design], native_acc)),
            "softmax_T": (s_pre_T, err_pre, grid(s_pre_T[design], native_acc)),
        }
        runs = []
        for seed, cert, test in splits:
            row = {"seed": seed}
            own = g_learn[test] > 0
            row["own_rule"] = {**evaluate_pair(own, err_h[test]),
                               "test_chow_cost": float(np.mean(err_h[test] * own + c_rej * (1.0 - own)))}
            for name, (score, err, taus) in systems.items():
                row[name] = certify_system(score[cert], err[cert], score[test], err[test], taus)
            taus = systems["learned"][2]
            row["empirical_selection"] = empirical_selection(g_learn[cert], err_h[cert], g_learn[test], err_h[test], taus)
            runs.append(row)
            print(f"c_rej={c_rej} seed={seed} own p={row['own_rule']['test_p_acc']:.3f} R={row['own_rule']['test_R_sel']} | "
                  + " | ".join(f"{n}: p={row[n].get('test_p_acc')} R={row[n].get('test_R_sel')}" for n in systems)
                  + f" | emp p={row['empirical_selection'].get('test_p_acc')} R={row['empirical_selection'].get('test_R_sel')}",
                  flush=True)
        own_rows = [r["own_rule"] for r in runs]
        report["by_c_rej"][str(c_rej)] = {
            "rho": rho, "rho_val_loss": {str(k): v for k, v in val_loss.items()},
            "rho_candidate_fits": {str(k): v for k, v in cand_diag.items()}, "fit": diag,
            "design_native_acceptance": native_acc,
            "realised_design_acceptance": {n: [float((sys_[0][design] > t).mean()) for t in sys_[2]]
                                           for n, sys_ in systems.items()},
            "head_top1_error_pool": float(err_h[pool].mean()),
            "prediction_agreement_with_pretrained_pool": float((Z[pool].argmax(1) == logits_pre[pool].argmax(1)).mean()),
            "grids": {n: sys_[2].tolist() for n, sys_ in systems.items()},
            "own_rule": {"median_test_p_acc": median_or_none([o["test_p_acc"] for o in own_rows]),
                         "median_test_R_sel": median_or_none([o["test_R_sel"] for o in own_rows]),
                         "median_test_chow_cost": median_or_none([o["test_chow_cost"] for o in own_rows]),
                         "test_risk_exceedances": sum(o["test_risk_exceedance"] for o in own_rows),
                         "test_acceptance_shortfalls": sum(o["test_acceptance_shortfall"] for o in own_rows),
                         "splits_without_test_acceptance": sum(o["test_R_sel"] is None for o in own_rows)},
            "empirical_selection": summarise(runs, "empirical_selection"),
            **{n: summarise(runs, n) for n in systems},
            "test_curves": {n: {"mean_p_acc": np.mean([r[n]["test_curve"]["p_acc"] for r in runs], axis=0).tolist(),
                                "median_R_sel": [median_or_none([r[n]["test_curve"]["R_sel"][j] for r in runs])
                                          for j in range(len(systems[n][2]))]}
                            for n in systems},
            "per_split": runs,
        }
        print(json.dumps({k: v for k, v in report["by_c_rej"][str(c_rej)].items() if k != "per_split"}, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "imagenet_learned_deferral.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-rej", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    main(parser.parse_args().c_rej)
