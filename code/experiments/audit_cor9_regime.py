"""Corollary 9 (regime-separation) FULL AUDIT.

Addresses reviewer M4: the manuscript instantiates the closed-form regime
predictor (Eq. cor-V, leading-order; Eq. cor-V-exact, finite-sample iff) on
five representative pairs (Table III). This script runs the systematic
per-(grid, seed) audit the manuscript defers to future work, producing a
confusion matrix of the closed-form prediction vs the realised per-pair winner
and the misclassification rate inside/outside the explicit near-threshold band
of Eq. (cor-explicit-slack).

It is PURE POST-PROCESSING of artefacts already in the submission:
  - COCO: recomputed from the bundled per-image arrays (data/coco/*.npy),
          giving the genuine leading-order test (T_obs needs the accepted-loss
          variance, computed directly from data) on all 15 grid pairs x 30 seeds
          across three loss families spanning the accepted-variance range.
  - ImageNet: reconstructed from results/imagenet_primary_real/results.json,
          which caches per-(pair, seed) `ours_r_margin` (= eta_Z / p_hat) and
          `p_hat` but NOT the per-pair accepted-loss variance. We therefore
          verify the finite-sample iff (Eq. cor-V-exact) and report the realised
          per-pair winner census over 35 pairs x 30 seeds; the leading-order
          T_obs leg is not independently computable there and is omitted.

Definitions (B = 1 on every surface here; matched to src/selective_crc and
sections/4_theory.tex):
    L_O          = log(64 m / delta)                       (eta_Z / sigma* log arg)
    L_H          = log(m / delta)                          (Hoeffding-CRC log arg)
    kappa_n      = n / (n - 1)
    eta_Z        = maurer_pontil_two_sided_radius(Sigma_Z, n, delta/(16 m), B)
                 = sqrt(2 Sigma_Z L_O / n) + 7 B L_O / (3 (n-1))
    UCB_ours-hw  = eta_Z / p_hat                           (half-width above R_sel_hat)
    UCB_hoeff-hw = B sqrt(L_H / (2 s))                     (s = accepted count)
    actual: Ours tighter  <=>  UCB_ours-hw < UCB_hoeff-hw  (R_sel_hat cancels)
    T_obs        = sigma_hat^2_acc + (1 - p_hat)(R_sel_hat - alpha)^2   (leading-order)
    T_ex         = Sigma_Z / p_hat                                       (exact)
    sigma*(s)    = (B^2 / 2 L_O) [ sqrt(L_H/2) - 7 L_O/(3 sqrt s) ]_+^2          (cor-V)
    sigma_ex(s)  = (B^2 / 2 L_O) [ sqrt(L_H/2) - kappa_n 7 L_O/(3 sqrt s) ]_+^2  (cor-V-exact)
    pred (LO):   Ours tighter <=> T_obs < sigma*(s)
    pred (exact):Ours tighter <=> T_ex  < sigma_ex(s)      (== actual, by the identity)
    near-threshold band: |T_obs - sigma*(s)| <= 4 B^2 (1/s + 1/n)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE / "src"))

# Import the *canonical* radius used by the certifier so eta_Z is identical.
from selective_crc.bounds import maurer_pontil_two_sided_radius

COCO_DIR = CODE / "data" / "coco"
IMAGENET_JSON = CODE / "results" / "imagenet_primary_real" / "results.json"
OUT_JSON = CODE / "results" / "analysis" / "cor9_regime_audit.json"
B = 1.0


# --------------------------------------------------------------------------
# Closed-form pieces (Corollary 9)
# --------------------------------------------------------------------------
def sigma_star(s, m, delta, B=1.0):
    """Leading-order threshold, Eq. (cor-V). Vectorised over s."""
    s = np.asarray(s, dtype=np.float64)
    Lo = np.log(64.0 * m / delta)
    Lh = np.log(m / delta)
    inner = np.maximum(np.sqrt(Lh / 2.0) - 7.0 * Lo / (3.0 * np.sqrt(s)), 0.0)
    return (B**2 / (2.0 * Lo)) * inner**2


def sigma_exact(s, n, m, delta, B=1.0):
    """Finite-sample threshold, Eq. (cor-V-exact), with kappa_n correction."""
    s = np.asarray(s, dtype=np.float64)
    Lo = np.log(64.0 * m / delta)
    Lh = np.log(m / delta)
    kappa = n / (n - 1.0)
    inner = np.maximum(np.sqrt(Lh / 2.0) - kappa * 7.0 * Lo / (3.0 * np.sqrt(s)), 0.0)
    return (B**2 / (2.0 * Lo)) * inner**2


def eta_Z_from_sigma_Z(sigma_Z, n, m, delta, B=1.0):
    """eta_Z at the certifier's delta/(16 m) level (identical to certify_grid)."""
    return maurer_pontil_two_sided_radius(sigma_Z, n, delta / (16.0 * m), B)


def hoeff_halfwidth(s, m, delta, B=1.0):
    s = np.asarray(s, dtype=np.float64)
    return B * np.sqrt(np.log(m / delta) / (2.0 * s))


def band_halfwidth(s, n, B=1.0):
    """Explicit near-threshold band radius, Eq. (cor-explicit-slack): 4 B^2 (1/s + 1/n)."""
    s = np.asarray(s, dtype=np.float64)
    return 4.0 * B**2 * (1.0 / s + 1.0 / n)


# --------------------------------------------------------------------------
# COCO: full per-(pair, seed) recomputation from bundled arrays
# --------------------------------------------------------------------------
def build_grid_from_taus(g, L_raw, tau_grid):
    """Identical to experiments/ablation_g_ade20k_segmentation.build_grid_from_taus."""
    n = len(g)
    m = len(tau_grid)
    L = np.broadcast_to(L_raw.reshape(n, 1), (n, m)).astype(np.float64)
    A = (g.reshape(n, 1) > tau_grid.reshape(1, m)).astype(np.float64)
    return L, A


def load_coco_loss(loss_type, binary_threshold=0.3):
    miou = np.load(COCO_DIR / "val_mask2former_coco_miou.npy").astype(np.float64)
    if loss_type == "pixacc":
        L = np.load(COCO_DIR / "val_mask2former_coco_loss_pixacc.npy").astype(np.float64)
        return np.clip(L, 0.0, 1.0)
    if loss_type == "continuous":
        return np.clip(1.0 - miou, 0.0, 1.0)
    if loss_type == "binary":
        return (miou < binary_threshold).astype(np.float64)
    raise ValueError(loss_type)


def audit_coco_config(loss_type, alpha, pi_min, delta, g_name="g_softmax",
                      n_cal=4000, n_test=1000, n_tau=15, n_seeds=30,
                      binary_threshold=0.3, min_s=2):
    """Recompute every per-(pair, seed) statistic and classify it.

    Mirrors ablation_g_coco_segmentation.run_one_seed_coco's split exactly:
    rng = default_rng(42 + s); perm; cal = perm[:n_cal]; tau = quantile(g_cal, 0.5..0.95).
    """
    L_raw = load_coco_loss(loss_type, binary_threshold)
    g = np.load(COCO_DIR / f"val_mask2former_coco_{g_name}.npy").astype(np.float64)
    m = n_tau
    cells = []  # one record per (seed, grid-pair) with s >= min_s
    for s_idx in range(n_seeds):
        seed = 42 + s_idx
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(g))
        cal_idx = perm[:n_cal]
        g_cal = g[cal_idx]
        tau_grid = np.quantile(g_cal, np.linspace(0.5, 0.95, n_tau))
        L_cal, A_cal = build_grid_from_taus(g_cal, L_raw[cal_idx], tau_grid)
        Z = A_cal * (L_cal - alpha)
        for k in range(m):
            a = A_cal[:, k]
            s = int(a.sum())
            if s < min_s:
                continue
            p_hat = s / n_cal
            L_acc = L_cal[a > 0.5, k]
            R_sel_hat = float(L_acc.mean())
            sigma_acc = float(L_acc.var(ddof=1))           # within-accepted loss variance
            sigma_Z = float(Z[:, k].var(ddof=1))           # certify_grid convention (ddof=1)
            eta_Z = float(eta_Z_from_sigma_Z(sigma_Z, n_cal, m, delta, B))
            ucb_ours = eta_Z / p_hat
            ucb_hoeff = float(hoeff_halfwidth(s, m, delta, B))
            T_obs = sigma_acc + (1.0 - p_hat) * (R_sel_hat - alpha) ** 2
            T_ex = sigma_Z / p_hat
            sstar = float(sigma_star(s, m, delta, B))
            sexact = float(sigma_exact(s, n_cal, m, delta, B))
            band = float(band_halfwidth(s, n_cal, B))
            cells.append(dict(
                seed=seed, k=k, s=s, p_hat=p_hat, R_sel_hat=R_sel_hat,
                sigma_acc=sigma_acc, sigma_Z=sigma_Z,
                ucb_ours=ucb_ours, ucb_hoeff=ucb_hoeff,
                T_obs=T_obs, T_ex=T_ex, sigma_star=sstar, sigma_exact=sexact, band=band,
                actual_ours=bool(ucb_ours < ucb_hoeff),
                pred_lo_ours=bool(T_obs < sstar),
                pred_exact_ours=bool(T_ex < sexact),
                in_band=bool(abs(T_obs - sstar) <= band),
            ))
    return dict(
        surface=f"COCO/{loss_type}", config=dict(
            loss=loss_type, alpha=alpha, pi_min=pi_min, delta=delta, m=m,
            n_cal=n_cal, n_test=n_test, n_seeds=n_seeds, g=g_name,
            binary_threshold=binary_threshold if loss_type == "binary" else None),
        cells=cells)


# --------------------------------------------------------------------------
# ImageNet: reconstruct from cached ours_r_margin + p_hat (finite-sample iff leg)
# --------------------------------------------------------------------------
def invert_sigma_Z(eta_Z, n, m, delta, B=1.0):
    """Invert eta_Z = sqrt(2 Sigma_Z L_O / n) + 7 B L_O/(3(n-1)) for Sigma_Z."""
    Lo = np.log(64.0 * m / delta)
    lower = 7.0 * B * Lo / (3.0 * (n - 1.0))
    a = np.maximum(eta_Z - lower, 0.0)
    return (a**2) * n / (2.0 * Lo)


def audit_imagenet(n_cert=33000, m=35, delta=0.05, min_s=2):
    d = json.load(open(IMAGENET_JSON))
    cfg = d["config"]
    B_in = float(cfg.get("B", 1.0))
    cells = []
    for ps in d["per_seed"]:
        seed = ps["split_seed"]
        margins = np.asarray(ps["ours_r_margin"], dtype=np.float64)   # eta_Z / p_hat
        p_hat = np.asarray(ps["p_hat"], dtype=np.float64)
        s_arr = np.rint(p_hat * n_cert).astype(int)
        eta_Z = margins * p_hat
        sigma_Z = invert_sigma_Z(eta_Z, n_cert, m, delta, B_in)
        T_ex = sigma_Z / p_hat
        ucb_hoeff = hoeff_halfwidth(s_arr, m, delta, B_in)
        sexact = sigma_exact(s_arr, n_cert, m, delta, B_in)
        for k in range(len(margins)):
            s = int(s_arr[k])
            if s < min_s:
                continue
            cells.append(dict(
                seed=seed, k=k, s=s, p_hat=float(p_hat[k]),
                ucb_ours=float(margins[k]), ucb_hoeff=float(ucb_hoeff[k]),
                T_ex=float(T_ex[k]), sigma_exact=float(sexact[k]),
                actual_ours=bool(margins[k] < ucb_hoeff[k]),
                pred_exact_ours=bool(T_ex[k] < sexact[k]),
            ))
    return dict(surface="ImageNet/RN50-V2",
                config=dict(n_cert=n_cert, m=m, delta=delta, note="reconstructed from cached ours_r_margin + p_hat; leading-order T_obs not cached"),
                cells=cells)


# --------------------------------------------------------------------------
# Confusion-matrix reporting
# --------------------------------------------------------------------------
def confusion(cells, pred_key):
    """2x2: rows = prediction (Ours/Hoeff), cols = actual (Ours/Hoeff)."""
    cm = {("O", "O"): 0, ("O", "H"): 0, ("H", "O"): 0, ("H", "H"): 0}
    for c in cells:
        p = "O" if c[pred_key] else "H"
        a = "O" if c["actual_ours"] else "H"
        cm[(p, a)] += 1
    n = len(cells)
    agree = cm[("O", "O")] + cm[("H", "H")]
    return cm, n, agree


def report_surface(res, leading_order=True):
    cells = res["cells"]
    lines = []
    lines.append(f"### {res['surface']}  (n_cells={len(cells)}; {res['config']})")
    # exact-iff leg (always available)
    cm_e, n_e, agree_e = confusion(cells, "pred_exact_ours")
    acc_e = agree_e / n_e if n_e else float("nan")
    lines.append(f"  finite-sample iff (Eq. cor-V-exact)  vs actual : "
                 f"agree {agree_e}/{n_e} = {acc_e*100:.2f}%   "
                 f"[OO={cm_e[('O','O')]} HH={cm_e[('H','H')]} OH={cm_e[('O','H')]} HO={cm_e[('H','O')]}]")
    n_ours_actual = sum(c["actual_ours"] for c in cells)
    lines.append(f"  realised per-pair winner census       : Ours {n_ours_actual}/{len(cells)} "
                 f"({100*n_ours_actual/len(cells):.1f}%) | Hoeffding {len(cells)-n_ours_actual}")
    out = dict(surface=res["surface"], n_cells=len(cells),
               iff_agree=agree_e, iff_n=n_e, iff_acc=acc_e,
               actual_ours=n_ours_actual)
    if leading_order:
        cm, n, agree = confusion(cells, "pred_lo_ours")
        acc = agree / n if n else float("nan")
        in_band = [c for c in cells if c["in_band"]]
        out_band = [c for c in cells if not c["in_band"]]
        ob_dis = sum(1 for c in out_band if c["pred_lo_ours"] != c["actual_ours"])
        ib_dis = sum(1 for c in in_band if c["pred_lo_ours"] != c["actual_ours"])
        ob_acc = (len(out_band) - ob_dis) / len(out_band) if out_band else float("nan")
        lines.append(f"  leading-order rule (Eq. cor-V)        vs actual : "
                     f"agree {agree}/{n} = {acc*100:.2f}%   "
                     f"[OO={cm[('O','O')]} HH={cm[('H','H')]} OH={cm[('O','H')]} HO={cm[('H','O')]}]")
        lines.append(f"    out-of-band  : {len(out_band)} cells, {ob_dis} disagreements "
                     f"({ob_acc*100:.2f}% agree)   <- the predictor's real accuracy")
        lines.append(f"    in-band      : {len(in_band)} cells ({100*len(in_band)/n:.1f}%), "
                     f"{ib_dis} disagreements (band is where Eq. cor-explicit-slack permits either verdict)")
        out.update(lo_agree=agree, lo_n=n, lo_acc=acc,
                   n_in_band=len(in_band), n_out_band=len(out_band),
                   out_band_disagree=ob_dis, out_band_acc=ob_acc, in_band_disagree=ib_dis)
    return "\n".join(lines), out


def main():
    np.seterr(all="ignore")
    results = []
    print("=" * 78)
    print("Corollary 9 regime-separation FULL AUDIT  (reviewer M4)")
    print("=" * 78)

    # COCO: three loss families spanning the accepted-variance range.
    coco_cfgs = [
        dict(loss_type="pixacc",     alpha=0.10, pi_min=0.10, delta=0.10),  # headline, low variance
        dict(loss_type="continuous", alpha=0.40, pi_min=0.10, delta=0.10),  # mid variance
        dict(loss_type="binary",     alpha=0.20, pi_min=0.10, delta=0.10),  # high variance (Hoeffding side)
    ]
    coco_audits = []
    for cfg in coco_cfgs:
        res = audit_coco_config(**cfg)
        coco_audits.append(res)
        txt, summ = report_surface(res, leading_order=True)
        print("\n" + txt)
        results.append(summ)

    # Pooled COCO leading-order audit
    pooled_cells = [c for r in coco_audits for c in r["cells"]]
    pooled = dict(surface="COCO/POOLED (3 losses)", config="pixacc+continuous+binary, 15 pairs x 30 seeds each",
                  cells=pooled_cells)
    print("\n" + "-" * 78)
    txt, summ = report_surface(pooled, leading_order=True)
    print(txt)
    results.append(summ)

    # Small-n_cal STRESS audit: at n_cal=400 the 1/s + 1/n corrections are
    # non-negligible, so the leading-order rule CAN disagree with the exact
    # winner. The claim under test is that any such disagreement is confined to
    # the near-threshold band of Eq. (cor-explicit-slack).
    print("\n" + "=" * 78)
    print("STRESS: n_cal=400 (large finite-sample corrections -> band should bind)")
    print("=" * 78)
    stress_audits = []
    for cfg in coco_cfgs:
        res = audit_coco_config(n_cal=400, n_test=1000, **cfg)
        stress_audits.append(res)
    stress_pooled = dict(surface="COCO/STRESS-POOLED (n_cal=400)",
                         config="3 losses, 15 pairs x 30 seeds, n_cal=400",
                         cells=[c for r in stress_audits for c in r["cells"]])
    txt, summ_stress = report_surface(stress_pooled, leading_order=True)
    print("\n" + txt)
    results.append(summ_stress)

    # ImageNet: finite-sample iff + census (no independent leading-order leg)
    # The ImageNet block re-reads the cached per-(pair, seed) run. That artifact is not
    # part of this code-only bundle; the COCO blocks above are self-contained.
    if IMAGENET_JSON.exists():
        img = audit_imagenet()
        print("\n" + "-" * 78)
        txt, summ = report_surface(img, leading_order=False)
        print(txt)
        results.append(summ)
    else:
        print("\n" + "-" * 78)
        print(f"### ImageNet/RN50-V2  SKIPPED: {IMAGENET_JSON} not present in this bundle.")
        print("    Regenerate it with experiments/imagenet_scale.py "
              "--config configs/imagenet_primary_real.yaml (needs the ImageNet logit cache).")

    # Sanity vs Table III: COCO pixacc certifier-selected (p_hat~0.22) & q=0.85 (k=11) pairs.
    # tau_grid = quantile(g_cal, linspace(0.5,0.95,15)) => k=11 is the q~0.854 threshold.
    print("\n" + "-" * 78)
    print("Sanity check vs Table III (COCO pixacc, medians over 30 seeds):")
    print(f"  [direct] sigma*(s=880) = {float(sigma_star(880, 15, 0.10, 1.0)):.4f}  (paper Table III: 0.041)")
    pix = coco_audits[0]["cells"]
    ks = sorted(set(c["k"] for c in pix))
    med_phat = {kk: float(np.median([c["p_hat"] for c in pix if c["k"] == kk])) for kk in ks}
    k_cert = min(ks, key=lambda kk: abs(med_phat[kk] - 0.221))   # certifier-selected p_acc~0.221
    for k_target, label in [(k_cert, f"certifier pair (k={k_cert}, p_hat~0.22)"),
                            (11, "q=0.85 pair (k=11)")]:
        sub = [c for c in pix if c["k"] == k_target]
        if sub:
            print(f"  {label:34s}: T_obs~{np.median([c['T_obs'] for c in sub]):.4f}  "
                  f"sigma*~{np.median([c['sigma_star'] for c in sub]):.4f}  "
                  f"s~{int(np.median([c['s'] for c in sub]))}  p_hat~{np.median([c['p_hat'] for c in sub]):.3f}  "
                  f"Ours-wins {sum(c['actual_ours'] for c in sub)}/{len(sub)}")
    print("  (paper Table III COCO rows: T_obs=0.007<sigma*(880)=0.041 [certifier]; 0.007<0.027 [q=0.85])")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(summary=results,
                       coco=[dict(surface=r["surface"], config=r["config"], cells=r["cells"]) for r in coco_audits],
                       imagenet=(dict(surface=img["surface"], config=img["config"], cells=img["cells"])
                              if IMAGENET_JSON.exists() else None)),
                  f, indent=2)
    print(f"\nFull per-cell records written to: {OUT_JSON.relative_to(CODE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
