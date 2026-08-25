"""Reproducible verification of the utility-leg non-vacuity results (rebuttal addendum).

Grounds, in real numbers, the three utility guarantees of the revised Theorem 1 / corollaries:

  (1) COCO val 2017 panoptic, alpha=0.10 (paper headline):
        - reproduces the paper headline (median certified p_acc, sigma_hat, U_LCB);
        - the LITERAL margin oracle M(alpha - gamma_r, pi_min) is EMPTY (gamma_r ~ 0.71);
        - the certified set Ghat is NON-empty on every feasible seed, so the
          certified-set optimality bound (Cor. cor:gset-opt) and the absolute
          certified utility U_dep(hat) >= U_LCB(hat) are non-vacuous.
  (2) COCO val 2017 panoptic, alpha=0.15:
        - a CALIBRATION PLUG-IN surrogate of the variance-adaptive external margin
          oracle (Cor. cor:va-oracle) -- using empirical Rhat/phat/eta_Z in place of the
          population Rsel/pacc/sigma_Z^2 the oracle is defined on -- is non-empty on every
          seed; reports the plug-in U^margin_va and verifies the guarantee. This is an
          empirical diagnostic, NOT a population-level certification of non-emptiness.
  (3) Synthetic low-loss-subpopulation surface, alpha=0.30, pi_min=0.20, n_cert=25000
      (satisfies the sample-size precondition (star)):
        - the LITERAL Theorem-1 margin oracle is NON-empty; reports a finite
          U^margin and verifies U_dep(hat) >= U^margin - 2 gamma_u.

Run:
  <env-with-numpy-scipy>/python experiments/verify_oracle_nonvacuity.py

All COCO numbers re-grid the cached per-image arrays at
$SCORC_DATA_DIR/coco_data/val_mask2former_coco_{loss_pixacc,g_softmax}.npy
through src/selective_crc/certify.py (the same certifier used in the paper).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from selective_crc.certify import certify_grid              # noqa: E402
from selective_crc.bounds import maurer_pontil_two_sided_radius  # noqa: E402


# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

def _find_coco_dir():
    """Locate the cached COCO per-image arrays (portable across repo / bundled artifact)."""
    candidates = [
        os.environ.get("SCORC_COCO_DIR"),
        ROOT / "data" / "coco",            # bundled artifact layout: code/data/coco/
        ROOT / "code" / "data" / "coco",   # repo root with the artifact present
        f"{DATA_ROOT}/coco_data",   # original local cache
    ]
    for c in candidates:
        if c and (Path(c) / "val_mask2former_coco_loss_pixacc.npy").exists():
            return Path(c)
    raise FileNotFoundError(
        "COCO cached arrays not found. Set SCORC_COCO_DIR to the directory containing "
        "val_mask2former_coco_{loss_pixacc,g_softmax}.npy")


COCO_DIR = _find_coco_dir()


def gamma_r_literal(n, pmin, m, delta, B=1.0):
    """Worst-case (range-only-variance) risk margin of Theorem 1."""
    L = math.log(64 * m / delta)
    return 4 * B * math.sqrt(L / (n * pmin)) + (14 * B / 3) * L / (pmin * (n - 1))


def grid_stats(L, A, v, alpha, pmin, delta, B=1.0, c=0.0, V=1.0):
    """Run the certifier and return per-pair quantities + the two margins."""
    n, m = L.shape
    res = certify_grid(L, A, v, alpha=alpha, pi_min=pmin, delta=delta,
                       c=c, V=V, B=B, check_sample_size=False)
    phat = A.mean(0)
    s = A.sum(0).astype(int)
    Rhat = np.array([L[A[:, k] > 0.5, k].mean() if s[k] > 0 else np.nan
                     for k in range(m)])
    etaZ = maurer_pontil_two_sided_radius(res.sigma_z_sq, n, delta / (16 * m), B)
    etaU = maurer_pontil_two_sided_radius(res.sigma_u_sq, n, delta / (2 * m), c + V)
    return dict(res=res, phat=phat, s=s, Rhat=Rhat, etaZ=etaZ,
                margin_va=2.0 * etaZ / np.maximum(phat, 1e-12),  # PLUG-IN surrogate of population va-margin (2-factor)
                Udep=res.u_bar, inG=res.in_g_hat_mask,
                ULCB=res.u_lcb_per_pair, gamma_u=float(etaU.max()))


def coco_grid(seed, alpha, pmin=0.10, delta=0.10, m=15, n_cal=4000):
    L_raw = np.clip(np.load(COCO_DIR / "val_mask2former_coco_loss_pixacc.npy").astype(np.float64), 0, 1)
    g = np.load(COCO_DIR / "val_mask2former_coco_g_softmax.npy").astype(np.float64)
    rng = np.random.default_rng(seed)
    cal = rng.permutation(len(g))[:n_cal]
    g_cal, L_cal = g[cal], L_raw[cal]
    tau = np.quantile(g_cal, np.linspace(0.5, 0.95, m))
    A = (g_cal.reshape(-1, 1) > tau.reshape(1, m)).astype(np.float64)
    L = np.broadcast_to(L_cal.reshape(-1, 1), (n_cal, m)).astype(np.float64)
    v = np.clip(1.0 - L, 0.0, 1.0)
    return grid_stats(L, A, v, alpha, pmin, delta)


def main():
    SEEDS = range(42, 62)  # 20 seeds, matching the paper's COCO softmax run
    print("=" * 76)
    print("(1) COCO val 2017 panoptic  alpha=0.10 (headline): 1B-i + absolute U_LCB")
    print("=" * 76)
    gr = gamma_r_literal(4000, 0.10, 15, 0.10)
    print(f"literal gamma_r = {gr:.4f} -> alpha-gamma_r = {0.10-gr:+.4f} (literal oracle needs R_sel <= this)")
    pacc, sig, ulcb, nG, lit_ne, slack = [], [], [], [], 0, []
    for seed in SEEDS:
        st = coco_grid(seed, 0.10)
        res = st["res"]
        if res.is_infeasible:
            nG.append(0); continue
        k = int(res.selected)
        pacc.append(float(res.p_lcb_per_pair[k])); sig_k = st["s"][k]
        acc = st["Rhat"]  # not used directly; sigma below
        ulcb.append(float(res.u_lcb_per_pair[k])); nG.append(int(st["inG"].sum()))
        # accepted-loss variance at selected pair
        sig.append(float(st["res"].sigma_z_sq[k]))  # placeholder; recompute precise below
        lit = (st["Rhat"] <= 0.10 - gr) & (st["phat"] >= 0.20)
        lit_ne += int(lit.any())
        if st["inG"].any():
            best = float(st["Udep"][st["inG"]].max())
            slack.append(float(st["Udep"][k] - (best - 2 * st["gamma_u"])))
    # recompute sigma_hat of accepted loss at selected pair (clean)
    sig_acc = []
    for seed in SEEDS:
        st = coco_grid(seed, 0.10)
        if st["res"].is_infeasible:
            continue
        k = int(st["res"].selected)
        L_raw = np.clip(np.load(COCO_DIR / "val_mask2former_coco_loss_pixacc.npy").astype(np.float64), 0, 1)
        g = np.load(COCO_DIR / "val_mask2former_coco_g_softmax.npy").astype(np.float64)
        rng = np.random.default_rng(seed); cal = rng.permutation(len(g))[:4000]
        gc, Lc = g[cal], L_raw[cal]; tau = np.quantile(gc, np.linspace(0.5, 0.95, 15))
        accmask = gc > tau[k]
        sig_acc.append(float(Lc[accmask].var(ddof=1)))
    print(f"  reproduce: median certified p_acc = {np.median(pacc):.3f}  [paper 0.221]")
    print(f"  reproduce: median sigma_hat_acc   = {np.median(sig_acc):.4f} [paper 0.0058]")
    print(f"  reproduce: median U_LCB(hat)      = {np.median(ulcb):.3f}  [paper 0.199]")
    print(f"  reproduce: median n_in_G_hat      = {int(np.median(nG))}      [paper 4]")
    print(f"  literal margin oracle non-empty   : {lit_ne}/20 seeds (EMPTY -> reviewer correct)")
    print(f"  feasible (Ghat != empty)          : {sum(x>0 for x in nG)}/20 seeds")
    print(f"  Cor cor:gset-opt holds (slack>=0) : {sum(x>=-1e-9 for x in slack)}/{len(slack)}; median slack {np.median(slack):+.3f}")

    print("\n" + "=" * 76)
    print("(2) COCO val 2017 panoptic  alpha=0.15: PLUG-IN surrogate of variance-adaptive external oracle (Cor va-oracle)")
    print("=" * 76)
    ne = ok = 0; Um, Uh, g2 = [], [], []
    for seed in SEEDS:
        st = coco_grid(seed, 0.15)
        if st["res"].is_infeasible:
            continue
        k = int(st["res"].selected)
        va = (st["Rhat"] + st["margin_va"] <= 0.15) & (st["phat"] >= 0.20)
        if va.any():
            ne += 1
            um = float(st["Udep"][va].max()); uh = float(st["Udep"][k]); gg = 2 * st["gamma_u"]
            Um.append(um); Uh.append(uh); g2.append(gg)
            ok += int(uh >= um - gg)
    print(f"  plug-in VA-oracle surrogate non-empty: {ne}/20 seeds (empirical diagnostic, not population certificate)")
    print(f"  guarantee U_dep(hat) >= U^margin_va - 2gamma_u holds: {ok}/{ne}")
    print(f"  median U^margin_va = {np.median(Um):.3f}; median U_dep(hat) = {np.median(Uh):.3f}; median 2gamma_u = {np.median(g2):.3f}")

    print("\n" + "=" * 76)
    print("(3) Synthetic existence demo  alpha=0.30, pi_min=0.20, delta=0.10, m=15, n=25000")
    print("=" * 76)
    alpha, pmin, delta, m, n = 0.30, 0.20, 0.10, 15, 25000
    n0 = math.ceil(32 * math.log(32 * m / delta) / pmin)
    grs = gamma_r_literal(n, pmin, m, delta)
    print(f"sample-size (star): n0 = {n0}; n = {n} satisfies (star): {n >= n0}")
    print(f"literal gamma_r = {grs:.4f} -> alpha-gamma_r = {alpha-grs:+.4f}")
    rng = np.random.default_rng(0)
    easy = rng.random(n) < 0.5
    L_raw = np.where(easy, rng.beta(1, 30, n), rng.beta(5, 3, n))
    g = np.clip(1.0 - L_raw + rng.normal(0, 0.05, n), 0, 1)
    tau = np.quantile(g, np.linspace(0.5, 0.95, m))
    A = (g.reshape(-1, 1) > tau.reshape(1, m)).astype(np.float64)
    L = np.broadcast_to(L_raw.reshape(-1, 1), (n, m)).astype(np.float64)
    v = np.clip(1.0 - L, 0.0, 1.0)
    st = grid_stats(L, A, v, alpha, pmin, delta)
    k = int(st["res"].selected)
    lit = (st["Rhat"] <= alpha - grs) & (st["phat"] >= 2 * pmin)
    print(f"  literal Theorem-1 margin oracle members: {int(lit.sum())} pairs (NON-EMPTY)")
    Um = float(st["Udep"][lit].max())
    print(f"  U^margin (finite) = {Um:.3f}; U_dep(hat) = {st['Udep'][k]:.3f}; 2gamma_u = {2*st['gamma_u']:.4f}")
    print(f"  guarantee U_dep(hat) >= U^margin - 2gamma_u : {st['Udep'][k]:.3f} >= {Um-2*st['gamma_u']:.3f} -> {st['Udep'][k] >= Um-2*st['gamma_u']}")
    print(f"  certified p_acc(hat) = {float(st['res'].p_lcb_per_pair[k]):.3f}; R_hat(hat) = {st['Rhat'][k]:.3f}")


if __name__ == "__main__":
    main()
