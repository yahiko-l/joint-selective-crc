"""Held-out, high-confidence certification that the external margin oracle is non-empty.

Addresses reviewer M1 / Q1: upgrade the calibration *plug-in* diagnostic of
Mset(alpha - gamma_r, pi_min) non-emptiness (verify_oracle_nonvacuity.py case 3,
which compares the empirical Rhat/phat to thresholds on the same split) to a
*population-level* certification on an INDEPENDENT held-out split.

Certificate (for a grid built independently of the held-out split). For each pair
(lambda, tau) we compute, on the held-out split, at per-event level delta'/(3m):
  - EB-UCB on E[Z'] with Z' = A (L - alpha'),  alpha' := alpha - gamma_r;
  - CP-LCB on p_acc = E[A];
  - EB-LCB on U_dep = E[A v - c(1 - A)].
Union over the m pairs and the 3 event families gives a (1 - delta') event on which
ALL bounds hold simultaneously. On that event, any pair with

      EB-UCB(Z') <= 0   AND   CP-LCB(p_acc) >= 2 pi_min

satisfies R_sel <= alpha' and p_acc >= 2 pi_min, i.e. it is a *certified member* of
Mset(alpha - gamma_r, pi_min); its existence certifies the oracle set is non-empty,
and U_margin(alpha, pi_min) >= EB-LCB(U_dep) at that member (a finite, population-valid
lower bound on the oracle value, replacing the plug-in point estimate).

gamma_r is the certificate's deterministic margin gamma_r_literal(n_cert, pi_min, m, delta);
the held-out certification uses a fresh confidence budget delta' independent of delta.

Run: <env-with-numpy-scipy>/python experiments/verify_oracle_heldout.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from selective_crc.bounds import (                              # noqa: E402
    clopper_pearson_lower,
    empirical_bernstein_ucb,
    empirical_bernstein_lcb,
)
from verify_oracle_nonvacuity import gamma_r_literal, _find_coco_dir  # noqa: E402

OUT_JSON = ROOT / "results" / "analysis" / "oracle_heldout_cert.json"


def certify_mset_nonempty(L_ho, A_ho, v_ho, alpha_prime, two_pmin, delta_prime,
                          B=1.0, c=0.0, V=1.0):
    """High-confidence certification of Mset(alpha', pi_min) non-emptiness on a held-out split.

    Returns a dict with the certified-member mask, the binding-constraint diagnostics,
    and (if any member is certified) a population-valid lower bound on U_margin.
    """
    n_ho, m = L_ho.shape
    de = delta_prime / (3.0 * m)            # per-event Bonferroni share (3 families x m pairs)

    Zp = A_ho * (L_ho - alpha_prime)        # E[Z'] <= 0  <=>  R_sel <= alpha' (given p_acc > 0)
    u = A_ho * v_ho - c * (1.0 - A_ho)
    s = A_ho.sum(0).astype(int)
    p_hat = A_ho.mean(0)

    eb_ucb_Zp = empirical_bernstein_ucb(Zp.mean(0), Zp.var(0, ddof=1), n_ho, de, range_b=B)
    p_lcb = np.array([clopper_pearson_lower(int(sk), n_ho, de) for sk in s])
    u_lcb = empirical_bernstein_lcb(u.mean(0), u.var(0, ddof=1), n_ho, de, range_b=c + V)

    risk_ok = eb_ucb_Zp <= 0.0              # R_sel <= alpha' certified
    acc_ok = p_lcb >= two_pmin              # p_acc >= 2 pi_min certified
    member = risk_ok & acc_ok               # certified member of Mset(alpha', pi_min)

    out = dict(
        n_ho=n_ho, m=m, alpha_prime=float(alpha_prime), two_pmin=float(two_pmin),
        delta_prime=float(delta_prime), per_event_delta=float(de),
        n_risk_ok=int(risk_ok.sum()), n_acc_ok=int(acc_ok.sum()),
        n_certified_members=int(member.sum()),
        certified=bool(member.any()),
    )
    if member.any():
        out["U_margin_LCB"] = float(u_lcb[member].max())          # population-valid oracle-value LB
        kbest = int(np.argmax(np.where(member, u_lcb, -np.inf)))
        out["witness"] = dict(k=kbest, p_hat=float(p_hat[kbest]), s=int(s[kbest]),
                              p_acc_LCB=float(p_lcb[kbest]), EB_UCB_Zp=float(eb_ucb_Zp[kbest]),
                              U_LCB=float(u_lcb[kbest]))
    else:
        # report the closest-to-feasible pair to explain WHY it failed
        slack = np.maximum(eb_ucb_Zp, 0.0) + np.maximum(two_pmin - p_lcb, 0.0)
        kbest = int(np.argmin(slack))
        out["closest"] = dict(k=kbest, p_hat=float(p_hat[kbest]), s=int(s[kbest]),
                              p_acc_LCB=float(p_lcb[kbest]), EB_UCB_Zp=float(eb_ucb_Zp[kbest]),
                              risk_ok=bool(risk_ok[kbest]), acc_ok=bool(acc_ok[kbest]))
    return out


def make_grid_AcLv(g_split, L_split, tau):
    A = (g_split.reshape(-1, 1) > tau.reshape(1, len(tau))).astype(np.float64)
    L = np.broadcast_to(L_split.reshape(-1, 1), (len(g_split), len(tau))).astype(np.float64)
    v = np.clip(1.0 - L, 0.0, 1.0)
    return L, A, v


# ----------------------------------------------------------------------------
# (A) Synthetic existence surface  (paper case 3), upgraded to held-out certification
# ----------------------------------------------------------------------------
def run_synthetic(delta_prime=0.05):
    alpha, pmin, delta, m = 0.30, 0.20, 0.10, 15
    n_cert = 25000                                   # certificate operating point (defines gamma_r)
    gr = gamma_r_literal(n_cert, pmin, m, delta)
    alpha_prime = alpha - gr
    n_tune, n_ho = 15000, 50000                      # independent grid-build and held-out splits
    rng = np.random.default_rng(0)
    n = n_tune + n_ho
    easy = rng.random(n) < 0.5
    L_raw = np.where(easy, rng.beta(1, 30, n), rng.beta(5, 3, n))
    g = np.clip(1.0 - L_raw + rng.normal(0, 0.05, n), 0.0, 1.0)
    # split: tune (build tau grid) | held-out (certify) -- disjoint, independent
    tau = np.quantile(g[:n_tune], np.linspace(0.5, 0.95, m))
    L_ho, A_ho, v_ho = make_grid_AcLv(g[n_tune:], L_raw[n_tune:], tau)
    cert = certify_mset_nonempty(L_ho, A_ho, v_ho, alpha_prime, 2 * pmin, delta_prime)
    cert.update(surface="synthetic-existence", alpha=alpha, pi_min=pmin, delta=delta,
                m=m, n_cert=n_cert, gamma_r=float(gr), n_tune=n_tune)
    return cert


# ----------------------------------------------------------------------------
# (B) COCO real surface  -- best-effort scan for a gamma_r < alpha config with
#     a held-out-certifiable non-empty Mset.  Honest: report pass/fail per config.
# ----------------------------------------------------------------------------
def coco_arrays():
    d = _find_coco_dir()
    pix = np.clip(np.load(d / "val_mask2former_coco_loss_pixacc.npy").astype(np.float64), 0, 1)
    miou = np.load(d / "val_mask2former_coco_miou.npy").astype(np.float64)
    g = np.load(d / "val_mask2former_coco_g_softmax.npy").astype(np.float64)
    return dict(pixacc=pix, binary=(miou < 0.3).astype(np.float64),
                continuous=np.clip(1 - miou, 0, 1)), g


def run_coco(delta_prime=0.05):
    losses, g = coco_arrays()
    n = len(g); m, delta = 15, 0.10
    n_cal = n // 2                                     # grid-build split
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    cal, ho = perm[:n_cal], perm[n_cal:]
    g_cal, g_ho = g[cal], g[ho]
    tau = np.quantile(g_cal, np.linspace(0.5, 0.95, m))
    configs = [  # (loss, alpha, pi_min); gamma_r computed at n_cert=n_cal
        ("binary", 0.40, 0.20), ("binary", 0.50, 0.30), ("binary", 0.60, 0.30),
        ("pixacc", 0.50, 0.20), ("pixacc", 0.60, 0.30),
        ("continuous", 0.70, 0.30),
    ]
    results = []
    for loss, alpha, pmin in configs:
        gr = gamma_r_literal(n_cal, pmin, m, delta)
        ap = alpha - gr
        L_ho, A_ho, v_ho = make_grid_AcLv(g_ho, losses[loss][ho], tau)
        cert = certify_mset_nonempty(L_ho, A_ho, v_ho, ap, 2 * pmin, delta_prime)
        cert.update(surface=f"COCO/{loss}", alpha=alpha, pi_min=pmin, delta=delta,
                    m=m, n_cert=n_cal, gamma_r=float(gr), gamma_r_lt_alpha=bool(gr < alpha))
        results.append(cert)
    return results


def main():
    np.seterr(all="ignore")
    print("=" * 80)
    print("Held-out high-confidence certification of external margin-oracle non-emptiness (M1/Q1)")
    print("=" * 80)

    syn = run_synthetic()
    print("\n(A) Synthetic existence surface (alpha=0.30, pi_min=0.20, n_cert=25000):")
    print(f"    gamma_r = {syn['gamma_r']:.4f} < alpha  =>  alpha' = alpha - gamma_r = {syn['alpha_prime']:.4f}")
    print(f"    held-out n = {syn['n_ho']}, delta' = {syn['delta_prime']}, per-event delta = {syn['per_event_delta']:.2e}")
    print(f"    risk_ok pairs (R_sel<=alpha' certified): {syn['n_risk_ok']}/{syn['m']}; "
          f"acc_ok pairs (p_acc>=2pi_min certified): {syn['n_acc_ok']}/{syn['m']}")
    if syn["certified"]:
        w = syn["witness"]
        print(f"    >>> Mset NON-EMPTY certified at conf {1-syn['delta_prime']:.2f}: "
              f"{syn['n_certified_members']} member(s); witness k={w['k']} "
              f"(p_acc_LCB={w['p_acc_LCB']:.3f}>={syn['two_pmin']:.2f}, EB_UCB(Z')={w['EB_UCB_Zp']:+.4f}<=0)")
        print(f"    >>> population-valid oracle-value lower bound: U_margin >= U_LCB = {syn['U_margin_LCB']:.3f}")
    else:
        print(f"    >>> NOT certified ({syn['closest']})")

    coco = run_coco()
    print("\n(B) COCO real surface -- best-effort scan (gamma_r computed at n_cert=2500):")
    print(f"    {'config':28s} {'gamma_r':>8s} {'a-gr':>7s} {'risk_ok':>8s} {'acc_ok':>7s}  result")
    any_coco = False
    for r in coco:
        tag = f"{r['surface']} a={r['alpha']} pi={r['pi_min']}"
        if r["certified"]:
            any_coco = True
            res = f"CERTIFIED (U_margin>={r['U_margin_LCB']:.3f})"
        elif not r["gamma_r_lt_alpha"]:
            res = "skip (gamma_r >= alpha)"
        else:
            res = f"empty (k*={r['closest']['k']}: risk_ok={r['closest']['risk_ok']} acc_ok={r['closest']['acc_ok']})"
        print(f"    {tag:28s} {r['gamma_r']:8.3f} {r['alpha']-r['gamma_r']:7.3f} "
              f"{r['n_risk_ok']:8d} {r['n_acc_ok']:7d}  {res}")

    print("\n" + "-" * 80)
    print("SUMMARY:")
    print(f"  Synthetic surface: external oracle non-emptiness {'CERTIFIED (held-out, population-valid)' if syn['certified'] else 'NOT certified'}.")
    print(f"  COCO real surface: {'a config certified' if any_coco else 'no scanned config certifies a non-empty Mset'} "
          f"-- the gamma_r vs 2*pi_min tension (Sec. V-C) is{'' if any_coco else ' confirmed'} on real headline-scale data.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(synthetic=syn, coco=coco), f, indent=2)
    print(f"\n  records -> {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
