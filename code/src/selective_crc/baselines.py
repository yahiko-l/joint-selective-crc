"""Per-pair comparators for Algorithm 1.

- Baseline A (range-only Hoeffding): Hoeffding radius on E[Z] = E[A(L - α)],
  no variance adaptation, divided by π_min, by the empirical acceptance, or by
  a Clopper-Pearson lower bound on p_acc.
- Baseline B (accepted-sample Bernstein): empirical-Bernstein radius on the
  losses of the s = Σ A_i accepted samples, Bonferroni over the m pairs. Its
  per-pair rate matches Algorithm 1's, but it provides no joint certificate.
- WSR betting and fixed-bet product e-value UCBs on E[Z].
- Simplified ports of SCRC-T and SCoRE.
"""

from __future__ import annotations

import numpy as np

from .bounds import clopper_pearson_lower


def baseline_a_range_hoeffding_pacc(
    L: np.ndarray,
    A: np.ndarray,
    alpha: float,
    delta: float,
    B: float,
) -> np.ndarray:
    """Baseline A with the empirical acceptance p_hat = s/n as denominator.

    Same range-only Hoeffding radius on E[Z = A(L-α)] as
    `baseline_a_range_hoeffding`, divided per pair by p_hat instead of π_min.
    Without a lower bound on p_acc this is not a valid (1-δ) UCB on R_sel;
    informational comparator only.

    Returns the per-pair R_sel radius (inf where p_hat = 0).
    """
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    n_cert, m = L.shape
    delta_per = delta / m
    hoeffding_radius_Z = B * np.sqrt(np.log(2.0 / delta_per) / (2.0 * n_cert))
    p_hat = A.mean(axis=0)  # (m,)
    margin = np.where(p_hat > 0, hoeffding_radius_Z / np.maximum(p_hat, 1e-12), np.inf)
    return margin


def baseline_a_range_hoeffding_plcb(
    L: np.ndarray,
    A: np.ndarray,
    alpha: float,
    delta: float,
    B: float,
) -> np.ndarray:
    """Baseline A with a Clopper-Pearson lower bound on p_acc as denominator.

    Range-only Hoeffding radius on E[Z] at δ/(2m), divided by the
    Clopper-Pearson LCB on p_acc at δ/(2m). Added to the empirical selective
    risk it gives a valid, conservative per-pair UCB on R_sel: since
    p_LCB <= s/n, the radius dominates the Hoeffding radius of the s accepted
    losses.

    Returns the per-pair R_sel radius (NaN where p_LCB = 0).
    """
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    n_cert, m = L.shape
    delta_per_z = delta / (2.0 * m)
    delta_per_p = delta / (2.0 * m)
    hoeffding_radius_Z = B * np.sqrt(np.log(2.0 / delta_per_z) / (2.0 * n_cert))
    s = A.sum(axis=0).astype(int)
    p_lcb = np.array([clopper_pearson_lower(int(s_k), n_cert, delta_per_p) for s_k in s])
    margin = np.where(p_lcb > 0, hoeffding_radius_Z / np.maximum(p_lcb, 1e-12), np.nan)
    return margin


def baseline_a_range_hoeffding(
    L: np.ndarray,
    A: np.ndarray,
    alpha: float,
    pi_min: float,
    delta: float,
    B: float,
) -> np.ndarray:
    """Baseline A: range-only Hoeffding radius on E[Z], divided by π_min.

    radius = B * sqrt(log(2m/δ) / (2n)) / π_min, the same for every pair.
    Added to the empirical selective risk it gives the π_min-saturated
    Hoeffding-CRC selective bound A(π_min).

    Parameters
    ----------
    L, A : np.ndarray, shape (n_cert, m)
        Per-sample losses and acceptances.
    alpha : float
        Target risk (not used by the radius).
    pi_min : float
        Acceptance floor used as the denominator.
    delta : float
        Failure probability (Bonferroni over the grid at δ/m).
    B : float
        Loss range upper bound.

    Returns
    -------
    np.ndarray, shape (m,)
        Per-pair R_sel radius.
    """
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    n_cert, m = L.shape

    delta_per = delta / m  # Bonferroni
    # Z = A(L - α) lies in [-α, B - α], an interval of width B.
    hoeffding_radius_Z = B * np.sqrt(np.log(2.0 / delta_per) / (2.0 * n_cert))

    margin = hoeffding_radius_Z / pi_min
    return np.full(m, margin, dtype=np.float64)


def baseline_b_full_certificate(
    L: np.ndarray,
    A: np.ndarray,
    alpha: float,
    pi_min: float,
    delta: float,
    B: float,
) -> dict:
    """Full Baseline B with feasibility mask (R_sel UCB + p_acc LCB couple).

    Not called by any experiment; the per-pair width comparisons use
    `baseline_b_accepted_bernstein`.

    For each (λ, τ):
      - δ split: δ/(3m) for accepted-sample Bernstein on L, δ/(3m) for
        Bernstein on E[A]
      - Bernstein LCB on p_acc at δ/(3m): p_LCB_bern = p_hat - sqrt(p_hat·log/(2n)) (Hoeffding-Bernoulli style, simple LCB)
      - Per-pair (R_sel UCB) = mean(L_accepted) + Bernstein_radius_on_L
      - Feasible iff (R_sel UCB ≤ α) ∧ (p_LCB_bern ≥ π_min)

    Returns a dict with per-pair R_sel UCB, p_LCB_bern, feasibility mask,
    and the union of these as the full Baseline B "certified set".
    """
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    n, m = L.shape
    delta_third = delta / (3.0 * m)

    R_sel_ucb = np.full(m, np.nan)
    p_acc_lcb = np.full(m, np.nan)
    feasible = np.zeros(m, dtype=bool)
    for k in range(m):
        accepted_mask = A[:, k] > 0.5
        s = int(accepted_mask.sum())
        # Bernstein-LCB on E[A] (m/3 share)
        p_hat = s / n
        # Simple Bernoulli Bernstein lower one-sided bound at δ/(3m):
        # use t = sqrt(2·p̂·(1-p̂)·log(1/δ_third)/n) + log(1/δ_third)/(3·n)
        if n > 1:
            log_t = np.log(1.0 / delta_third)
            bern_p_radius = np.sqrt(2.0 * p_hat * (1.0 - p_hat) * log_t / n) + log_t / (3.0 * n)
            p_acc_lcb[k] = max(0.0, p_hat - bern_p_radius)
        if s < 2:
            continue
        L_accepted = L[accepted_mask, k]
        mean_L = float(L_accepted.mean())
        var_L = float(L_accepted.var(ddof=1))
        bern_r = (
            np.sqrt(2.0 * var_L * np.log(3.0 / delta_third) / s)
            + 7.0 * B * np.log(3.0 / delta_third) / (3.0 * (s - 1))
        )
        R_sel_ucb[k] = mean_L + bern_r
        feasible[k] = (R_sel_ucb[k] <= alpha) and (p_acc_lcb[k] >= pi_min)

    return {
        "R_sel_ucb": R_sel_ucb,
        "p_acc_lcb_bern": p_acc_lcb,
        "feasible_mask": feasible,
        "n_feasible": int(feasible.sum()),
    }


def baseline_b_accepted_bernstein(
    L: np.ndarray,
    A: np.ndarray,
    alpha: float,
    delta: float,
    B: float,
) -> np.ndarray:
    """Baseline B: empirical-Bernstein radius on the accepted-sample losses.

    For each (λ, τ), with s = Σ A accepted samples and losses L_acc,
        radius = sqrt(2 * var(L_acc) * log(3m/δ) / s) + 7 * B * log(3m/δ) / (3 * (s - 1)),
    a Bonferroni allocation over the m pairs. Added to mean(L_acc), the
    empirical selective risk, it is a per-pair UCB on E[L | A = 1]; there is
    no joint certificate.

    Returns
    -------
    np.ndarray, shape (m,)
        Per-pair radius around the empirical selective risk; NaN where s < 2.
    """
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    n_cert, m = L.shape
    delta_per = delta / m

    widths = np.empty(m, dtype=np.float64)
    for k in range(m):
        accepted_mask = A[:, k] > 0.5
        s = int(accepted_mask.sum())
        if s < 2:
            widths[k] = np.nan
            continue
        L_accepted = L[accepted_mask, k]
        mean_L = L_accepted.mean()
        var_L = L_accepted.var(ddof=1)
        bernstein_radius = (
            np.sqrt(2.0 * var_L * np.log(3.0 / delta_per) / s)
            + 7.0 * B * np.log(3.0 / delta_per) / (3.0 * (s - 1))
        )
        widths[k] = bernstein_radius
    return widths


# =============================================================================
# Betting and e-value UCBs on E[Z].
# =============================================================================


def wsr_betting_ucb(
    Z: np.ndarray,
    delta_prime: float,
    range_b: float,
    c_clip: float = 0.5,
) -> np.ndarray:
    """Betting UCB on E[Z] after Waudby-Smith & Ramdas (2024, arXiv 2010.09686).

    Per-pair one-sided UCB on `E[Z(λ,τ)]` at level `1 - delta_prime`. Z in
    [-range_b, range_b] is shifted to `Y = Z + range_b` in [0, 2·range_b]. For a
    candidate mean m the capital `K_n(m) = ∏_t (1 - λ_t · (Y_t - m))`, with the
    predictable bets `λ_t = min(c_clip / (2·range_b),
    sqrt(2 · log(1/δ') / (n · σ̂²_{t-1})))`, is a non-negative supermartingale
    under E[Y] ≥ m, so by Ville's inequality the UCB on E[Y] is the smallest m
    with `K_n(m) > 1/δ'` (found by bisection); it is shifted back by range_b.
    A simplified port; the `confseq` package implements the full method.

    Parameters
    ----------
    Z : np.ndarray, shape (n, m)
        Per-sample contributions across n samples and m grid points.
    delta_prime : float
        One-sided level per pair; the caller splits δ over the grid.
    range_b : float
        Range bound: assumes `Z[i, k] ∈ [-range_b, range_b]`.
    c_clip : float
        Bet cap in (0, 1) (default 0.5).

    Returns
    -------
    np.ndarray, shape (m,)
        Per-pair UCB on `E[Z(λ_k, τ_k)]`.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n, m = Z.shape
    if range_b <= 0:
        raise ValueError(f"range_b must be > 0 (got {range_b}).")
    if not (0.0 < delta_prime < 1.0):
        raise ValueError(f"delta_prime must be in (0, 1).")
    if c_clip <= 0 or c_clip >= 1:
        raise ValueError("c_clip must be in (0, 1).")

    # Shift to non-negative range [0, 2·range_b]
    Y = Z + range_b  # shape (n, m); Y ∈ [0, 2·range_b]
    range_Y = 2.0 * range_b
    log_target = -np.log(delta_prime)  # = log(1/δ')

    ucb = np.empty(m, dtype=np.float64)
    for k in range(m):
        y = Y[:, k]
        # Predictable mean/variance: at time t use stats from y_1..y_{t-1}
        cum_y = np.cumsum(y)
        cum_y2 = np.cumsum(y * y)
        t_arr = np.arange(1, n + 1, dtype=np.float64)
        # Mean prior at t=1: range center; otherwise (Σ_{s<t} y_s) / (t-1)
        denom = np.maximum(t_arr - 1.0, 1.0)
        mu_prev = np.concatenate(([range_Y / 2.0], cum_y[:-1] / denom[1:]))
        # Variance prior at t=1: range_Y²/4, the largest variance on [0, range_Y]; otherwise
        # ( Σ_{s<t} y_s² / (t-1) ) − μ_prev²
        var_prev = np.concatenate((
            [range_Y ** 2 / 4.0],
            np.maximum(cum_y2[:-1] / denom[1:] - mu_prev[1:] ** 2, 1e-6 * range_Y ** 2),
        ))
        # Predictable bet schedule
        lam_unclipped = np.sqrt(2.0 * np.log(1.0 / delta_prime) / (n * var_prev))
        lam_cap = c_clip / range_Y
        lam = np.minimum(lam_unclipped, lam_cap)

        # Binary search for UCB on E[Y]
        def log_capital_at(m_test):
            # log K_n(m_test) = Σ_t log(1 - λ_t · (y_t - m_test))
            # = Σ_t log(1 + λ_t · (m_test - y_t))
            terms = 1.0 + lam * (m_test - y)
            # Clip to avoid log(<=0)
            terms = np.maximum(terms, 1e-12)
            return np.sum(np.log(terms))

        # K is increasing in m_test: bisect on [0, range_Y] for the smallest
        # rejected m_test (K > 1/δ'); hi stays at range_Y if none is rejected.
        lo, hi = 0.0, range_Y
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if log_capital_at(mid) > log_target:
                hi = mid  # mid rejects; UCB ≤ mid
            else:
                lo = mid
        # UCB on E[Y] is hi; shift back to E[Z]:
        ucb[k] = hi - range_b

    return ucb


def ebh_product_evalue_ucb(
    Z: np.ndarray,
    delta: float,
    range_b: float,
) -> np.ndarray:
    """Fixed-bet product e-value UCB on E[Z] with a Bonferroni grid correction.

    For each pair, `e_n(m_test) = ∏_t (1 + η · (m_test - Z_t))` with the fixed
    bet `η = 1/(2·range_b)` is a non-negative supermartingale under
    E[Z] ≥ m_test. The UCB is `inf{m_test : log e_n(m_test) > log(m/δ)}`, i.e.
    per-pair level δ/m, a Bonferroni stand-in for the e-BH thresholds of
    Wang & Ramdas (2022). Unlike `wsr_betting_ucb`, the bet does not adapt to
    the variance.

    Parameters
    ----------
    Z : np.ndarray, shape (n, m)
    delta : float
        Total failure budget over the grid; the per-pair level is δ/m.
    range_b : float
        Range bound: assumes `Z[i, k] ∈ [-range_b, range_b]`.

    Returns
    -------
    np.ndarray, shape (m,)
    """
    Z = np.asarray(Z, dtype=np.float64)
    n, m = Z.shape
    if range_b <= 0:
        raise ValueError(f"range_b must be > 0.")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1).")

    delta_per = delta / m  # Bonferroni grid (conservative proxy for e-BH)
    eta = 1.0 / (2.0 * range_b)
    log_target = np.log(1.0 / delta_per)

    ucb = np.empty(m, dtype=np.float64)
    for k in range(m):
        z = Z[:, k]

        def log_evalue_at(m_test):
            terms = 1.0 + eta * (m_test - z)
            terms = np.maximum(terms, 1e-12)
            return np.sum(np.log(terms))

        # Binary search; e_n(m_test) is monotone increasing in m_test
        lo, hi = -range_b, range_b
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if log_evalue_at(mid) > log_target:
                hi = mid
            else:
                lo = mid
        ucb[k] = hi

    return ucb


# =============================================================================
# Simplified ports of SCRC-T and SCoRE for the per-pair comparison. They keep
# the structural choice of each method rather than reproduce the published
# algorithm verbatim; each docstring gives the mapping.
# =============================================================================


def scrct_quantile_ucb(
    L: np.ndarray,
    A: np.ndarray,
    alpha: float,
    pi_min: float,
    delta: float,
    B: float,
) -> np.ndarray:
    """SCRC-T adapted (Xu, Guo, Wei 2025; arXiv 2512.12844) — simplified port.

    SCRC-T's key idea: joint (λ_1, λ_2) selective CRC via quantile-based threshold
    on (L − α) restricted to accepted samples. Their assumption is loss non-
    increasing in λ_2; our setting is bounded but possibly non-monotone, so this
    is a SIMPLIFIED adaptation rather than exact reproduction.

    Simplified adapted form:
    For each (λ, τ) pair k, compute the (1 − δ/m)-quantile of (L_i − α) restricted
    to accepted samples i: q_k = Quantile_{1−δ/m}({L_i(λ_k) − α : A_i(λ_k, τ_k) = 1}).
    The per-pair R_sel UCB margin is `q_k` (positive → infeasible; ≤ 0 → certified).

    This captures SCRC-T's structural choice: AVOID variance adaptation, use
    quantile-based threshold instead of moment inequality. Pessimistic on
    accepted-sample size (uses 1-δ/m grid Bonferroni; no CP relative-error inversion).

    Returns
    -------
    np.ndarray, shape (m,)
        Per-pair R_sel margin (UCB on R_sel - α). Caller can compare to 0 for
        feasibility or to other widths for tightness comparison.
    """
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    n_cert, m = L.shape

    delta_per = delta / m
    quantile_level = 1.0 - delta_per
    margins = np.empty(m, dtype=np.float64)
    for k in range(m):
        accepted_mask = A[:, k] > 0.5
        s = int(accepted_mask.sum())
        if s < 5:  # too few accepted samples for stable quantile
            margins[k] = B  # max possible margin (effectively infeasible)
            continue
        L_accepted = L[accepted_mask, k]
        # 1-δ/m quantile of (L - α) on accepted samples
        margin = float(np.quantile(L_accepted - alpha, quantile_level, method="higher"))
        margins[k] = margin
    return margins


def score_evalue_ucb(
    L: np.ndarray,
    A: np.ndarray,
    alpha: float,
    delta: float,
    B: float,
) -> np.ndarray:
    """SCoRE adapted (Bai & Jin 2026; arXiv 2603.24704) — simplified port.

    SCoRE's key idea: e-value framework `E[L · E] ≤ 1` (product form, AVOIDS
    the ratio reformulation E[A(L-α)] ≤ 0 that OURS uses). Single trust threshold
    via product e-value; multiplicity correction via e-BH (Wang & Ramdas 2022).

    Simplified adapted form:
    For each pair k, construct a per-sample e-value contribution:
        e_i(k) := exp(η · A_i(k) · (alpha - L_i(k))) where η chosen to maximize
        expected growth under H_1: R_sel < α.

    Per-pair product e-value: E_k(n) := ∏_i e_i(k)
    Per-pair test: reject H_0: R_sel(k) ≥ α if E_k(n) > m/δ (e-BH Bonferroni proxy).

    Per-pair R_sel UCB: invert by finding the largest alpha_test such that
        ∏_i exp(η · A_i · (alpha_test - L_i)) ≤ m/δ
    which simplifies to
        η · Σ_i A_i · (alpha_test - L_i) ≤ log(m/δ)
        alpha_test ≤ (log(m/δ) / η + Σ_i A_i · L_i) / Σ_i A_i  (if Σ A_i > 0)

    For a fixed η = 1/B (default), this gives a closed-form UCB on R_sel:
        R_sel_UCB(k) = (B · log(m/δ) + Σ_i A_i L_i) / Σ_i A_i

    UCB margin from α: R_sel_UCB - α. Caller compares to 0.
    """
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    n_cert, m = L.shape

    eta = 1.0 / B  # standard η for bounded L ∈ [0, B]
    log_target = np.log(m / delta)
    margins = np.empty(m, dtype=np.float64)
    for k in range(m):
        s = float(A[:, k].sum())
        if s < 1:
            margins[k] = B  # infeasible (no accepted samples to compute e-value)
            continue
        sum_AL = float((A[:, k] * L[:, k]).sum())
        # Closed-form R_sel UCB from product e-value inversion
        r_sel_ucb = (log_target / eta + sum_AL) / s
        margins[k] = r_sel_ucb - alpha
    return margins
