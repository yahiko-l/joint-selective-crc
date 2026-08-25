"""Baselines for comparison with Algorithm 1.

Per EXPERIMENT_PLAN.md §B2 / B4, we compare against:

- **Baseline A (Range-only Hoeffding moment bound)**: a UCB on E[Z] = E[A(L-α)]
  using Hoeffding's inequality (no variance adaptation). The implied R_sel UCB
  margin is `(B/π_min) * sqrt(log(2m/δ)/(2n))`. Used to demonstrate the
  variance-adaptive advantage of our method at low π_acc.

- **Baseline B (Accepted-sample Bernstein + Bonferroni)**: applies Bernstein on
  the loss L restricted to accepted samples (random subset of size s = Σ A_i),
  union-bounded over m candidates at δ/(3m), plus a separate Bernstein LCB on
  E[A] at δ/(3m). Per-pair rate matches ours `O(B·√(log m/(n·p_acc)))` but
  does NOT provide the joint certificate (requires manual coupling of two
  Bernstein bounds + no canonical utility selection rule).

These are intentionally simplified for the per-pair UCB comparison in
experiments B2/B4. `baseline_b_accepted_bernstein_radius` returns only the
per-pair *radius* (margin width) used in PC2 width comparison; the full
Baseline B feasibility coupling is in `baseline_b_full_certificate`.
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
    """Baseline-A / p_acc (stronger comparator).

    Same range-only Hoeffding radius on E[Z = A(L-α)] as `baseline_a_range_hoeffding`,
    but post-divided by the *empirical acceptance* p_hat (= s/n) per pair instead
    of the worst-case π_min lower bound. This is what a knowledgeable but slightly
    optimistic reviewer would compute. NOT a valid (1−δ) UCB on R_sel (no LCB on
    p_acc) — informational comparator only.

    Returns the per-pair R_sel margin radius.
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
    """Baseline-A / p_LCB (valid stronger comparator).

    Same range-only Hoeffding radius on E[Z] divided by a Clopper-Pearson lower
    bound on p_acc at δ/(2m) (using half the budget for the p_acc LCB and half
    for the Z UCB, in the spirit of Baseline-B's δ-split). This IS a valid
    (1−δ) UCB on R_sel per pair — strictly tighter than the worst-case π_min
    Baseline-A but still range-only on the numerator.

    Returns the per-pair R_sel margin radius (NaN where p_LCB ≤ 0 or s = 0).
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
    """Baseline A: range-only Hoeffding UCB on E[Z], divided by π_min.

    UCB on E[Z = A(L-α)]: `mean(Z) + B * sqrt(log(2m/δ)/(2n))`.
    Implied UCB on R_sel = E[Z]/p_acc, divided by lower bound π_min for p_acc:
      UCB_R = UCB_Z / π_min + α (after rearrangement)

    For per-pair WIDTH comparison, we return the implied R_sel UCB margin:
      width = (UCB on E[Z]) / π_min

    Parameters
    ----------
    L, A : np.ndarray, shape (n_cert, m)
        Per-sample losses and acceptances.
    alpha : float
        Target risk (used to center Z; appears in the UCB as a shift).
    pi_min : float
        Acceptance lower bound used for the post-division.
    delta : float
        Failure probability (m-grid Bonferroni at δ/m).
    B : float
        Loss range upper bound.

    Returns
    -------
    np.ndarray, shape (m,)
        Per-pair R_sel UCB *margin* (i.e., distance from α; this is the
        "tightness" metric used in PC2 comparison).
    """
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    n_cert, m = L.shape

    delta_per = delta / m  # Bonferroni
    # Range-only Hoeffding: range of Z = A(L-α) is max(α, B-α) ≤ B; use B as range bound.
    hoeffding_radius_Z = B * np.sqrt(np.log(2.0 / delta_per) / (2.0 * n_cert))

    # Implied UCB on R_sel - α: (mean_Z + radius_Z) / π_min
    # (Plus α if comparing absolute level; we return the *margin* from α.)
    margin = hoeffding_radius_Z / pi_min  # divide by lower bound for conservative UCB
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

    NOTE — NOT CALLED BY ANY EXPERIMENT: this function captures the conceptual
    "full Baseline B" coupling described in EXPERIMENT_PLAN.md §B2 but is not
    invoked by `experiments/`. Per-pair width comparison uses
    `baseline_b_accepted_bernstein` instead; the joint-certificate question
    that this function would address is answered structurally (Baseline B has
    no canonical joint certificate without the kind of margin oracle our
    Algorithm 1 provides). Retained as a reference implementation.

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
    """Baseline B: Bernstein on L restricted to accepted samples, with
    Bonferroni at δ/(3m) per pair. No joint certificate.

    For each (λ, τ):
      s = Σ A
      L_accepted = {L_i : A_i = 1}
      Bernstein UCB on E[L | A=1] from s i.i.d. samples (effective sample size):
        mean(L_accepted) + sqrt(2 * sample_var(L_accepted) * log(3/δ_per) / s)
                        + 7 * B * log(3/δ_per) / (3 * (s - 1))   [if s >= 2]
      (Returns NaN if s < 2.)

    Returns
    -------
    np.ndarray, shape (m,)
        Per-pair R_sel UCB *margin* (distance from α). NaN where s < 2.

    Notes
    -----
    This is the *strongest* simple per-pair baseline, matching our leading rate.
    Our structural delta is the joint certificate, NOT per-pair tightness.
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
        widths[k] = bernstein_radius  # margin from mean_L, which is the empirical R_sel
        # (Caller can add mean_L to recover absolute UCB; we return the radius for
        # comparison with baseline_a_range_hoeffding which also returns a radius.)
    return widths


# =============================================================================
# Beast-mode additions (Phase 9 Sub-order D, 2026-05-26):
# WSR betting confidence sequence + e-BH product-e-value baselines.
# =============================================================================


def wsr_betting_ucb(
    Z: np.ndarray,
    delta_prime: float,
    range_b: float,
    c_clip: float = 0.5,
) -> np.ndarray:
    """WSR (Waudby-Smith & Ramdas 2024) betting confidence sequence UCB on E[Z].

    Per-pair one-sided UCB on `E[Z(λ,τ)]` at confidence level `1 - delta_prime`,
    using the predictable-mixture betting CS form (PrPl-MS, Algorithm 1 of
    arXiv 2010.09686 / Theorem 4). For `Z ∈ [-range_b, +range_b]`, we shift
    `Y = Z + range_b` to enforce `Y ∈ [0, 2·range_b]`, run WSR on `Y`, and
    return `UCB(E[Z]) = UCB(E[Y]) - range_b`.

    Algorithm:
    - Predictable bet schedule: `λ_t = min(c_clip / (2·range_b),
       sqrt(2 · log(1/δ') / (n · σ̂²_{t-1})))` (capped so capital stays positive).
    - Capital process for null H_0(m): E[Y] ≥ m:
       `K_n(m) = ∏_{t=1}^n (1 - λ_t · (Y_t - m))`.
    - Under H_0(m), K_n is a non-negative supermartingale starting at 1
      (Waudby-Smith & Ramdas 2024 Lemma 3.1). By Ville's inequality,
       `P(K_n > 1/δ' | H_0(m)) ≤ δ'`.
    - UCB: smallest m such that `K_n(m) > 1/δ'` (binary search).

    Parameters
    ----------
    Z : np.ndarray, shape (n, m)
        Per-sample contributions across n samples and m grid points.
    delta_prime : float
        One-sided confidence level (per pair; caller does Bonferroni externally).
    range_b : float
        Range bound: assumes `Z[i, k] ∈ [-range_b, range_b]`.
    c_clip : float
        Clipping constant for predictable bet (default 0.5; controls bet
        aggressiveness).

    Returns
    -------
    np.ndarray, shape (m,)
        Per-pair UCB on `E[Z(λ_k, τ_k)]`.

    Notes
    -----
    This is a simplified port of WSR PrPl-MS. For the canonical implementation
    with all bells and whistles (anytime-valid CS, optimal hyperparameters),
    use the `confseq` Python package (Howard et al. 2021).
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
        # Variance prior at t=1: range²/4 (var of uniform[0, range_Y]); otherwise
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

        # K is monotone increasing in m_test (each factor is +λ_t · (m_test - y_t))
        # so binary search:
        # - lo: K(lo) ≤ 1/δ' (don't reject H_0: E[Y] ≥ lo) → lo bound
        # - hi: K(hi) > 1/δ' (reject) → hi bound
        # Bracket: search lo at empirical mean (likely K < 1/δ'); hi at range_Y
        lo, hi = 0.0, range_Y
        # Ensure hi is high enough
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
    """e-BH (SCoRE-style) baseline: per-pair UCB on E[Z] via product e-values + Bonferroni grid correction.

    This is a simplified port of the SCoRE (Bai & Jin 2026, arXiv 2603.24704) e-value
    framework, combined with e-BH (Wang & Ramdas 2022, arXiv 2009.02824) for grid
    multiplicity correction. Specifically, for each (λ, τ) pair we use a fixed-bet
    product e-value test for `H_0: E[Z(λ,τ)] = m_test`:

    `e_n(m_test) = ∏_{t=1}^n (1 + η · (m_test - Z_t))` where η is chosen to maximize
    expected growth under H_1: E[Z] ≪ m_test. For bounded |Z| ≤ B with H_0
    expecting E[Z] = m_test, the optimal η is approximately `η* ≈ 1 / (range_b + |m_test|)`
    (clipped to avoid negativity). We use the SIMPLIFIED form: fixed `η = 1/(2·range_b)`
    across all pairs and steps (no variance adaptation, intentionally simpler than WSR).

    Per-pair UCB at level `δ' = δ / m` (Bonferroni grid; stricter than e-BH's
    sorted thresholds but cleanly comparable):
    UCB = inf{m_test : log e_n(m_test) > log(1/δ')}.

    Differences from WSR:
    - WSR: predictable adaptive bet `λ_t` based on running variance estimates.
    - e-BH: fixed bet `η`; pure product e-value structure (no variance bridging).
    - WSR uses δ' directly per pair (called externally with δ' = δ/(16m) per Algorithm 1).
    - e-BH applies Bonferroni-grid δ' = δ/m as a conservative proxy for e-BH sorted thresholds.

    Parameters
    ----------
    Z : np.ndarray, shape (n, m)
    delta : float
        TOTAL grid failure budget (per-pair level becomes δ/m).
    range_b : float

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
# Phase 9 Sub-order E: Prior-work simplified ports for empirical head-to-head.
# These are SIMPLIFIED faithful-spirit ports, not exact reproductions; they
# capture the structural difference vs OURS rather than the published algorithm
# verbatim. See per-function docstrings for the exact mapping to the cited
# paper's algorithm.
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
