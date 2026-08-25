"""Confidence-bound primitives for Algorithm 1.

Implements:
- Maurer-Pontil empirical-Bernstein (two-sided form), per Theorem 4 of Maurer & Pontil 2009.
- Clopper-Pearson binomial lower confidence bound via scipy.stats.beta.ppf.
- Multiplicative Chernoff lower-tail / upper-tail threshold computations
  (used inside the CP Sub-Lemma derivation and Lemma 1's variance bridge).

All routines are vectorized over the grid index when applicable.

References
----------
- Maurer, A. & Pontil, M. (2009). Empirical Bernstein bounds and sample
  variance penalization. COLT 2009.
- Clopper, C. J. & Pearson, E. S. (1934). The use of confidence or fiducial
  limits illustrated in the case of the binomial. Biometrika, 26(4).
- Mitzenmacher, M. & Upfal, E. (2017). Probability and Computing, §4.4 (Chernoff).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as scipy_beta


def maurer_pontil_two_sided_radius(
    sigma_sq_hat: np.ndarray,
    n: int,
    delta_prime: float,
    range_b: float,
) -> np.ndarray:
    """Two-sided Maurer-Pontil empirical-Bernstein radius at confidence 1 - delta_prime.

    The two-sided form is obtained by union over both tails of the one-sided
    MP bound, each at level delta_prime / 2. The resulting radius is:

        rad = sqrt(2 * sigma_sq_hat * log(4 / delta_prime) / n)
            + 7 * range_b * log(4 / delta_prime) / (3 * (n - 1))

    Parameters
    ----------
    sigma_sq_hat : np.ndarray
        Bessel-corrected sample variance, shape (m,) or scalar.
    n : int
        Sample size (>= 2).
    delta_prime : float
        Two-sided confidence level (failure probability for both tails combined).
    range_b : float
        Range b - a of the bounded random variable.

    Returns
    -------
    np.ndarray or float
        Two-sided radius, same shape as sigma_sq_hat.

    Notes
    -----
    The closed-form constant 7/3 follows from Maurer & Pontil 2009 Theorem 4
    after the Bernstein-style derivation. The variance term is sample-variance
    based (computable from data alone); the residual range term controls the
    higher-order correction.
    """
    if n < 2:
        raise ValueError(f"Maurer-Pontil requires n >= 2 (got n={n}).")
    if not (0.0 < delta_prime < 1.0):
        raise ValueError(f"delta_prime must be in (0, 1) (got {delta_prime}).")
    if range_b <= 0:
        raise ValueError(f"range_b must be positive (got {range_b}).")

    sigma_sq_hat = np.asarray(sigma_sq_hat, dtype=np.float64)
    if np.any(sigma_sq_hat < -1e-12):
        raise ValueError("Sample variance must be non-negative.")
    sigma_sq_hat = np.clip(sigma_sq_hat, 0.0, None)

    log_term = np.log(4.0 / delta_prime)  # log(4/δ') = log(2/(δ'/2))
    variance_part = np.sqrt(2.0 * sigma_sq_hat * log_term / n)
    range_part = 7.0 * range_b * log_term / (3.0 * (n - 1))
    return variance_part + range_part


def clopper_pearson_lower(
    s: np.ndarray,
    n: int,
    delta_prime: float,
) -> np.ndarray:
    """One-sided Clopper-Pearson lower confidence bound on Bin(n, p).

    Definition: p_LCB(s; n, δ') = sup { q ∈ [0, 1] : P_q[Bin(n, q) >= s] <= δ' }
    Equivalent closed form (for s >= 1): p_LCB = Beta_{δ'}^{-1}(s, n - s + 1).
    For s = 0: p_LCB = 0.

    Parameters
    ----------
    s : np.ndarray or int
        Observed acceptance count(s), shape (m,) or scalar; integer values in [0, n].
    n : int
        Sample size (>= 1).
    delta_prime : float
        Failure probability (one-sided).

    Returns
    -------
    np.ndarray or float
        Clopper-Pearson lower bound, same shape as s.

    Notes
    -----
    This is the *exact* binomial inversion (Clopper-Pearson 1934). It is
    conservative (the LCB strictly under-covers vs the discrete binomial
    coverage probability) but always valid: P(p_LCB <= p_acc) >= 1 - delta_prime.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 (got n={n}).")
    if not (0.0 < delta_prime < 1.0):
        raise ValueError(f"delta_prime must be in (0, 1) (got {delta_prime}).")

    s = np.asarray(s, dtype=np.int64)
    if np.any(s < 0) or np.any(s > n):
        raise ValueError(f"s must be in [0, n] for n={n}.")

    out = np.where(
        s == 0,
        0.0,
        scipy_beta.ppf(delta_prime, s, np.maximum(n - s + 1, 1)),
    )
    # scipy may return nan for the s==0 branch internally; mask it.
    out = np.where(np.isnan(out), 0.0, out)
    return out


def chernoff_lower_tail_threshold(
    p_acc_min: float,
    n: int,
    delta_prime: float,
    epsilon: float = 0.25,
) -> float:
    """Return the minimum n s.t. multiplicative-Chernoff lower-tail at level epsilon
    holds with probability >= 1 - delta_prime under any p >= p_acc_min.

    Used as a sanity helper for the sample-size condition (★) of Algorithm 1.

    The bound is: P(s <= (1 - epsilon) * n * p) <= exp(-epsilon^2 * n * p / 2).
    Setting RHS <= delta_prime gives n * p >= 2 * log(1/delta_prime) / epsilon^2.
    For p >= p_acc_min: n >= 2 * log(1/delta_prime) / (epsilon^2 * p_acc_min).

    For Algorithm 1's sample-size condition, epsilon = 1/4 gives the canonical
    n_0 = 32 * log(1/delta_prime) / p_acc_min from the Sub-Lemma proof.
    """
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0, 1).")
    return float(2.0 * np.log(1.0 / delta_prime) / (epsilon**2 * p_acc_min))


def chernoff_upper_tail_threshold(
    p_acc_min: float,
    n: int,
    delta_prime: float,
    epsilon: float = 0.5,
) -> float:
    """Symmetric multiplicative-Chernoff upper-tail threshold helper.

    The bound is: P(s >= (1 + epsilon) * n * p) <= exp(-epsilon^2 * n * p / (2 + epsilon)).
    Used in Lemma 1's variance-bridge step to argue p_hat_acc <= (3/2) * p_acc.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    return float(
        (2.0 + epsilon) * np.log(1.0 / delta_prime) / (epsilon**2 * p_acc_min)
    )


def empirical_bernstein_ucb(
    mean_hat: np.ndarray,
    sigma_sq_hat: np.ndarray,
    n: int,
    delta_prime: float,
    range_b: float,
) -> np.ndarray:
    """One-sided UCB from two-sided Maurer-Pontil radius.

    Returns mean_hat + radius, where radius is the two-sided MP at level
    delta_prime. By the two-sided guarantee, with probability >= 1 - delta_prime,
    both E[X] <= mean_hat + radius AND mean_hat <= E[X] + radius hold
    simultaneously, so this is a valid UCB.
    """
    return mean_hat + maurer_pontil_two_sided_radius(
        sigma_sq_hat, n, delta_prime, range_b
    )


def empirical_bernstein_lcb(
    mean_hat: np.ndarray,
    sigma_sq_hat: np.ndarray,
    n: int,
    delta_prime: float,
    range_b: float,
) -> np.ndarray:
    """One-sided LCB from two-sided Maurer-Pontil radius."""
    return mean_hat - maurer_pontil_two_sided_radius(
        sigma_sq_hat, n, delta_prime, range_b
    )


def sample_size_condition(
    m: int, delta: float, pi_min: float
) -> int:
    """Theorem 1 sample-size condition: n_cert >= 32 * log(32m/delta) / pi_min."""
    if pi_min <= 0 or pi_min >= 1:
        raise ValueError("pi_min must be in (0, 1).")
    if m < 1:
        raise ValueError("m must be >= 1.")
    return int(np.ceil(32.0 * np.log(32.0 * m / delta) / pi_min))
