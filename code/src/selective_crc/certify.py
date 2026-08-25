"""Algorithm 1 of the paper: certified-utility selective risk control.

Inputs to certify_grid:
- L: array (n_cert, m) of bounded losses, L[i, k] in [0, B]
- A: array (n_cert, m) of {0, 1} acceptance indicators
- v: array (n_cert, m) of bounded deployment values, v[i, k] in [0, V]
- alpha, pi_min, delta, c, V, B: scalars
- (Optional) Lambda, T arrays for index decoding (each (λ, τ) → grid index k)

Output: AlgorithmResult dataclass containing (lambda_hat, tau_hat) or INFEASIBLE
        plus full per-pair diagnostics (EB, p_LCB, U_LCB, In_G_hat flags).

Spec: Algorithm 1; constants from Theorem 1 / Lemma 1 / Lemma 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bounds import (
    clopper_pearson_lower,
    empirical_bernstein_lcb,
    empirical_bernstein_ucb,
    sample_size_condition,
)

# Sentinel return value when Ĝ is empty
INFEASIBLE = "INFEASIBLE"


@dataclass
class AlgorithmResult:
    """Output of Algorithm 1 + per-pair diagnostics."""

    selected: object  # tuple (lambda_hat_idx, tau_hat_idx) or INFEASIBLE
    lambda_hat_idx: Optional[int] = None
    tau_hat_idx: Optional[int] = None
    eb_per_pair: np.ndarray = field(default_factory=lambda: np.empty(0))
    p_lcb_per_pair: np.ndarray = field(default_factory=lambda: np.empty(0))
    u_lcb_per_pair: np.ndarray = field(default_factory=lambda: np.empty(0))
    in_g_hat_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    z_bar: np.ndarray = field(default_factory=lambda: np.empty(0))
    sigma_z_sq: np.ndarray = field(default_factory=lambda: np.empty(0))
    s_count: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    u_bar: np.ndarray = field(default_factory=lambda: np.empty(0))
    sigma_u_sq: np.ndarray = field(default_factory=lambda: np.empty(0))
    config: dict = field(default_factory=dict)

    @property
    def is_infeasible(self) -> bool:
        return self.selected == INFEASIBLE

    @property
    def n_certified(self) -> int:
        return int(self.in_g_hat_mask.sum())


def certify_grid(
    L: np.ndarray,
    A: np.ndarray,
    v: np.ndarray,
    *,
    alpha: float,
    pi_min: float,
    delta: float,
    c: float,
    V: float,
    B: float,
    check_sample_size: bool = True,
) -> AlgorithmResult:
    """Run Algorithm 1 on a flattened grid of size m = Λ × T.

    Parameters
    ----------
    L : (n_cert, m) array
        Bounded losses, L[i, k] in [0, B].
    A : (n_cert, m) array
        Acceptance indicators, A[i, k] in {0, 1}.
    v : (n_cert, m) array
        Deployment values, v[i, k] in [0, V].
    alpha, pi_min, delta : float
        User parameters in (0, 1).
    c : float
        Abstention cost (>= 0).
    V, B : float
        Upper bounds on v and L respectively.
    check_sample_size : bool
        If True, raise ValueError when n_cert < n_0(m, delta, pi_min).
        Set False for sanity testing.

    Returns
    -------
    AlgorithmResult
        Contains selected (lambda_hat_idx, tau_hat_idx) or INFEASIBLE,
        plus per-pair diagnostics (EB, p_LCB, U_LCB, membership, raw statistics).

    Notes
    -----
    Per Algorithm 1:
    - EB at level δ/(16m): two-sided Maurer-Pontil on Z = A * (L - α).
    - p_LCB at level δ/(16m): Clopper-Pearson on s = Σ A.
    - U_LCB at level δ/(2m): two-sided Maurer-Pontil on u_dep = A*v - c*(1-A).
    - Ĝ = {(λ, τ) : EB ≤ 0 ∧ p_LCB ≥ π_min}.
    - If Ĝ = ∅: return INFEASIBLE. Else: argmax U_LCB over Ĝ.
    """
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if L.ndim != 2 or A.shape != L.shape or v.shape != L.shape:
        raise ValueError(
            f"L, A, v must all be 2D arrays of shape (n_cert, m). "
            f"Got L={L.shape}, A={A.shape}, v={v.shape}."
        )
    n_cert, m = L.shape

    # Validate bounds (algorithmic sanity, not just paranoia)
    if not np.all((L >= 0) & (L <= B + 1e-9)):
        raise ValueError(f"L must lie in [0, B={B}].")
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("A must be {0, 1}-valued.")
    if not np.all((v >= 0) & (v <= V + 1e-9)):
        raise ValueError(f"v must lie in [0, V={V}].")

    # Validate user parameters
    for name, p in [("alpha", alpha), ("pi_min", pi_min), ("delta", delta)]:
        if not (0.0 < p < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got {p}.")
    if c < 0:
        raise ValueError(f"c must be >= 0, got {c}.")

    # Sample-size condition from Theorem 1
    n_required = sample_size_condition(m, delta, pi_min)
    if check_sample_size and n_cert < n_required:
        raise ValueError(
            f"Sample-size condition violated: n_cert={n_cert} < "
            f"n_0(m={m}, δ={delta}, π_min={pi_min}) = {n_required}. "
            f"Set check_sample_size=False to bypass (NOT recommended for production)."
        )

    # Per-sample quantities: Z[i, k] := A[i, k] * (L[i, k] - alpha)
    Z = A * (L - alpha)  # range [-alpha, B-alpha], magnitude bound = B
    u_dep = A * v - c * (1.0 - A)  # range [-c, V]

    # Empirical statistics (vectorized over m grid points)
    Z_bar = Z.mean(axis=0)  # shape (m,)
    u_bar = u_dep.mean(axis=0)  # shape (m,)
    sigma_Z_sq = Z.var(axis=0, ddof=1)  # Bessel-corrected
    sigma_u_sq = u_dep.var(axis=0, ddof=1)
    s = A.sum(axis=0).astype(np.int64)  # shape (m,)

    # Confidence-bound levels (per the union-bound ledger, Table 3)
    delta_eb = delta / (16.0 * m)  # MP on Z
    delta_cp = delta / (16.0 * m)  # Clopper-Pearson on s
    delta_u = delta / (2.0 * m)  # MP on u_dep

    # Compute EB (UCB on E[Z]), p_LCB, U_LCB
    EB = empirical_bernstein_ucb(
        Z_bar, sigma_Z_sq, n_cert, delta_eb, range_b=B
    )
    p_LCB = clopper_pearson_lower(s, n_cert, delta_cp)
    U_LCB = empirical_bernstein_lcb(
        u_bar, sigma_u_sq, n_cert, delta_u, range_b=c + V
    )

    # Membership test for Ĝ
    in_G_hat = (EB <= 0.0) & (p_LCB >= pi_min)

    config = dict(
        n_cert=n_cert,
        m=m,
        alpha=alpha,
        pi_min=pi_min,
        delta=delta,
        c=c,
        V=V,
        B=B,
        delta_eb=delta_eb,
        delta_cp=delta_cp,
        delta_u=delta_u,
        n_required=n_required,
    )

    result = AlgorithmResult(
        selected=INFEASIBLE,
        eb_per_pair=EB,
        p_lcb_per_pair=p_LCB,
        u_lcb_per_pair=U_LCB,
        in_g_hat_mask=in_G_hat,
        z_bar=Z_bar,
        sigma_z_sq=sigma_Z_sq,
        s_count=s,
        u_bar=u_bar,
        sigma_u_sq=sigma_u_sq,
        config=config,
    )

    if not in_G_hat.any():
        # INFEASIBLE — no pair satisfies (EB ≤ 0 ∧ p_LCB ≥ π_min)
        return result

    # argmax U_LCB over Ĝ, with lexicographic tie-breaking by grid index
    U_LCB_masked = np.where(in_G_hat, U_LCB, -np.inf)
    flat_idx = int(np.argmax(U_LCB_masked))
    result.selected = flat_idx
    # If caller uses Lambda × T indexing convention k = lambda_idx * |T| + tau_idx,
    # this decoding happens in grid_search.flat_to_lambda_tau().
    return result
