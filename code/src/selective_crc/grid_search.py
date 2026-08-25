"""Grid construction utilities for Algorithm 1.

The (λ, τ) grid is flattened as k = lambda_idx * |T| + tau_idx so that
all per-pair quantities can be stored as 1D arrays of length m = |Λ × T|.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def build_lambda_tau_grid(
    Lambda: Sequence[float],
    T: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a flat grid index mapping.

    Returns
    -------
    Lambda_flat : np.ndarray, shape (m,)
        λ value at each flat grid index k.
    T_flat : np.ndarray, shape (m,)
        τ value at each flat grid index k.

    Convention
    ----------
    Flat index k = lambda_idx * |T| + tau_idx. Use flat_to_lambda_tau and
    lambda_tau_to_flat to convert between representations.
    """
    Lambda = np.asarray(Lambda, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    LL, TT = np.meshgrid(Lambda, T, indexing="ij")
    return LL.ravel(), TT.ravel()


def flat_to_lambda_tau(flat_idx: int, n_lambda: int, n_tau: int) -> Tuple[int, int]:
    """Decode flat grid index k to (lambda_idx, tau_idx)."""
    if not (0 <= flat_idx < n_lambda * n_tau):
        raise ValueError(f"flat_idx={flat_idx} out of range for {n_lambda}x{n_tau}.")
    return flat_idx // n_tau, flat_idx % n_tau


def lambda_tau_to_flat(lambda_idx: int, tau_idx: int, n_tau: int) -> int:
    return lambda_idx * n_tau + tau_idx
