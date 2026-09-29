"""Three-way split of the evaluation pool into D_tune / D_cert / D_test.

The guarantee requires f, g and the grid Λ × T to be fixed independently of
D_cert. With a pre-trained model and a grid fixed independently of
certification, this reduces to a random split:
  D_tune: reserved for fitting the selector or building the grid; never read
          by Algorithm 1
  D_cert: certification samples for Algorithm 1
  D_test: held-out samples for reporting realised performance
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def three_split_indices(
    n_total: int,
    n_tune: int,
    n_cert: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random 3-split of indices [0, n_total) into (tune, cert, test).

    The remaining (n_total - n_tune - n_cert) indices go to D_test.

    Parameters
    ----------
    n_total : int
        Total number of samples available (e.g., 10000 for CIFAR-100 test split).
    n_tune : int
        Number of samples for D_tune.
    n_cert : int
        Number of samples for D_cert.
    seed : int
        Random seed (controls the shuffle).

    Returns
    -------
    tune_idx, cert_idx, test_idx : np.ndarray of int64
        Disjoint index arrays. tune ∪ cert ∪ test = [0, n_total).

    Raises
    ------
    ValueError
        If n_tune + n_cert > n_total.
    """
    if n_tune + n_cert > n_total:
        raise ValueError(
            f"n_tune ({n_tune}) + n_cert ({n_cert}) > n_total ({n_total})."
        )
    if n_tune < 0 or n_cert < 0:
        raise ValueError("n_tune and n_cert must be non-negative.")

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n_total)
    tune_idx = permuted[:n_tune]
    cert_idx = permuted[n_tune : n_tune + n_cert]
    test_idx = permuted[n_tune + n_cert :]
    return tune_idx, cert_idx, test_idx
