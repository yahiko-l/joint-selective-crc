"""Certified-Utility Selective Risk Control via a Margin-Oracle Ratio Bound.

Implementation of Algorithm 1 of the paper.
"""

from .bounds import (
    maurer_pontil_two_sided_radius,
    clopper_pearson_lower,
    chernoff_lower_tail_threshold,
    chernoff_upper_tail_threshold,
)
from .certify import certify_grid, AlgorithmResult
from .grid_search import build_lambda_tau_grid
from .protocol import three_split_indices
from .baselines import (
    baseline_a_range_hoeffding,
    baseline_a_range_hoeffding_pacc,
    baseline_a_range_hoeffding_plcb,
    baseline_b_accepted_bernstein,
    baseline_b_full_certificate,
    wsr_betting_ucb,
    ebh_product_evalue_ucb,
    scrct_quantile_ucb,
    score_evalue_ucb,
)

__all__ = [
    "maurer_pontil_two_sided_radius",
    "clopper_pearson_lower",
    "chernoff_lower_tail_threshold",
    "chernoff_upper_tail_threshold",
    "certify_grid",
    "AlgorithmResult",
    "build_lambda_tau_grid",
    "three_split_indices",
    "baseline_a_range_hoeffding",
    "baseline_a_range_hoeffding_pacc",
    "baseline_a_range_hoeffding_plcb",
    "baseline_b_accepted_bernstein",
    "baseline_b_full_certificate",
    "wsr_betting_ucb",
    "ebh_product_evalue_ucb",
    "scrct_quantile_ucb",
    "score_evalue_ucb",
]
