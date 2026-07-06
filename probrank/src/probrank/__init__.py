"""probrank: a calibrated hypothesis test for the rank of a probability matrix."""

from ._core import (
    empirical_matrix,
    multinomial_covariance,
    rank_pvalue,
    low_rank,
    d_separated,
    rank_pvalue_from_data,
)

__version__ = "0.1.0"

__all__ = [
    "empirical_matrix",
    "multinomial_covariance",
    "rank_pvalue",
    "low_rank",
    "d_separated",
    "rank_pvalue_from_data",
    "__version__",
]
