"""Core implementation of the calibrated rank hypothesis test.

Given an m x n matrix of probabilities estimated from N multinomial samples,
tests whether the matrix is consistent with rank at most k (Ratsimalahelo 2001).
Under H0: rank <= k the statistic is asymptotically chi-squared with
(m - k)(n - k) degrees of freedom. Small p-values are evidence that rank > k.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "empirical_matrix",
    "multinomial_covariance",
    "rank_pvalue",
    "low_rank",
    "d_separated",
    "rank_pvalue_from_data",
]


def _val_to_index(columns):
    samples = np.transpose(columns)
    mapping = {}
    nxt = 0
    for row in samples:
        key = tuple(row)
        if key not in mapping:
            mapping[key] = nxt
            nxt += 1
    return mapping


def empirical_matrix(A, B, data):
    """Empirical joint-probability matrix M[a, b] = Pr(A = a, B = b)."""
    data = np.asarray(data)
    N = len(data)
    dt = np.transpose(data)
    a_idx = _val_to_index(dt[A])
    b_idx = _val_to_index(dt[B])
    M = np.zeros((len(a_idx), len(b_idx)))
    for a_row, b_row in zip(np.transpose(dt[A]), np.transpose(dt[B])):
        M[a_idx[tuple(a_row)], b_idx[tuple(b_row)]] += 1
    return M / N


def multinomial_covariance(matrix):
    """Covariance of the vectorized cell proportions under a multinomial draw."""
    matrix = np.asarray(matrix, dtype=float)
    R, C = matrix.shape
    cov = np.zeros((R * C, R * C))
    for r1 in range(R):
        for c1 in range(C):
            for r2 in range(R):
                for c2 in range(C):
                    p1 = matrix[r1, c1]
                    p2 = matrix[r2, c2]
                    if r1 == r2 and c1 == c2:
                        cov[c1 * R + r1, c2 * R + r2] = p1 * (1 - p1)
                    else:
                        cov[c1 * R + r1, c2 * R + r2] = -p1 * p2
    return cov


def rank_pvalue(matrix, k, N):
    """p-value for H0: rank(matrix) <= k from N samples."""
    matrix = np.asarray(matrix, dtype=float)
    m, n = matrix.shape
    if not (0 <= k < min(m, n)):
        raise ValueError(
            "need 0 <= k < min(m, n); got k={!r} for a {}x{} matrix".format(k, m, n)
        )
    sigma = multinomial_covariance(matrix)
    U, S, Vh = np.linalg.svd(matrix, full_matrices=True)
    d = len(S)
    U2 = U[:, -(d - k):]
    V2 = Vh.T[:, -(d - k):]
    l = np.diag(S[-(d - k):]).flatten("F")
    W = np.kron(V2, U2)
    Omega = W.T @ sigma @ W
    stat = N * l.T @ np.linalg.pinv(Omega) @ l
    df = (m - k) * (n - k)
    return float(stats.chi2.sf(stat, df))


def low_rank(matrix, k, N, alpha=0.05):
    """True if we fail to reject rank <= k at level alpha."""
    return rank_pvalue(matrix, k, N) >= alpha


def d_separated(matrix, k, N, alpha=0.05):
    """Causal-discovery alias for low_rank."""
    return low_rank(matrix, k, N, alpha=alpha)


def rank_pvalue_from_data(A, B, data, k):
    """Build the matrix for column-sets A and B from data, then test its rank."""
    N = len(data)
    return rank_pvalue(empirical_matrix(A, B, data), k, N)
