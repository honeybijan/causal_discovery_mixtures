import numpy as np

from probrank import rank_pvalue, d_separated, empirical_matrix, rank_pvalue_from_data


def _rank_one_matrix():
    u = np.array([0.1, 0.2, 0.3, 0.4])
    v = np.array([0.25, 0.25, 0.25, 0.25])
    return np.outer(u, v)


def test_exact_rank_one_not_rejected_at_k1():
    M = _rank_one_matrix()
    assert rank_pvalue(M, k=1, N=10000) > 0.999
    assert d_separated(M, k=1, N=10000) is True


def test_exact_rank_one_rejected_at_k0():
    M = _rank_one_matrix()
    assert rank_pvalue(M, k=0, N=10000) < 0.5


def test_pvalue_in_unit_interval():
    rng = np.random.default_rng(1)
    data = rng.integers(0, 2, size=(2000, 4))
    p = rank_pvalue_from_data([0, 1], [2, 3], data, k=2)
    assert 0.0 <= p <= 1.0


def test_empirical_matrix_shape_and_normalization():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2, size=(500, 4))
    M = empirical_matrix([0, 1], [2, 3], data)
    assert abs(M.sum() - 1.0) < 1e-9
    assert M.shape[0] <= 4 and M.shape[1] <= 4
