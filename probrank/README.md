# probrank

A calibrated **hypothesis test for the rank of a probability matrix**.

Given an `m x n` matrix of probabilities (or cell proportions) estimated from `N`
multinomial samples, `probrank` tests whether the matrix is consistent with rank
at most `k`, following Ratsimalahelo (2001). Under the null `H0: rank <= k` the
statistic is asymptotically chi-squared with `(m - k)(n - k)` degrees of freedom,
so the p-value is (asymptotically) Uniform[0, 1] under H0. A small p-value is
evidence that `rank(M) > k`.

The test is generic. One motivating application is causal discovery in mixtures
of populations, where `rank(M) <= k` corresponds to two (super)variables being
d-separated given a `k`-class latent source — but nothing in the test is specific
to that use.

## Install

```bash
pip install probrank
```

## Usage

```python
import numpy as np
from probrank import rank_pvalue, low_rank, rank_pvalue_from_data

p = rank_pvalue(M, k=2, N=5000)      # small p => rank > k
ok = low_rank(M, k=2, N=5000)        # True    => fail to reject rank <= k

data = np.random.randint(0, 2, size=(5000, 4))
p = rank_pvalue_from_data([0, 1], [2, 3], data, k=2)
```

`d_separated` is provided as a causal-discovery alias for `low_rank`.

## Dependencies

`numpy` and `scipy` only.
