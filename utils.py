import scipy as sp
import numpy as np
import scipy.linalg as scipyLA

def get_val_to_index(data_t):
  data = np.transpose(data_t)
  val_to_index = dict()
  vals = 0
  for val in list(data):
    t_val = tuple(val)
    if t_val not in val_to_index:
      val_to_index[t_val] = vals
      vals += 1
  return val_to_index

def Condition_Data(C, data):
  data_t = np.transpose(data)
  C_vals = get_val_to_index(data_t[C]).keys()
  data_subsets = []
  for val in C_vals:
    C_data = np.transpose(data_t[C])
    indices = np.where((C_data == val).all(axis=1))[0]
    data_subset = data[indices]
    data_subsets.append(data_subset)
  return data_subsets

# Test using singular values
def effective_rank_SV(matrix, epsilon = .01):
  svdvals = scipyLA.svdvals(matrix)
  largest = svdvals[0]
  adjusted_svdvals = svdvals / largest
  #print(np.log(adjusted_svdvals))
  big_enough = [svdval for svdval in adjusted_svdvals if svdval > epsilon]
  return len(big_enough)

# Get covariance matrix of the matrix entries (used for hypothesis test)
# The rank-test primitives now live in the installable `probrank` package
# (pip install probrank). We re-export them under their historical names so the
# rest of this pipeline is unchanged.
from probrank import (
    empirical_matrix as get_matrix,
    multinomial_covariance as get_sigma,
    rank_pvalue as _rank_pvalue,
)


def hyp_rank_test(matrix, k, N):
  # Thin wrapper over probrank.rank_pvalue. A matrix with <= k rows or columns
  # has rank <= k by construction (this can happen for a conditioned subset in
  # which some category is unobserved), so it is trivially consistent with the
  # low-rank/separation null; return a p-value of 1.0 rather than testing.
  m, n = matrix.shape
  if min(m, n) <= k:
    return 1.0
  return _rank_pvalue(matrix, k, N)


def Rank_Adjacency_Test(A, B, k, data, conditioning = [], epsilon = .01):
  if conditioning:
    # Then try for each subset
    result = False
    for data_subset in Condition_Data(conditioning, data):
      result = result or Rank_Adjacency_Test(A, B, k, data_subset, epsilon = epsilon)
    return result
  matrix = get_matrix(A, B, data)
  return effective_rank_SV(matrix, epsilon=epsilon) > k
  
def Rank_Adjacency_Hyp_Test(A, B, k, data, conditioning = [], alpha = .05):
  # Declares ADJACENCY (returns True) when we reject H0: rank<=k at level alpha,
  # i.e. when pval < alpha. Non-adjacency (edge removed) when we fail to reject.
  # NOTE: the old rule was `pval < 1 - p`, which only made sense because the
  # previous (miscalibrated) test piled p-values near 1 under the null. With the
  # corrected, calibrated hyp_rank_test the standard rule below is the right one.
  N = len(data)
  if conditioning:
    # Then try for each subset
    result = False
    for data_subset in Condition_Data(conditioning, data):
      result = result or Rank_Adjacency_Hyp_Test(A, B, k, data_subset, alpha = alpha)
    return result
  matrix = get_matrix(A, B, data)
  pval = hyp_rank_test(matrix, k, N)
  return pval < alpha