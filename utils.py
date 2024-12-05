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
def get_sigma(matrix):
  R, C = matrix.shape
  cov = np.zeros((R*C, R*C))
  for r1 in range(R):
    for c1 in range(C):
      for r2 in range(R):
        for c2 in range(C):
          p1 = matrix[r1, c1]
          p2 = matrix[r2, c2]
          if r1 == r2 and c1 == c2:
            # Regular Bernouli cov
            cov[c1 * R + r1, c2 * R + r2] = p1*(1 - p1)
          else:
            cov[c1 * R + r1, c2 * R + r2] = -p1 * p2
  return cov

# Hypothesis rank test
def hyp_rank_test(matrix, k, N):
  sigma = get_sigma(matrix)
  f = np.linalg.matrix_rank(sigma)
  svd = np.linalg.svd(matrix, full_matrices=True, compute_uv=True)
  S = svd.S
  d = len(S)
  U2 = svd.U[:,-(d-k):]
  V2= svd.Vh.transpose()[:, -(d-k):]
  L = np.diag(S[-(d-k):])
  l = L.flatten('F')

  Q_dag = np.kron(V2.T, U2.T) @ np.linalg.pinv(sigma) @ np.kron(V2, U2)
  stat = N * l.T @ Q_dag @ l
  return sp.stats.chi2.sf(stat, f)

def get_matrix(A, B, data):
  N = len(data)
  data_t = np.transpose(data)
  A_val_to_index = get_val_to_index(data_t[A])
  B_val_to_index = get_val_to_index(data_t[B])
  matrix = np.zeros((len(A_val_to_index.keys()), len(B_val_to_index.keys())))
  for a, b in zip(np.transpose(data_t[A]), np.transpose(data_t[B])):
    index_a = A_val_to_index[tuple(a)]
    index_b = B_val_to_index[tuple(b)]
    matrix[index_a][index_b] += 1
  return matrix/N

def Rank_Adjacency_Test(A, B, k, data, conditioning = [], epsilon = .01):
  if conditioning:
    # Then try for each subset
    result = False
    for data_subset in Condition_Data(conditioning, data):
      result = result or Rank_Adjacency_Test(A, B, k, data_subset, epsilon = epsilon)
    return result
  matrix = get_matrix(A, B, data)
  return effective_rank_SV(matrix, epsilon=epsilon) > k
  
def Rank_Adjacency_Hyp_Test(A, B, k, data, conditioning = [], p = .05):
  N = len(data)
  if conditioning:
    # Then try for each subset
    result = False
    for data_subset in Condition_Data(conditioning, data):
      result = result or Rank_Adjacency_Hyp_Test(A, B, k, data_subset, p = p)
    return result
  matrix = get_matrix(A, B, data)
  pval = hyp_rank_test(matrix, k, N)
  return  pval < 1 - p