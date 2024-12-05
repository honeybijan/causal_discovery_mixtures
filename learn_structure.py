import itertools
import numpy as np
from utils import *

def get_degree(adjacency_matrix):
    # Assume adjacency matrix is symmetric (we have an undirected skeleton at this step)
    return max(np.sum(adjacency_matrix, axis=0))

# gives all subsets size n and complement of that subset
def comb_and_comp(lst, n):
    # no combinations
    if len(lst) < n:
        return
    # trivial 'empty' combination
    if n == 0 or lst == []:
        yield [], lst
    else:
        first, rest = lst[0], lst[1:]
        # combinations that contain the first element
        for in_, out in comb_and_comp(rest, n - 1):
            yield [first] + in_, out
        # combinations that do not contain the first element
        for in_, out in comb_and_comp(rest, n):
            yield in_, [first] + out

def find_adj_structure(n, data, k, eps = .01):
    adjacency_matrix = np.ones((n,n)) - np.diag(np.ones(n))
    degree = get_degree(adjacency_matrix)
    lgk = int(np.log2(k) + 1)
    sep_set_size = 0
    while sep_set_size <= degree:
        for comb in itertools.combinations(range(n), 2 * lgk + sep_set_size):
            for A, comp in comb_and_comp(list(comb), lgk):
                for B, Cond in comb_and_comp(comp, lgk):
                    if not (Rank_Adjacency_Test(A, B, k, data, conditioning = Cond, epsilon = eps)):
                        for i in A:
                            for j in B:
                                adjacency_matrix[i,j] = 0
                                adjacency_matrix[j,i] = 0
        degree = get_degree(adjacency_matrix)
        sep_set_size +=1 
    return adjacency_matrix

def find_adj_structure_hyp_test(n, data, k, p = .05):
    adjacency_matrix = np.ones((n,n)) - np.diag(np.ones(n))
    degree = get_degree(adjacency_matrix)
    lgk = int(np.log2(k) + 1)
    sep_set_size = 0
    while sep_set_size <= degree:
        for comb in itertools.combinations(range(n), 2 * lgk + sep_set_size):
            for A, comp in comb_and_comp(list(comb), lgk):
                for B, Cond in comb_and_comp(comp, lgk):
                    if not (Rank_Adjacency_Hyp_Test(A, B, k, data, conditioning = Cond, p = p)):
                        for i in A:
                            for j in B:
                                adjacency_matrix[i,j] = 0
                                adjacency_matrix[j,i] = 0
        degree = get_degree(adjacency_matrix)
        sep_set_size +=1 
    return adjacency_matrix

def get_errors(correct_matrix, returned_matrix):
    #print("Correct and Returned")
    #print(correct_matrix)
    #print(returned_matrix)
    shape = correct_matrix.shape
    
    total_edges = np.sum(correct_matrix)
    total_missing = np.sum(np.ones(shape) - correct_matrix)
    edges_correct = np.sum((2 * np.ones(shape) == correct_matrix + returned_matrix).astype(int))
    missing_correct = np.sum((np.zeros(shape) == correct_matrix + returned_matrix).astype(int))
    print("Edges: {} out of {}".format(edges_correct, total_edges))
    print("Missing: {} out of {}".format(missing_correct, total_missing))
    if total_edges > 0:
        true_positive = edges_correct / total_edges
    else:
        true_positive = 1
    if total_missing > 0:
        true_negative = missing_correct / total_missing
    else:
        true_negative = 1
    return true_positive, true_negative