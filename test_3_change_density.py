# Test 2: Changing Density
from data_generating import *
from utils import *
from learn_structure import *
import matplotlib.pyplot as plt

def Erdos_Renyi(n, p):
    adjacency_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            coin_flip = np.random.binomial(1, p)
            adjacency_matrix[i,j] = coin_flip
            adjacency_matrix[j,i] = coin_flip
    return adjacency_matrix

def Generate_Data_from_ER(n, datapoints, adjacency_matrix):
    U = binary_var_from_parents(datapoints, [])
    random_order = np.random.permutation(n)
    data_dict = dict()
    in_degrees = list()
    for i in random_order:
        # parents are all peviously considered vertices
        parents = [U]
        for j in range(n):
            if j in data_dict and adjacency_matrix[i, j] == 1:
                parents = parents + [data_dict[j]]
        in_degrees.append(len(parents) - 1) # subtract one to get rid of U
        new_samples = binary_var_from_parents(datapoints, parents)
        data_dict[i] = new_samples
    data_list = []
    for i in range(n):
        data_list.append(data_dict[i])
    edges = np.sum(adjacency_matrix)
    return np.transpose(data_list), max(in_degrees), edges

n = 7
pval = .0005
ps_tp = dict()
edges_tp = dict()
indegree_tp = dict()
ps_tn = dict()
edges_tn = dict()
indegree_tn = dict()
for i in range(9):
    p = (i+1)/10
    print("Now doing p={}".format(p))
    for rep in range(20):
        ER = Erdos_Renyi(n, p)
        data, max_in_degree, edges = Generate_Data_from_ER(n, 10000, ER)
        tp, tn = get_errors(ER, find_adj_structure_hyp_test(n, data, 2, p = pval))
        
        ps_tp[p] = ps_tp.get(p, []) + [tp]
        edges_tp[edges] = edges_tp.get(edges, []) + [tp]
        indegree_tp[max_in_degree] = indegree_tp.get(max_in_degree, []) + [tp]
        ps_tn[p] = ps_tn.get(p, []) + [tn]
        edges_tn[edges] = edges_tn.get(edges, []) + [tn]
        indegree_tn[max_in_degree] = indegree_tn.get(max_in_degree, []) + [tn]

import matplotlib.patches as mpatches

def process_data_dict(data):
    xs = list(data.keys())
    xs.sort()
    ys = []
    stds = []
    for key in xs:
        ys.append(np.mean(data[key])) #Flipping from error to accuracy
        stds.append(np.std(data[key]))
    return xs, ys, stds

def process_data_dict_for_violin(data):
    xs = list(data.keys())
    xs.sort()
    ys = []
    for key in xs:
        ys.append(data[key]) #Flipping from error to accuracy
    return xs, ys

def make_plot(data_tp, data_tn, xlabel, ylabel, title, filename):
    xs, ys, stds = process_data_dict(data_tp)
    plt.errorbar(xs, ys, yerr=stds, fmt = 'o', label="Edges", capsize=4)
    xs, ys, stds = process_data_dict(data_tn)
    plt.errorbar(xs, ys, yerr=stds, fmt = 'o', label="Missing Edges", capsize=4)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.savefig(filename + ".pdf")
    plt.clf()

def add_label(labels, violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    return labels
    
def make_violin_plot(data_tp, data_tn, xlabel, ylabel, title, filename):
    labels = []
    
    xs, ys = process_data_dict_for_violin(data_tp)
    labels = add_label(labels, plt.violinplot(ys, showmeans=False, showmedians=True, showextrema=False), "Edges")
    
    xs, ys = process_data_dict_for_violin(data_tn)
    labels = add_label(labels, plt.violinplot(ys, showmeans=False, showmedians=True, showextrema=False), "Missing Edges")
    plt.xticks([i + 1 for i in range(len(xs))],
                  labels=[str(x) for x in xs])
    
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(*zip(*labels), loc="lower left")
    #plt.legend()
    plt.savefig(filename + ".pdf")
    plt.clf()
#eps = .01
pval = .05
make_violin_plot(ps_tp, ps_tn, "Graph density (probability of adding each edge)", "Fraction correctly identified", "Graph Density and Edge Accuracy", "density")
make_violin_plot(indegree_tp, indegree_tn, "Maximum in-degree", "Fraction correctly identified", "Maximum In-Degree and Edge Accuracy", "indegree")