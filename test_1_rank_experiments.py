from data_generating import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def add_label(labels, violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    return labels

def make_violin_plot(Ns, y_con, y_uncon, xlabel, ylabel, title, filename):
    labels = []
    labels = add_label(labels, plt.violinplot(y_con, showmeans=False, showmedians=True, showextrema=False), "Connected") 
    labels = add_label(labels, plt.violinplot(y_uncon, showmeans=False, showmedians=True, showextrema=False), "Split")
    xs = Ns
    plt.xticks([i + 1 for i in range(len(xs))], labels=[str(x) for x in xs])
    
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(*zip(*labels), loc="lower left")
    #plt.legend()
    plt.savefig(filename + ".pdf")
    plt.clf()


def test_sv(matrix, k):
  m, _ = matrix.shape
  svdvals = scipyLA.svdvals(matrix)
  largest = svdvals[0]
  adjusted_svdvals = svdvals / largest
  adjusted_svdvals.sort()
  return adjusted_svdvals[m-k - 1]


Ns = []
con_sv = []
uncon_sv = []
con_p = []
uncon_p = []
for N in range(1000, 10000, 1000):
  Ns.append(N)
  con_sv_col = []
  uncon_sv_col = []
  con_p_col = []
  uncon_p_col = []
  for i in range(200):
    connected_data = generate_data_connected(N) # Data with (V0, V1) dependent on (V2, V3) through V1 -> V2
    unconnected_data = generate_data_not_connected(N) # Data with (V0, V1) cond independent from (V2, V3)

    matrix_c = get_matrix([0, 1], [2, 3], connected_data)
    p_test_c = hyp_rank_test(matrix_c, 2, N)
    sv_test_c = test_sv(matrix_c, 2)

    matrix_uc = get_matrix([0, 1], [2, 3], unconnected_data)
    p_test_uc = hyp_rank_test(matrix_uc, 2, N)
    sv_test_uc = test_sv(matrix_uc, 2)

    con_sv_col.append(sv_test_c)
    uncon_sv_col.append(sv_test_uc)
    con_p_col.append(p_test_c)
    uncon_p_col.append(p_test_uc)
  
  con_sv.append(con_sv_col)
  uncon_sv.append(uncon_sv_col)
  con_p.append(con_p_col)
  uncon_p.append(uncon_p_col)

make_violin_plot(Ns, con_sv, uncon_sv, "Number of Data Samples", "$k+1$th Singular Value", "Singular Value Rank Test", "SV_rank_test")
make_violin_plot(Ns, con_p, uncon_p, "Number of Data Samples", "P-value", "Rank Hypothesis Test", "P_rank_test")