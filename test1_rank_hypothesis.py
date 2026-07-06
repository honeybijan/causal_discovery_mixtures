"""Test 1 (paper Section "Test 1: Rank Hypothesis Test").

Compares the rank hypothesis test (probrank) against naive thresholding of the
(k+1)-th singular value, on a "connected" vs a "split" 4-variable graph.

Outputs (paper Figure "test1res"):
    SV_rank_test.pdf   -- (k+1)-th singular value, connected vs split
    P_rank_test.pdf    -- rank-test p-value, connected vs split

Run:
    python test1_rank_hypothesis.py
"""

import argparse

import numpy as np
import scipy.linalg as scipyLA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_generating import generate_data_connected, generate_data_not_connected
from probrank import empirical_matrix, rank_pvalue

K = 2  # true mixture cardinality


def add_label(labels, violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    return labels


def make_violin_plot(Ns, y_con, y_uncon, xlabel, ylabel, title, filename):
    labels = []
    labels = add_label(labels, plt.violinplot(y_con, showmeans=False, showmedians=True, showextrema=False), "Connected")
    labels = add_label(labels, plt.violinplot(y_uncon, showmeans=False, showmedians=True, showextrema=False), "Split")
    plt.xticks([i + 1 for i in range(len(Ns))], labels=[str(x) for x in Ns])
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(*zip(*labels), loc="lower left")
    plt.savefig(filename + ".pdf")
    plt.clf()
    print("Wrote {}.pdf".format(filename))


def kth_singular_value(matrix, k):
    """The (k+1)-th singular value, normalized by the largest (as in the paper)."""
    m, _ = matrix.shape
    sv = scipyLA.svdvals(matrix)
    sv = np.sort(sv / sv[0])
    return sv[m - k - 1]


def main():
    pp = argparse.ArgumentParser(description=__doc__)
    pp.add_argument("--reps", type=int, default=200, help="Repetitions per sample size.")
    pp.add_argument("--min-n", type=int, default=1000)
    pp.add_argument("--max-n", type=int, default=9000)
    pp.add_argument("--step-n", type=int, default=1000)
    args = pp.parse_args()

    Ns, con_sv, uncon_sv, con_p, uncon_p = [], [], [], [], []
    for N in range(args.min_n, args.max_n + 1, args.step_n):
        print("Now doing N={}".format(N))
        Ns.append(N)
        csv, usv, cp, up = [], [], [], []
        for _ in range(args.reps):
            cdata = generate_data_connected(N)       # (V0,V1) depend on (V2,V3) via V1->V2
            udata = generate_data_not_connected(N)   # (V0,V1) independent of (V2,V3) given U

            Mc = empirical_matrix([0, 1], [2, 3], cdata)
            Mu = empirical_matrix([0, 1], [2, 3], udata)
            cp.append(rank_pvalue(Mc, K, N))
            up.append(rank_pvalue(Mu, K, N))
            csv.append(kth_singular_value(Mc, K))
            usv.append(kth_singular_value(Mu, K))

        con_sv.append(csv); uncon_sv.append(usv)
        con_p.append(cp);   uncon_p.append(up)

    make_violin_plot(Ns, con_sv, uncon_sv, "Number of Data Samples",
                     "$k+1$th Singular Value", "Singular Value Rank Test", "SV_rank_test")
    make_violin_plot(Ns, con_p, uncon_p, "Number of Data Samples",
                     "P-value", "Rank Hypothesis Test", "P_rank_test")


if __name__ == "__main__":
    main()
