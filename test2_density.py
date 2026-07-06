"""Test 2a (paper Section "Test 2: Varying Density", panel (a)).

Sweeps random Erdos-Renyi graphs over edge densities 0.1..0.9 and plots the
fraction of true edges kept ("Edges") and true non-edges removed ("Missing
Edges") by the structure-learning algorithm.

Output (paper Figure "test2res" (a)):
    density.pdf

Run:
    python test2_density.py                 # defaults: k=2, alpha=0.05, 20 reps
    python test2_density.py --alpha 0.3     # sweep alpha while tuning (see README)
"""

import argparse

from experiments_common import run_density_sweep, edge_accuracy_violin


def main():
    pp = argparse.ArgumentParser(description=__doc__)
    pp.add_argument("--alpha", type=float, default=0.05,
                    help="Rank-test threshold (adjacency when p < alpha). NEEDS TUNING; "
                         "see README. Larger alpha keeps more edges.")
    pp.add_argument("--k", type=int, default=2, help="Rank parameter (true mixture cardinality is 2).")
    pp.add_argument("--reps", type=int, default=20, help="Graphs per density level.")
    pp.add_argument("--datapoints", type=int, default=10000, help="Samples per graph.")
    pp.add_argument("--vertices", type=int, default=7)
    pp.add_argument("--seed", type=int, default=None)
    pp.add_argument("--workers", type=int, default=None, help="Parallel processes (default: all cores).")
    pp.add_argument("--correction", choices=["none","bonferroni"], default="none",
                    help="Multiple-testing correction for the removal threshold.")
    pp.add_argument("--fwer", type=float, default=0.05, help="Family-wise level when --correction bonferroni.")
    pp.add_argument("--outfile", type=str, default="density")
    args = pp.parse_args()

    print("Test 2a: density sweep (k={}, alpha={}, reps={})".format(args.k, args.alpha, args.reps))
    tp, tn = run_density_sweep(args.k, args.alpha, n=args.vertices,
                               datapoints=args.datapoints, reps=args.reps, seed=args.seed, workers=args.workers,
                               correction=args.correction, fwer=args.fwer)
    edge_accuracy_violin(tp, tn,
                         "Graph density (probability of adding each edge)",
                         "Fraction correctly identified",
                         "Graph Density and Edge Accuracy",
                         args.outfile)


if __name__ == "__main__":
    main()
