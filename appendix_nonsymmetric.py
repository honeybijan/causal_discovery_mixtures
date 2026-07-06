"""Appendix (paper Section "Non-Symmetric Structural Equations").

Repeats the Test 2a density sweep, but generates data with the non-symmetric
weighted-logistic SEM (each parent gets an independent random weight) instead of
the symmetric SEM. The structure-learning algorithm is unchanged (it is
nonparametric); this probes whether recovery is robust to asymmetric, non-
additive dependence.

Output:
    density_nonsymmetric.pdf

Run:
    python appendix_nonsymmetric.py
"""

import argparse

from data_generating import weighted_logistic_from_parents
from experiments_common import run_density_sweep, edge_accuracy_violin


def main():
    pp = argparse.ArgumentParser(description=__doc__)
    pp.add_argument("--alpha", type=float, default=0.05,
                    help="Rank-test threshold; keep consistent with the other experiments.")
    pp.add_argument("--k", type=int, default=2)
    pp.add_argument("--reps", type=int, default=20)
    pp.add_argument("--datapoints", type=int, default=10000)
    pp.add_argument("--vertices", type=int, default=7)
    pp.add_argument("--seed", type=int, default=None)
    pp.add_argument("--workers", type=int, default=None, help="Parallel processes (default: all cores).")
    pp.add_argument("--correction", choices=["none","bonferroni"], default="none",
                    help="Multiple-testing correction for the removal threshold.")
    pp.add_argument("--fwer", type=float, default=0.05, help="Family-wise level when --correction bonferroni.")
    pp.add_argument("--outfile", type=str, default="density_nonsymmetric")
    args = pp.parse_args()

    print("Non-symmetric SEM density sweep (k={}, alpha={}, reps={})".format(
        args.k, args.alpha, args.reps))
    tp, tn = run_density_sweep(args.k, args.alpha, n=args.vertices,
                               datapoints=args.datapoints, reps=args.reps, seed=args.seed, workers=args.workers,
                               correction=args.correction, fwer=args.fwer,
                               var_generator=weighted_logistic_from_parents)
    edge_accuracy_violin(tp, tn,
                         "Graph density (probability of adding each edge)",
                         "Fraction correctly identified",
                         "Edge Accuracy (non-symmetric SEM)",
                         args.outfile)


if __name__ == "__main__":
    main()
