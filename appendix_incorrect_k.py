"""Appendix (paper Section "Repeating Test 2 with Incorrect k").

Reruns the Test 2a density sweep but with the rank parameter deliberately
mis-specified. The data still come from a true k=2 mixture; the algorithm is run
with k=1 (too small) and k=3 (too large).

Outputs (paper Figure "test2_wrongk"):
    density_smallk.pdf   -- k = 1 (too small)
    density_bigk.pdf     -- k = 3 (too large)

Run:
    python appendix_incorrect_k.py
"""

import argparse

from experiments_common import run_density_sweep, edge_accuracy_violin


def main():
    pp = argparse.ArgumentParser(description=__doc__)
    pp.add_argument("--alpha", type=float, default=0.05,
                    help="Rank-test threshold; keep consistent with test2_density.py.")
    pp.add_argument("--reps", type=int, default=20)
    pp.add_argument("--datapoints", type=int, default=10000)
    pp.add_argument("--vertices", type=int, default=7)
    pp.add_argument("--seed", type=int, default=None)
    pp.add_argument("--workers", type=int, default=None, help="Parallel processes (default: all cores).")
    pp.add_argument("--correction", choices=["none","bonferroni"], default="none",
                    help="Multiple-testing correction for the removal threshold.")
    pp.add_argument("--fwer", type=float, default=0.05, help="Family-wise level when --correction bonferroni.")
    args = pp.parse_args()

    for k, outfile, tag in [(1, "density_smallk", "too small"), (3, "density_bigk", "too large")]:
        print("Incorrect-k sweep: k={} ({}), alpha={}".format(k, tag, args.alpha))
        tp, tn = run_density_sweep(k, args.alpha, n=args.vertices,
                                   datapoints=args.datapoints, reps=args.reps, seed=args.seed, workers=args.workers,
                               correction=args.correction, fwer=args.fwer)
        edge_accuracy_violin(tp, tn,
                             "Graph density (probability of adding each edge)",
                             "Fraction correctly identified",
                             "Graph Density and Edge Accuracy (k = {})".format(k),
                             outfile)


if __name__ == "__main__":
    main()
