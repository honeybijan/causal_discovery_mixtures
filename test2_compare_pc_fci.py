"""Test 2b (paper Section "Test 2: Varying Density", panel (b)).

Compares our algorithm (the probrank rank test inside the structure-learning
pipeline) against PC and FCI from causal-learn, over the same Erdos-Renyi density
sweep as test2_density.py, scoring overall adjacency accuracy. A dashed
"always-adjacent" baseline (= graph density) is drawn for reference.

Output (paper Figure "test2res" (b)):
    compare_pc_fci.pdf

Trials are independent and run in parallel across CPU cores (--workers).
Requires causal-learn for the PC/FCI baselines: pip install causal-learn

Run:
    python test2_compare_pc_fci.py
    python test2_compare_pc_fci.py --diagnostics    # print PC/FCI edge counts

NOTE ON --our-alpha: the rank-test threshold NEEDS TUNING (see README); the
default 0.05 over-removes. Keep it consistent with test2_density.py.
"""

# Keep BLAS single-threaded so it doesn't oversubscribe against process-level
# parallelism. Must be set before numpy is imported (also re-run in each spawned
# worker, since the module is re-imported there).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from experiments_common import erdos_renyi, generate_data_from_er, DEFAULT_DENSITIES
from learn_structure import find_adj_structure_hyp_test

DENSITIES = DEFAULT_DENSITIES
PCFCI_TEST = "chisq"   # discrete (binary) data -> Chi-squared CI test

# causal-learn is optional; PC/FCI are skipped (with a warning) if it is absent.
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    HAVE_CAUSAL_LEARN = True
except ImportError:
    HAVE_CAUSAL_LEARN = False


def cl_to_adjacency(graph_matrix, n):
    """Undirected skeleton from a causal-learn graph matrix: two nodes are
    adjacent iff either directed entry is nonzero (orientation-agnostic; works
    for PC's CPDAG and FCI's PAG)."""
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i):
            if graph_matrix[i, j] != 0 or graph_matrix[j, i] != 0:
                A[i, j] = 1
                A[j, i] = 1
    return A


def adjacency_accuracy(true_adj, pred_adj):
    """Fraction of unordered pairs classified correctly (strict upper triangle)."""
    n = true_adj.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean((true_adj[iu] != 0) == (pred_adj[iu] != 0)))


def _n_edges(adj):
    iu = np.triu_indices(adj.shape[0], k=1)
    return int(np.sum(adj[iu] != 0))


def run_pc(data, n, alpha, test):
    cg = pc(data.astype(int), alpha, test, show_progress=False)
    return cl_to_adjacency(cg.G.graph, n)


def run_fci(data, n, alpha, test):
    g, _ = fci(data.astype(int), independence_test_method=test, alpha=alpha,
               verbose=False, show_progress=False)
    return cl_to_adjacency(g.graph, n)


# ---- one independent trial (top-level so it is picklable for spawn) ----
def run_one_trial(task):
    density, seed, n, datapoints, our_alpha, pcfci_alpha, test, want_pcfci = task
    np.random.seed(seed)
    n_pairs = n * (n - 1) // 2

    ER = erdos_renyi(n, density)
    data, _, _ = generate_data_from_er(n, datapoints, ER)

    res = {"density": density, "true_edges": _n_edges(ER)}
    res["baseline"] = res["true_edges"] / n_pairs   # always-adjacent accuracy

    ours = find_adj_structure_hyp_test(n, data, k=2, alpha=our_alpha)
    res["ours"] = adjacency_accuracy(ER, ours)

    if HAVE_CAUSAL_LEARN and want_pcfci:
        try:
            a = run_pc(data, n, pcfci_alpha, test)
            res["pc"] = adjacency_accuracy(ER, a); res["pc_edges"] = _n_edges(a)
        except Exception as e:
            res["pc_err"] = repr(e)
        try:
            a = run_fci(data, n, pcfci_alpha, test)
            res["fci"] = adjacency_accuracy(ER, a); res["fci_edges"] = _n_edges(a)
        except Exception as e:
            res["fci_err"] = repr(e)
    return res


def main():
    pp = argparse.ArgumentParser(description="Test 2 panel (b): adjacency accuracy vs. density, ours vs PC/FCI.")
    pp.add_argument("--our-alpha", type=float, default=0.05,
                    help="Keep-edge threshold for OUR algorithm (larger => keeps MORE edges). "
                         "Default 0.05 OVER-REMOVES; this is the knob to tune.")
    pp.add_argument("--pcfci-alpha", type=float, default=0.05, help="Significance level for PC/FCI CI tests.")
    pp.add_argument("--reps", type=int, default=20, help="Graphs per density level.")
    pp.add_argument("--datapoints", type=int, default=10000, help="Samples per graph.")
    pp.add_argument("--vertices", type=int, default=7, help="Number of observed vertices.")
    pp.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Parallel worker processes (default: all cores).")
    pp.add_argument("--seed", type=int, default=0, help="Base RNG seed (per-trial seeds derive from it).")
    pp.add_argument("--outfile", type=str, default="compare_pc_fci", help="Output PDF basename.")
    pp.add_argument("--diagnostics", action="store_true",
                    help="Print, per density, mean edges PC/FCI return vs. true count and complete "
                         "(reveals whether PC/FCI behave like the always-adjacent baseline).")
    args = pp.parse_args()

    if not HAVE_CAUSAL_LEARN:
        warnings.warn("causal-learn not installed; PC and FCI will be skipped. "
                      "Run `pip install causal-learn` for the full comparison.")

    n = args.vertices
    n_pairs = n * (n - 1) // 2
    print("Config: vertices={} datapoints={} reps={} workers={} OUR_ALPHA={} PCFCI_ALPHA={}".format(
        n, args.datapoints, args.reps, args.workers, args.our_alpha, args.pcfci_alpha))

    tasks = []
    idx = 0
    for p in DENSITIES:
        for _ in range(args.reps):
            tasks.append((p, args.seed + idx, n, args.datapoints,
                          args.our_alpha, args.pcfci_alpha, PCFCI_TEST, HAVE_CAUSAL_LEARN))
            idx += 1

    acc = {"Our Algorithm": {p: [] for p in DENSITIES},
           "PC": {p: [] for p in DENSITIES},
           "FCI": {p: [] for p in DENSITIES}}
    baseline = {p: [] for p in DENSITIES}
    edgecount = {"true": {p: [] for p in DENSITIES},
                 "PC": {p: [] for p in DENSITIES},
                 "FCI": {p: [] for p in DENSITIES}}

    done = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_one_trial, t) for t in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            p = r["density"]
            acc["Our Algorithm"][p].append(r["ours"])
            baseline[p].append(r["baseline"])
            edgecount["true"][p].append(r["true_edges"])
            if "pc" in r:
                acc["PC"][p].append(r["pc"]); edgecount["PC"][p].append(r["pc_edges"])
            if "fci" in r:
                acc["FCI"][p].append(r["fci"]); edgecount["FCI"][p].append(r["fci_edges"])
            for key in ("pc_err", "fci_err"):
                if key in r:
                    warnings.warn("{} at density {}: {}".format(key, p, r[key]))
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                print("  {}/{} trials done".format(done, total))

    if args.diagnostics:
        print("\n--- edge-count diagnostics (mean edges returned; complete graph = {}) ---".format(n_pairs))
        print("density |  true  |   PC   |  FCI   | complete")
        for p in DENSITIES:
            def _m(key):
                v = edgecount[key][p]
                return "{:5.1f}".format(np.mean(v)) if v else "  n/a"
            print("  {:.1f}   | {} | {} | {} |  {}".format(p, _m("true"), _m("PC"), _m("FCI"), n_pairs))
        print("If PC/FCI columns are close to `complete`, they behave like the "
              "always-adjacent baseline; if close to `true`, they recover structure.\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def add_label(labels, violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))
        return labels

    labels = []
    for method in ["Our Algorithm", "PC", "FCI"]:
        ys = [acc[method][p] for p in DENSITIES]
        if not any(len(col) for col in ys):
            continue
        v = plt.violinplot(ys, showmeans=False, showmedians=True, showextrema=False)
        labels = add_label(labels, v, method)

    xpos = [i + 1 for i in range(len(DENSITIES))]
    base_med = [np.median(baseline[p]) if baseline[p] else np.nan for p in DENSITIES]
    bline, = plt.plot(xpos, base_med, "k--", marker="o", markersize=3, linewidth=1.5)
    labels.append((bline, "Always adjacent (= density)"))

    plt.xticks(xpos, labels=[str(p) for p in DENSITIES])
    plt.title("Graph Density and Overall Adjacency Accuracy", fontsize=18)
    plt.xlabel("Graph density (probability of adding each edge)", fontsize=16)
    plt.ylabel("Adjacency Accuracy", fontsize=16)
    plt.legend(*zip(*labels), loc="lower left")
    plt.savefig(args.outfile + ".pdf")
    plt.clf()
    print("Wrote {}.pdf".format(args.outfile))


if __name__ == "__main__":
    main()
