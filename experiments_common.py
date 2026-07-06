"""Shared helpers for the density experiments (Test 2a and the incorrect-k appendix).

Both sweep random Erdos-Renyi graphs over a range of edge densities, run the
structure-learning algorithm, and plot per-density edge-recovery accuracy split
into true edges ("Edges") and true non-edges ("Missing Edges").

The sweep is parallelized across CPU cores (one process per trial); pass
``workers=1`` for the serial path.
"""

# Keep BLAS single-threaded so per-process parallelism doesn't oversubscribe.
# Must be set before numpy is imported (re-run in each spawned worker, which
# re-imports this module).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from data_generating import binary_var_from_parents
from learn_structure import find_adj_structure_hyp_test, get_errors

DEFAULT_DENSITIES = [(i + 1) / 10 for i in range(9)]  # 0.1 .. 0.9


def erdos_renyi(n, p):
    """Symmetric n x n adjacency matrix; each undirected edge present w.p. p."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            c = np.random.binomial(1, p)
            A[i, j] = c
            A[j, i] = c
    return A


def generate_data_from_er(n, datapoints, adjacency_matrix, var_generator=None):
    """Sample data from an SCM on the given DAG plus a k=2 mixture source U.

    ``var_generator(datapoints, parents)`` draws each observed vertex; it defaults
    to the symmetric SEM (``binary_var_from_parents``) but can be swapped for the
    non-symmetric ``weighted_logistic_from_parents``. The mixture source U is
    always a fair coin, independent of the chosen SEM.
    Returns (data, max_in_degree, n_edges) with data shaped (datapoints, n).
    """
    if var_generator is None:
        var_generator = binary_var_from_parents
    U = binary_var_from_parents(datapoints, [])  # fair-coin mixture source (k=2)
    order = np.random.permutation(n)
    d = {}
    in_degrees = []
    for i in order:
        parents = [U] + [d[j] for j in range(n) if j in d and adjacency_matrix[i, j] == 1]
        in_degrees.append(len(parents) - 1)  # exclude U
        d[i] = var_generator(datapoints, parents)
    data_list = [d[i] for i in range(n)]
    return np.transpose(data_list), max(in_degrees), int(np.sum(adjacency_matrix))


def _density_trial(task):
    """One independent trial (top-level so it is picklable for process spawn)."""
    p, seed, k, alpha, n, datapoints, var_generator, correction, fwer = task
    np.random.seed(seed)
    ER = erdos_renyi(n, p)
    data, _, _ = generate_data_from_er(n, datapoints, ER, var_generator=var_generator)
    adj = find_adj_structure_hyp_test(n, data, k, alpha=alpha, correction=correction, fwer=fwer)
    tp, tn = get_errors(ER, adj)
    return p, tp, tn


def run_density_sweep(k, alpha, n=7, datapoints=10000, reps=20,
                      densities=None, seed=None, var_generator=None, workers=None,
                      correction=None, fwer=0.05):
    """Sweep edge density and collect per-density edge-recovery accuracy.

    The algorithm is run with rank parameter ``k`` (which may deliberately differ
    from the true mixture cardinality of 2, as in the incorrect-k appendix).
    ``var_generator`` selects the SEM (see :func:`generate_data_from_er`).
    ``correction="bonferroni"`` replaces the fixed ``alpha`` with a per-pair,
    test-count-aware removal threshold at family-wise level ``fwer``.
    Trials run in parallel across ``workers`` processes (default: all cores);
    pass ``workers=1`` for a serial run. With ``seed`` set, per-trial seeds are
    derived deterministically from it.

    Returns two dicts keyed by density:
        tp_by_p[p] -> list of "fraction of true edges correctly kept"
        tn_by_p[p] -> list of "fraction of true non-edges correctly removed"
    """
    if densities is None:
        densities = DEFAULT_DENSITIES
    if workers is None:
        workers = os.cpu_count()
    base = seed if seed is not None else int(np.random.randint(0, 2 ** 31 - 1))

    tasks, idx = [], 0
    for p in densities:
        for _ in range(reps):
            tasks.append((p, (base + idx) % (2 ** 31 - 1), k, alpha, n, datapoints,
                          var_generator, correction, fwer))
            idx += 1

    tp_by_p = {p: [] for p in densities}
    tn_by_p = {p: [] for p in densities}
    total = len(tasks)

    if workers == 1:
        for done, t in enumerate(tasks, 1):
            p, tp, tn = _density_trial(t)
            tp_by_p[p].append(tp); tn_by_p[p].append(tn)
            if done % max(1, total // 10) == 0 or done == total:
                print("  {}/{} trials done".format(done, total))
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_density_trial, t) for t in tasks]
            for fut in as_completed(futures):
                p, tp, tn = fut.result()
                tp_by_p[p].append(tp); tn_by_p[p].append(tn)
                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    print("  {}/{} trials done".format(done, total))

    return tp_by_p, tn_by_p


def edge_accuracy_violin(tp_by_p, tn_by_p, xlabel, ylabel, title, filename):
    """Overlaid violins of the Edges / Missing-Edges accuracy vs. density."""
    import matplotlib
    matplotlib.use("Agg")  # headless: write PDFs without needing a display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def _ordered(dd):
        xs = sorted(dd.keys())
        return xs, [dd[x] for x in xs]

    def _add(labels, violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))
        return labels

    labels = []
    xs, ys = _ordered(tp_by_p)
    labels = _add(labels, plt.violinplot(ys, showmeans=False, showmedians=True, showextrema=False), "Edges")
    xs, ys = _ordered(tn_by_p)
    labels = _add(labels, plt.violinplot(ys, showmeans=False, showmedians=True, showextrema=False), "Missing Edges")

    plt.xticks([i + 1 for i in range(len(xs))], labels=[str(x) for x in xs])
    plt.ylim(0, 1)  # fixed axis so panels are visually comparable side by side
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(*zip(*labels), loc="lower left")
    plt.savefig(filename + ".pdf")
    plt.clf()
    print("Wrote {}.pdf".format(filename))