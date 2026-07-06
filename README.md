# Causal Discovery in Mixtures of Populations — experiments

Code to reproduce the experiments in *Causal Discovery in Mixtures of
Populations*. The method recovers a causal skeleton (up to its Markov
equivalence class) that is globally confounded by a discrete latent source `U`
with `k` classes, using **rank tests on probability matrices** in place of
conditional-independence tests.

The rank test itself lives in a standalone, pip-installable package,
[`probrank`](https://pypi.org/project/probrank/); this repository is the
experiment harness that uses it.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pulls in `probrank` from PyPI. `causal-learn` is optional and
only needed for the PC/FCI baseline in Test 2b.

## Repository layout

Core modules (imported by the experiments; not run directly):

| File | Purpose |
|---|---|
| `data_generating.py` | Synthetic SCMs: the connected/split graphs, the "Y" graph, and `binary_var_from_parents` (each vertex is Bernoulli given its parents plus the mixture source `U`). |
| `utils.py` | Pipeline helpers. The rank-test primitives (`get_matrix`, `get_sigma`, `hyp_rank_test`) are re-exported from `probrank`; the conditional test `Rank_Adjacency_Hyp_Test` and helpers live here. |
| `learn_structure.py` | Phase I/II structure learning (`find_adj_structure_hyp_test`) and scoring (`get_errors`). |
| `experiments_common.py` | Shared Erdos-Renyi sampling, the density sweep, and the edge-accuracy violin plot (used by the two density experiments). |

Experiment scripts (each writes the PDF(s) named below into the current folder):

| Script | Paper location | Output figure(s) |
|---|---|---|
| `test1_rank_hypothesis.py` | Test 1: Rank Hypothesis Test | `SV_rank_test.pdf`, `P_rank_test.pdf` |
| `test2_density.py` | Test 2: Varying Density, panel (a) | `density.pdf` |
| `test2_compare_pc_fci.py` | Test 2: Varying Density, panel (b) | `compare_pc_fci.pdf` |
| `appendix_incorrect_k.py` | Appendix: Repeating Test 2 with Incorrect `k` | `density_smallk.pdf` (k=1), `density_bigk.pdf` (k=3) |
| `appendix_nonsymmetric.py` | Appendix: Non-Symmetric Structural Equations | `density_nonsymmetric.pdf` |

## Reproducing the figures

```bash
python test1_rank_hypothesis.py
python test2_density.py
python test2_compare_pc_fci.py          # needs causal-learn; runs in parallel
python appendix_incorrect_k.py
python appendix_nonsymmetric.py
```

Each script has `--help`. Common options: `--reps`, `--datapoints`, `--seed`,
and (where applicable) `--alpha`, `--k`. `test2_compare_pc_fci.py` additionally
takes `--workers` (parallel processes; defaults to all cores) and
`--diagnostics` (prints, per density, how many edges PC/FCI actually return).

## The rank-test threshold `--alpha` (read before reporting numbers)

Structure learning removes an edge when the rank test **fails to reject** the
low-rank (d-separation) null. Because many rank tests are applied to each pair
of variables (over agglomerations, conditioning sets, and their assignments),
the per-test level `alpha` also behaves as an edge-removal threshold and must be
tuned; it is **not** a plain 0.05-style significance level.

- **Larger `alpha` keeps more edges** (removal requires stronger evidence of
  separation); smaller `alpha` removes more aggressively.
- The default `alpha = 0.05` **over-removes** and is a placeholder. Tune it by
  running `test2_density.py` and reading panel (a): raise `alpha` until the
  "Edges" (true-edge retention) curve is acceptable without collapsing the
  "Missing Edges" (correct-removal) curve.
- Use the **same** `alpha` across `test2_density.py`,
  `test2_compare_pc_fci.py`, `appendix_incorrect_k.py`, and `appendix_nonsymmetric.py`,
  and report it in the paper. Test 1 does not use `alpha` (it reports raw
  p-values and singular values).

Principled threshold selection under this multiplicity (e.g. a correction that
scales with the number of tests per pair) is left open in the paper. As an
option, `--correction bonferroni` counts the agglomeration tests applied to each
pair (`count_pair_tests`) and sets the removal threshold to `1 - fwer/T` at
family-wise level `--fwer` (default 0.05). Empirically this lands near the
paper's conservative `.9995`: it fully protects true edges but removes few
non-edges, i.e. it makes Phase I deliberately conservative and leaves the
false-positive edges for the (unimplemented) Phase II k-MixProd correction. No
single threshold — fixed or corrected — makes both the "Edges" and "Missing
Edges" curves high at once; that tradeoff is inherent to a one-phase rank test
under this multiplicity.

## Notes

- Experiments are stochastic. Pass `--seed` for reproducible runs; the density
  experiments accept a base seed, and `test2_compare_pc_fci.py` derives a
  distinct seed per trial from it.
- The slow part is our algorithm, not PC/FCI. All density experiments
  (`test2_density.py`, `appendix_incorrect_k.py`, `appendix_nonsymmetric.py`) and
  `test2_compare_pc_fci.py` run trials in parallel across CPU cores; control with
  `--workers` (default: all cores; pass `--workers 1` for a serial run).
- `test2_compare_pc_fci.py` uses `chisq` conditional-independence tests for the
  discrete data and reads adjacency from causal-learn's graph orientation-
  agnostically (works for both PC's CPDAG and FCI's PAG).

## Citation

Please cite the paper (BibTeX to be added on publication).
