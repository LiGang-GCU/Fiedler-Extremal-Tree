# Fiedler Extremal Tree — Experiment Code

Code and results for the paper:

> **The Outer-Leaf Extremal Property of the Fiedler Vector on Trees**  
> Gang Li, Guangzhou College of Commerce  
> Submitted to *Discrete Applied Mathematics*, 2026

---

## What this code does

For a tree *T* with simple algebraic connectivity λ₂ and Fiedler vector **f**,
the **Outer-Leaf Extremal Property (OLEP)** asserts that the global max and min
of **f** are attained at the deepest leaves of the positive and negative sign
supports, when *T* is rooted at the spectral center *c*.

This repository provides:

| Script | Purpose |
|---|---|
| `verify_olep.py` | Exhaustive OLEP check, n=5..17; random-model survey (Tables 3–4 in paper) |
| `run_experiments.py` | Additional counterexample analysis and algorithm accuracy sweep |
| `plot_counterexamples.py` | Generate Figure 2 (CE1, CE2 counterexample visualization) |

---

## Requirements

```
Python >= 3.9
numpy
scipy
networkx
matplotlib   # for plot_counterexamples.py only
```

Install with:

```bash
pip install numpy scipy networkx matplotlib
```

---

## Quick start

### Reproduce Table 3 and Table 4

```bash
python verify_olep.py
```

Runtime: ~25 s on a modern laptop.

Outputs written to `results/final_defB/`:

| File | Content |
|---|---|
| `table1_summary.txt` | Human-readable Table 3 |
| `table1_by_n.json` | Per-*n* statistics (JSON) |
| `table2_random.json` | Table 4 random-model rates (JSON) |
| `counterex_exhaustive.json` | All 830 counterexamples with full graph + Fiedler vector |
| `counterex_metadata.json` | Summary metadata |

### Generate Figure 2

```bash
python plot_counterexamples.py
# writes figures/olep_counterexamples.pdf
```

---

## Pre-computed results

The `results/` directory contains the results already computed and used in the paper:

```
results/
  final_defB/
    table1_by_n.json        Table 3 per-n data
    table1_summary.txt      Table 3 plain text
    table2_random.json      Table 4 data
  counterex_verify/
    table1_counterex_by_n.json   Cross-check: counterexample counts
    table2_random_trees.json     Cross-check: random model rates
figures/
  fiedler_bfs_tree.pdf          Figure 1 (BFS-tree structure)
  olep_counterexamples.pdf      Figure 2 (CE1, CE2)
```

---

## Key numbers

| Quantity | Value |
|---|---|
| Total non-isomorphic trees enumerated (n=5..17) | 81,132 |
| Trees with simple λ₂ (spectral gap > 10⁻⁸) | 81,099 |
| OLEP counterexamples | **830** (first at n=13) |
| Smallest counterexample size | n=13 (two trees) |
| All counterexamples have | D=5 (8) or D≥6 (822) |
| Caterpillar OLEP failures | 0 (proved in paper) |

---

## OLEP definition

```
Spectral center  c = argmin |f(v)|  (ties by vertex index)
V+ = {v : f(v) > 0},   V- = {v : f(v) < 0}
K+ = max BFS-depth(c, v)  over all V+-leaves
K- = max BFS-depth(c, v)  over all V--leaves
F+ = {V+-leaves at depth K+}
F- = {V--leaves at depth K-}
OLEP holds  iff  global-max(f) ∈ F+  AND  global-min(f) ∈ F-
```

---

## Citation

If you use this code, please cite:

```
Gang Li. The Outer-Leaf Extremal Property of the Fiedler Vector on Trees.
Discrete Applied Mathematics, submitted 2026.
```
