# Frank–Wolfe Variants for the LASSO Problem

[![Course](https://img.shields.io/badge/Course-Optimization%20for%20Data%20Science-blue.svg)](https://www.unipd.it)
[![Institution](https://img.shields.io/badge/Institution-University%20of%20Padua-red.svg)](https://www.unipd.it)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

This project investigates projection-free first-order optimization algorithms for the $\ell_1$-constrained LASSO problem, developed for the **Optimization for Data Science** course (M.Sc. in Data Science, University of Padua, A.Y. 2025/2026).

* **Authors:** Alessandro Di Frenna, Tamari Matikashvili
* **Supervisors:** Prof. Francesco Rinaldi, Prof. Gennaro Auricchio

---

## Overview

The goal is to solve the constrained LASSO formulation:

$$\min_{\beta \in \mathbb{R}^n} \frac{1}{2m} \|X\beta - y\|_2^2 \quad \text{subject to} \quad \|\beta\|_1 \le t$$

Standard projected gradient descent requires an explicit projection onto the $\ell_1$-norm ball at each iteration. In contrast, the **Frank–Wolfe (conditional gradient)** family solves a simple Linear Minimization Oracle (LMO) over the polytope at each step, yielding sparse iterates as convex combinations of extreme points (atoms).

This project implements and empirically benchmarks three Frank–Wolfe variants from scratch:
1. **Classic Frank–Wolfe (FW)** with exact line search.
2. **Away-Step Frank–Wolfe (AFW)** to prevent zig-zagging jamming near boundary faces.
3. **Pairwise Frank–Wolfe (PFW)** for simultaneous mass transfer between active atoms.

---

## Repository Structure

```text
LASSO Optimization/
├── optimization.ipynb    # Complete implementation, experiments, and visualizations
├── Optimization.pdf      # Detailed technical paper and theoretical derivations
└── README.md             # Project overview and summary
```

* **`optimization.ipynb`**: Contains clean NumPy implementations of the algorithms, parameter tuning for the constraint budget $t$, convergence diagnostics (duality gap and objective value tracking), and sparsity evaluations across all datasets.
* **`Optimization.pdf`**: The full report detailing mathematical derivations (LMO, exact line search, away/pairwise step limits), theoretical convergence rates, and extensive experimental analysis.

---

## Implemented Algorithms

| Algorithm | Key Mechanism | Step Size Strategy | Zig-Zagging Mitigation |
| :--- | :--- | :--- | :--- |
| **Classic FW** | Moves toward the LMO vertex: $d = s - \beta$ | Exact line search: $\alpha \in [0, 1]$ | None (suffers from slow sublinear convergence) |
| **Away-Step FW** | Dynamically chooses between FW direction or moving away from the worst active atom ($d = \beta - v$) | Exact line search bounded by $\gamma_{\max}$ | Yes (drops ineffective active atoms) |
| **Pairwise FW** | Directly shifts weight from the worst atom $v$ to the best vertex $s$: $d = s - v$ | Exact line search bounded by $\gamma_{\max} = \alpha_v$ | Yes (faster weight exchange, preserves total mass) |

---

## Benchmark Datasets

The algorithms were evaluated on both low- and high-dimensional regression and classification tasks:

1. **Concrete Compressive Strength** (UCI): Low-dimensional regression ($m = 1{,}030$, $n = 8$).
2. **California Housing** (scikit-learn): Moderate-scale regression ($m = 20{,}640$, $n = 8$).
3. **CT Slice Localization** (UCI): High-dimensional regression ($m = 53{,}500$, $n = 384$).
4. **Dorothea** (UCI): Very high-dimensional sparse binary classification ($m = 800$, $n = 100{,}000$ features with 50% distractor noise probes).

---

## Key Experimental Results

* **Convergence Speed:** Classic FW failed to achieve the target duality gap tolerance ($10^{-8}$) within the 5,000-iteration budget on any dataset due to severe zig-zagging. Both Away-step and Pairwise FW successfully certified convergence.
* **Pairwise FW Dominance:** Pairwise FW was consistently the fastest method, requiring **$2\times$ to $33\times$ fewer iterations** than Classic FW and achieving the lowest wall-clock runtime.
* **Solution Consistency:** All three variants converged to virtually identical objective values, test metrics ($\text{MSE}$, $R^2$, and $F_1$), and sparsity patterns, confirming the correctness of the iterate trajectories.
* **Extreme Sparsity on Dorothea:** Out of 100,000 features, the constrained optimization retained fewer than **65 active features**, demonstrating effective recovery of true predictive signals while discarding distractor probes.

---

## Setup and Usage

### Prerequisites

Install the required dependencies via `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib ucimlrepo
```

### Running the Notebook

Open and run `optimization.ipynb` in your Jupyter environment or Google Colab:

```bash
jupyter notebook optimization.ipynb
```
