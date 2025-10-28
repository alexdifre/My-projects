# Frank-Wolfe Algorithms for the Maximum Clique Problem
## Overview

This project implements and benchmarks several **Frank-Wolfe (FW)** optimization variants to solve the continuous relaxation of the **NP-hard Maximum Clique Problem**. A Projected Gradient Descent (PGD) baseline is included for comparison.

## Algorithms Implemented

- **Standard Frank-Wolfe**
- **Away-Step Frank-Wolfe (AFW)**
- **Pairwise Frank-Wolfe (PFW)**
- **Projected Gradient Descent (PGD)** (Baseline)

A comprehensive explanation of the technical intricacies can be found in the associated paper: **Maximum Clique Optimization.pdf**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Algorithms Implemented](#algorithms-implemented)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Installation and Usage](#installation-and-usage)
5. [Experimental Results](#experimental-results)
6. [Performance Analysis](#performance-analysis)
7. [Conclusions](#conclusions)

---

## Problem Statement

### The Maximum Clique Problem

Given an undirected graph G = (V, E), a **clique** is a subset of vertices C ⊆ V where every pair of vertices is connected by an edge. The **Maximum Clique Problem (MCP)** seeks to find the largest such subset.

**Formal Definition:**
```
maximize |C| such that C is a clique in G
```

**Complexity:** NP-hard

**Applications:**
- Social network analysis (community detection)
- Computational biology (protein complex identification)
- Scheduling and resource allocation
- Pattern recognition and computer vision

### Continuous Relaxation

We reformulate the discrete MCP as a continuous optimization problem over the probability simplex:

```
max_{x ∈ Δ} x^T A x + Φ(x)
```

where:
- **Δ** = {x ∈ ℝ^n : x_i ≥ 0, Σx_i = 1} (probability simplex)
- **A** is the adjacency matrix of the graph
- **Φ(x)** is a regularization term to avoid poor local optima

This formulation enables the application of first-order optimization methods while maintaining the ability to recover high-quality discrete cliques.

---

### Line Search Strategy

Instead of fixed step-size, we implement exact line search:

```
γ_t = arg min_{γ ∈ [0,1]} f(x_t + γ(s_t - x_t))
```

This adaptive approach improves convergence in practice across different problem instances.

---

## Experimental Results

We tested all algorithms on three benchmark datasets from the DIMACS clique benchmark library:

| Dataset | Vertices | Edges | Known Max Clique |
|---------|----------|-------|------------------|
| brock200_2 | 200 | 9,876 | 12 |
| brock800_2 | 800 | 207,643 | 24 |
| C125.9 | 125 | 6,963 | 34 |


#### Convergence Speed

| Dataset | FW Time | AFW Time | PFW Time | PGD Time |
|---------|---------|----------|----------|----------|
| brock200_2 | **0.02s** | 0.03s | 0.04s | 0.05s |
| brock800_2 | **0.20s** | 0.25s | 0.35s | 0.40s |
| C125.9 | **0.01s** | 0.02s | 0.03s | 0.03s |

**Observations:**
- FW achieved highest objective values in shortest time across all datasets
- AFW showed rapid early movement but stagnated quickly
- PFW displayed more oscillations during convergence
- PGD remained at consistently lower objective levels

---

## Performance Analysis


### Algorithm Characteristics

| Variant | Mass Removal | Sparse Updates | Linear Convergence | Best For |
|---------|-------------|----------------|-------------------|----------|
| FW | No | Yes | No | General problems, consistent quality |
| AFW | Yes | Yes | Yes (conditional) | Boundary solutions, theoretical interest |
| PFW | Yes | Yes | Yes (conditional) | ML applications, sparse solutions |
| PGD | N/A | No | No | Baseline comparison |

### Solution quality

**Frank-Wolfe:**
- Most robust across all test cases
- Excellent scalability to large graphs
- Interpretable solution structure
- Slower theoretical convergence (O(1/t))

**Away-Step FW:**
- Theoretical linear convergence
- Extreme sparsity
- Premature convergence in practice
- Poor solution diversity

**Pairwise FW:**
- Balance between sparsity and quality
- Efficient weight transfer
- Numerical instability (negative values)
- More oscillations than FW

**Projected Gradient Descent:**
- Predictable behavior
- Simple implementation
- Poor performance on MCP
- Expensive projection operations

---

## Conclusions

### Key Takeaways

1. **Standard Frank-Wolfe is the clear winner** for the Maximum Clique Problem, providing the best combination of speed, stability, and solution quality.

2. **Projection-free methods are highly effective** on structured constraints like the simplex, avoiding expensive projection operations.

3. **Theoretical guarantees don't always translate to practice**: AFW's linear convergence guarantee was overshadowed by its tendency to converge to poor local optima.

4. **Line search significantly improves performance** compared to fixed step-sizes, especially on diverse problem instances.

5. **Continuous relaxation provides meaningful solutions** despite the discrete nature of the original problem.



## References

### Core Publications

1. Jaggi, M. (2013). "Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization." *ICML*.

2. Lacoste-Julien, S., & Jaggi, M. (2015). "On the Global Linear Convergence of Frank-Wolfe Optimization Variants." *NIPS*.

3. Bomze, I. M., Budinich, M., Pardalos, P. M., & Pelillo, M. (1999). "The Maximum Clique Problem." *Handbook of Combinatorial Optimization*.

### Benchmark Datasets

DIMACS Clique Benchmark Library: [http://dimacs.rutgers.edu/Challenges/](http://dimacs.rutgers.edu/Challenges/)

---

### Authors

- **Alessandro Giuffrè** (alessandro.giuffre@studenti.unipd.it)
- **Vishwa Mittar** (vishwa.mittar@studenti.unipd.it)
- **Jaime Candau Otero** (jaime.candauotero@studenti.unipd.it)
- **Alessandro Di Frenna** (alessandro.difrenna@studenti.unipd.it)
