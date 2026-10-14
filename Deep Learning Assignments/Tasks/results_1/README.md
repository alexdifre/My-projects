# Task 1: Supervised Learning — Methodological Rationale & Experimental Progression

This document provides an in-depth breakdown of the conceptual framework, iterative methodology, and analytical reasoning implemented across the three notebooks in **Task 1** (`Task_1_A.ipynb`, `Task_1_B.ipynb`, and `Task_1_C.ipynb`).

---

## 1. Scientific Motivation & Core Hypothesis

The overarching goal of Task 1 is to train and evaluate supervised neural network classifiers on the classic MNIST dataset. Beyond basic training, the core scientific inquiry investigates:
1. **Model Capacity**: How network depth and parameter count influence representational expressiveness.
2. **Regularization**: How techniques such as Batch Normalization, Dropout, and Weight Decay govern generalization and prevent overfitting.

### The Methodological Confounding Problem
In a standard baseline comparison between a simple network (**ShallowNet**) and a regularized deep network (**DeepNet**), two primary independent variables are altered simultaneously:
* **Structural capacity** (1 hidden layer vs. 3 hidden layers; ~100k parameters vs. ~500k+ parameters).
* **Regularization mechanisms** (unregularized vs. Batch Normalization + Dropout).

If `DeepNet` achieves higher test accuracy, a naive comparison cannot determine whether the improvement stems from greater architectural capacity, the regularization techniques, or a combination of both. Resolving this confounding factor forms the foundational narrative across the three notebooks.

---

## 2. Experimental Progression Across Notebooks

### 2.1. `Task_1_A.ipynb`: Architecture Design, Hyperparameter Tuning, and Critical Self-Evaluation

#### Data Partitioning & Strict Reproducibility
* The official 60,000-sample training dataset is partitioned into an **80% training subset** (48,000 samples) and a **20% validation subset** (12,000 samples).
* A fixed random seed (`SEED = 42`) and a dedicated PyTorch generator (`torch.Generator().manual_seed(42)`) ensure identical sample splits across all runs, eliminating data partition variance as a source of performance discrepancy.

#### Architectures
* **`ShallowNet`**: An intentionally minimal single-hidden-layer baseline:
  $$\text{Input } (784) \longrightarrow \text{Linear}(784, 128) \longrightarrow \text{ReLU} \longrightarrow \text{Linear}(128, 10)$$
* **`DeepNet`**: A 3-hidden-layer architecture regularized via Batch Normalization and Dropout ($p = 0.2$):
  $$\text{Input } (784) \longrightarrow \text{Linear}(512) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.2) \longrightarrow \text{Linear}(256) \to \dots \to \text{Linear}(128) \to \dots \to \text{Linear}(10)$$

#### Hyperparameter Search Strategy
Hyperparameters were tuned sequentially using the Adam optimizer across 10-epoch validation runs:
1. **Learning Rate**: A grid of values was tested. While $lr = 0.1$ caused complete divergence for `ShallowNet`, $lr = 0.001$ emerged as the optimal rate for both networks.
2. **Batch Size**: Evaluated across $\{32, 64, 128\}$. Although a batch size of 128 offered slightly smoother gradients, $bs = 32$ achieved virtually identical validation accuracy while offering greater computational efficiency and regularization benefits from stochastic gradient noise.
3. **Weight Decay (on DeepNet)**: Tested over 30 epochs across $\{0.0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$. The optimal value was found to be $0.0$, indicating that the combination of Batch Normalization and Dropout already provided sufficient regularization without requiring an explicit $L_2$ penalty.

#### Exploratory Hypothesis Test: Weight Decay on ShallowNet
An exploratory side experiment was conducted to test whether adding weight decay to `ShallowNet` across extended training horizons (10, 15, and 25 epochs) would delay or prevent overfitting.
* **Observation**: Non-zero weight decay failed to improve validation performance.
* **Analytical Takeaway**: `ShallowNet` has limited intrinsic capacity relative to the 48,000 MNIST training samples. The network did not suffer from substantial late-stage parameter divergence; imposing weight decay acted as an unnecessary restriction on capacity, hindering the model from capturing the underlying data manifold.

#### Critical Methodological Critique Identified in `Task_1_A`
Two methodological shortcomings were recognized and documented upon concluding `Task_1_A`:
1. **Evaluation Inconsistency**: Validation metrics were collected over 10-epoch tuning trials, whereas test accuracy was evaluated after 40 epochs of retraining on the full 60,000-image dataset. Direct comparisons between validation and test scores were therefore structurally misaligned.
2. **Unresolved Confounding Factor**: The performance gap between `ShallowNet` and `DeepNet` could still not be causally attributed to regularization rather than depth.

#### Failure Mode Demonstration (*Bad Hyperparameters*)
To satisfy the course requirement of showing poor hyperparameter choices, `ShallowNet` was trained with $lr = 0.1$. The model completely failed to converge, remaining trapped at random-guess accuracy (~10%), demonstrating that the shallower network is acutely vulnerable to excessive step sizes.

---

### 2.2. `Task_1_B.ipynb`: Establishing Methodological Rigor (40-Epoch Validation)

`Task_1_B.ipynb` was designed specifically to rectify the first methodological limitation discovered in `Task_1_A`:
* Rather than comparing different epoch budgets, both selected models (`ShallowNet` and `DeepNet` with $lr=0.001$, $bs=32$, $wd=0.0$) were trained for **40 complete epochs directly on the 80/20 train/validation split**.
* Full training and validation loss and accuracy curves were tracked and plotted epoch-by-epoch.
* This standardized the training trajectory, allowing an honest observation of convergence dynamics and late-stage generalization gaps before moving to final evaluation.

---

### 2.3. `Task_1_C.ipynb`: The Ablation Study & Final Scientific Synthesis

To resolve the core confounding problem (capacity vs. regularization), `Task_1_C.ipynb` introduced an **ablation model**:

#### The Tripartite Model Suite
1. **`ShallowNet`**: Baseline model (Low Capacity, Unregularized).
2. **`DeepNet_2`**: Ablated deep model with identical 3-layer architecture (`784 -> 512 -> 256 -> 128 -> 10`), but with **all Batch Normalization and Dropout layers removed** (High Capacity, Unregularized).
3. **`DeepNet`**: Full deep model with identical layer structure plus **Batch Normalization and Dropout** (High Capacity, Regularized).

This three-way design enables two orthogonal, clean comparisons:
* **`ShallowNet` vs. `DeepNet_2`**: Isolates the pure effect of architectural capacity/depth.
* **`DeepNet_2` vs. `DeepNet`**: Isolates the pure effect of regularization under identical capacity.

---

## 3. Quantitative Results & Comparative Analysis

### 3.1. Performance on the 80/20 Split (40 Epochs)

| Model | Architecture | Regularization | Training Accuracy | Validation Accuracy | Test Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`ShallowNet`** | 1 Hidden Layer (128) | None | 99.74% | 97.66% | 97.80% |
| **`DeepNet_2`** | 3 Hidden Layers | None | **99.88%** | 98.07% | 98.03% |
| **`DeepNet`** | 3 Hidden Layers | BN + Dropout (0.2) | 99.05% | **98.43%** | **98.29%** |

### 3.2. Final Retraining on Full Training Set (60,000 samples, 40 Epochs)

| Model | Final Test Accuracy |
| :--- | :---: |
| **`ShallowNet`** | 98.00% |
| **`DeepNet_2`** | 98.03% |
| **`DeepNet`** | **98.44%** |

---

## 4. Key Takeaways & Scientific Inferences

### 1. The Role of Architectural Capacity (`ShallowNet` vs. `DeepNet_2`)
Increasing depth from 1 to 3 hidden layers without any regularization raised validation accuracy from 97.66% to 98.07%. This confirms that the deeper representation provides superior feature abstraction capabilities, allowing the network to better separate digit classes.

### 2. The Definitive Role of Regularization (`DeepNet_2` vs. `DeepNet`)
The comparison between the two identical deep architectures provides textbook empirical evidence of regularization in action:
* **On the training set**: The unregularized `DeepNet_2` achieves near-perfect memorization (99.88%), whereas regularized `DeepNet` is constrained to 99.05%.
* **On validation and test sets**: Despite lower training accuracy, `DeepNet` systematically outperforms `DeepNet_2` across both validation (98.43% vs. 98.07%) and test metrics (98.29% vs. 98.03% on split; 98.44% vs. 98.03% on full dataset).
* **Inference**: Regularization prevents the network from fitting non-generalizable sample-specific idiosyncrasies, successfully converting raw parameter capacity into true out-of-sample generalization.

### 3. Dataset Context & Final Evaluation
Because MNIST is a relatively clean and high-sample benchmark, even unregularized deep networks do not experience catastrophic overfitting. Nevertheless, the empirical margin achieved by `DeepNet` demonstrates that regularization remains indispensable for maximizing test generalization and stabilizing deep multi-layer models.

---

## 5. Saved Artifacts

* `weights_shallow.pth`: Serialized PyTorch state dictionary for the optimal `ShallowNet`.
* `weights_deep.pth`: Serialized PyTorch state dictionary for the optimal regularized `DeepNet`.
