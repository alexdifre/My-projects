# Deep Learning Assignments (DATA20046)

This repository contains the implementations, experimental investigations, and empirical findings for the assignments of the **Neural Networks and Deep Learning (DATA20046)** course. All implementations are developed in **PyTorch**, utilizing the classic **MNIST** handwritten digits dataset as the primary benchmark across all tasks.

---

## Repository Structure

```text
├── tasks.pdf                # Official assignment specifications
├── Tasks/
│   ├── results_1/           # Task 1: Supervised Learning (Notebooks, weights, and detailed README)
│   ├── results_2/           # Task 2: Double Descent Phenomenon
│   ├── results_3/           # Task 3: Transfer Learning & Domain Shift
│   ├── results_4/           # Task 4: Autoencoders with 2D Bottleneck
│   └── results_5/           # Task 5: Generative Adversarial Networks (Iterative progression)
└── README.md                # Project overview and summary
```

---

## Tasks Summary and Overview

### Task 0: Environment Setup
* **Task Objectives & Scope:** Configure the PyTorch development environment, download the MNIST dataset, implement data inspection routines, and establish modular data partitions (training, validation, and test sets).
* **Approach & Implemented Work:** The pipeline was established using standard PyTorch utilities and custom DataLoaders. A modular data-loading and partitioning strategy was built and shared as a consistent foundation for all subsequent tasks.

---

### Task 1: Supervised Learning Basics
* **Task Objectives & Scope:** Train supervised feed-forward classifiers on MNIST digits using two distinct architectures, establish a train/validation/test split, tune optimization hyperparameters, demonstrate failure modes from poor hyperparameter choices, and evaluate final test performance.
* **Approach & Implemented Work:** Implemented and benchmarked a shallow baseline against a regularized deep network, conducted structured hyperparameter tuning, and performed an ablation study to decouple the effects of network depth from regularization.
* **Detailed Report:** For a comprehensive, step-by-step breakdown of the experimental progression, ablation studies, and methodological reasoning, please see the dedicated [Task 1 README](Tasks/results_1/README.md).

---

### Task 2: Double Descent
* **Task Objectives & Scope:**
  - Investigate and empirically replicate the *Double Descent* phenomenon on MNIST, referencing the literature (arXiv:2303.14151) and course lecture material.
  - Examine the transition between the classical under-parameterized regime and the modern over-parameterized regime, identifying the critical interpolation threshold where test error typically displays an anomalous peak.
  - Systematically vary model and training properties (such as network width/capacity, training epochs, and sample size) and draw critical conclusions from the observed behaviors.

* **Approach & Implemented Work:**
  - Set up an experimental suite varying network capacity through hidden layer width (*hidden width*), spanning from narrow under-parameterized models to heavily over-parameterized networks.
  - Explored three complementary experimental dimensions:
    1. **Model Capacity:** Evaluated training and test error trajectories across model widths, inspecting the boundary where training error approaches zero (the interpolation threshold).
    2. **Training Epochs:** Monitored error curves over extended training durations.
    3. **Dataset Size:** Assessed the influence of training sample volume on test error behavior and generalization.
  - Critically discussed the characteristics of the interpolation transition on MNIST, highlighting how the clarity and magnitude of the double descent peak depend heavily on data conditioning and model architecture.

---

### Task 3: Transfer Learning
* **Task Objectives & Scope:**
  - Utilize a pre-trained MNIST digit classifier (from Task 1) as the source model.
  - Define a target task with a pronounced *domain shift* (e.g., transforming images via rotation or inversion, or modifying class subsets), assuming only a limited budget of training samples is available for the target task.
  - Select and justify a suitable transfer learning method (e.g., fine-tuning pre-trained representations).
  - Evaluate and compare three distinct experimental setups:
    - **(a)** The original model evaluated directly on the target task without retraining (zero-shot baseline / domain shift assessment).
    - **(b)** The same model trained from scratch solely on the limited target training set.
    - **(c)** The transfer learning solution adapting source knowledge to the new task.

* **Approach & Implemented Work:**
  - Reused the pre-trained `DeepNet` architecture from Task 1 as the source feature extractor.
  - Formulated the target domain by applying a 90-degree image rotation to MNIST digits and constraining the training set size to simulate a low-data regime.
  - Verified the severity of the domain shift, observing that the unadapted source model collapses to near-chance accuracy (~10-14%).
  - Compared the model trained from scratch against the fine-tuned transfer model, confirming that transfer learning delivers faster convergence and substantially higher classification accuracy under constrained sample sizes.

---

### Task 4: Autoencoders
* **Task Objectives & Scope:**
  - Train a dimension-reducing autoencoder featuring an extreme 2-dimensional latent bottleneck ($m = 2$).
  - Design and compare two distinct symmetric architectures (preserving identical depth between encoder and decoder), justifying architectural choices including layer dimensions, activations, and reconstruction loss.
  - Project 500 randomly selected test digits onto the 2D latent plane to evaluate cluster separation and qualitative reconstruction quality.

* **Approach & Implemented Work:**
  - Implemented two symmetric fully-connected autoencoder architectures with a 2-unit latent bottleneck, utilizing ReLU intermediate activations and an appropriate reconstruction loss.
  - Isolated a fixed subset of 500 test images to ensure an objective, uniform visual comparison across both architectures.
  - Generated 2D latent space scatter plots color-coded by digit class, analyzing geometric clustering, class overlap within the compressed space, and the visual fidelity of reconstructed digits.

---

### Task 5: Generative Models (GAN)
* **Task Objectives & Scope:**
  - Design and train a Generative Adversarial Network (GAN) to synthesize realistic MNIST-like digits.
  - Implement balanced, lightweight architectures for both Generator and Discriminator, managing typical adversarial optimization challenges (loss balancing, training stability, mode collapse).
  - Showcase at least 10 generated samples from the final model and provide a critical discussion on generation quality, convergence behavior, and potential failure factors.

* **Approach & Implemented Work:**
  - Implemented a GAN comprising a **Generator** (mapping a 100-dimensional latent noise vector to $28 \times 28$ images with `tanh` output activation) and a **Discriminator** (regularized with *Dropout* to prevent premature dominance over the generator).
  - Refined the training dynamics through five progressive experimental iterations:
    1. *Initial Setup*: Rapid baseline on 12,000 samples for 40 epochs to validate architectural and loss dynamics.
    2. *Dataset Expansion*: Scaled training data to 30,000 samples to provide richer gradient information.
    3. *Epoch Scaling*: Increased training to 60 epochs, monitoring generator and discriminator loss curves.
    4. *Extended Optimization*: Extended training to 100 epochs, achieving stable equilibrium (with discriminator accuracy hovering around a balanced ~55%).
    5. *Full Dataset Training*: Trained on the complete MNIST training set to maximize sample sharpness and structural coherence.
  - Visualized and analyzed synthetic outputs across all stages, tracking the transition from noisy textures to well-defined, recognizable handwritten digits.
