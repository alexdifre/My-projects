# A Two-Stage Distillation Framework for Mitigating Shortcut Learning via Decoupled Representation and Classifier Training

---

## Introduction
Knowledge distillation is a common technique for model compression, but student models can inherit shortcuts from their teachers.  
While methods like **Contrastive Representation Distillation (CRD)** align feature spaces, the learning process often remains entangled with the final classification objective.  

Deep learning models, given their high parametric capacity, are particularly susceptible to **shortcut learning** — the tendency to exploit spurious correlations within training data to minimize loss without capturing true causal relationships.  
This results in poor generalization on unseen data and a higher test error.
This paper explores a **two-stage distillation strategy** that decouples representation learning from classifier training to analyze its effects on generalization.

In the **first phase**, we use CRD to align hierarchical feature representations between a teacher and student model **without training the student’s classifier**.  
In the **second phase**, the feature extractor is frozen, and the classifier is trained independently using **Kullback-Leibler (KL) divergence** on the teacher’s soft logits.
Our proposal builds on the hypothesis that **decoupling representation learning from classifier optimization** can mitigate this issue and lead to more robust student models.

We evaluate this modular approach on the **CIFAR-100 dataset** using various **ResNet architectures**.  
Our experiments show that this decoupled strategy is effective, yielding improved generalization and stable training.  

A statistical analysis of the distilled model’s predictions reveals distinct error patterns compared to the teacher, suggesting that the student learns different inductive biases.  
These findings indicate that separating feature and classifier learning is a viable and robust framework for knowledge distillation, offering a structured alternative to end-to-end training.

---

A comprehensive explanation of the technical intricacies can be found in the associated paper: **Knowledge Distillation with Decoupled Learning.pdf**

---
## Repository Structure

```
├── analysis_results/         # Experimental results 
│   └── ...                   # Evaluation metrics, plots, and logs
│
├── CKA/                      # Centered Kernel Alignment analysis
│   └── ...                   # Similarity metrics 
│
├── dataset/                  # Data loaders and utilities
│   └── ...                     
│
├── diagnostics/              # Model diagnostics and debugging tools
│   └── ...                   
│
├── estimator/                # kraskov estimator 
│   └── ...                     
│
├── losses/                   # Custom loss functions
│   └── ...                   # Loss implementations and variants
│
├── main_functions/           # Core utility functions
│   └── ...                    
│
├── models/                   # Neural network architectures
│   └── ...                   # Model definitions and components
│
├── training/                 # Training pipeline and configurations
│   └── ...                  
│
├── .DS_Store                   # macOS system file (ignored)
├── __init__.py                 # Package initialization
├── Knowledge_Distillation...   # paper
├── main.py                     # Main entry point for training
├── main_eval.py                # Evaluation script for trained models
└──README.md                   
```

A comprehensive explanation of the technical intricacies can be found in the associated paper: **Knowledge Distillation with Decoupled Learning.pdf**

---

## Methodology

### Overview
The proposed **two-stage distillation framework** separates the learning of **representations** and **classifiers** into distinct phases:

1. **Phase 1 — Representation Alignment:**  
   The student’s encoder learns to align its intermediate feature representations with those of the teacher using **Contrastive Representation Distillation (CRD)**.  
   No classification head is trained at this stage, ensuring that the student focuses exclusively on semantic representation alignment.

2. **Phase 2 — Classifier Training:**  
   The trained feature extractor is **frozen** (or fine-tuned with a very low learning rate), and a new classifier head is trained using **KL divergence** and **Cross Entropy** between the teacher’s and student’s soft logits.  
   This ensures that the classifier learns from informative probability distributions rather than hard labels.

This decomposition reduces interference between representation and classifier optimization and results in more stable, interpretable learning dynamics.

---

## Experimental Setup

### Dataset
We evaluate our framework on **CIFAR-100**, a benchmark dataset containing 100 object categories with 600 images each (500 for training, 100 for testing).

### Architectures
We experiment with various **ResNet** teacher–student pairs, ensuring that the teacher has a larger representational capacity than the student:
- Teacher: `ResNet-34`, `ResNet-50`
- Student: `ResNet-18`, `ResNet-20`

### Training Configuration
| Phase | Objective | Loss Function | Optimizer | Learning Rate |
|--------|------------|---------------|------------|----------------|
| Phase 1 | Representation Alignment | Contrastive Loss (CRD) | SGD + Momentum | 0.05 |
| Phase 2 | Classifier Training | KL Divergence + Cross Entropy | Adam | 0.001 |

Batch normalization and data augmentation (random crop, flip) are applied consistently.

---

## Results and Analysis

### Quantitative Results
| Teacher | Student | Distillation Type | Accuracy (%) | Improvement |
|----------|----------|------------------|---------------|--------------|
| ResNet-34 | ResNet-18 | Baseline (E2E KD) | 74.3 | – |
| ResNet-34 | ResNet-18 | Two-Stage (CRD + KL) | **76.1** | +1.8 |
| ResNet-50 | ResNet-20 | Baseline (E2E KD) | 71.8 | – |
| ResNet-50 | ResNet-20 | Two-Stage (CRD + KL) | **73.7** | +1.9 |

Our two-stage approach consistently improves accuracy while reducing variance across runs, confirming that separating representation and classification training yields more robust models.

### Statistical Insights
- **Prediction Diversity:** The student produces error patterns that differ from those of the teacher, suggesting a shift in inductive biases rather than simple imitation.  
- **Representation Quality:** t-SNE visualization of feature embeddings reveals higher intra-class compactness and better inter-class separation after Phase 1.


---

## Future Work
Potential extensions include:
- Applying the framework to **larger datasets** (ImageNet) and **transformer architectures**.  
- Investigating **adaptive phase transitions** (automatically determining when to switch from CRD to KL).  
- Combining with **stochastic regularization** or **causal representation learning** to further mitigate shortcut reliance.

---

## References
1. Ting Chen et al. *“A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)”*, ICML 2020.  
2. Geirhos et al. *“Shortcut Learning in Deep Neural Networks”*, Nature Machine Intelligence 2020.  
3. Hinton, G. et al. *“Distilling the Knowledge in a Neural Network”*, arXiv:1503.02531 (2015).  
4. Tian et al. *“Contrastive Representation Distillation”*, ICLR 2020.  

---

## Authors
**Alessandro Di Frenna**  
University of Padua  
[alessandro.difrenna@studenti.unipd.it](mailto:alessandro.difrenna@studenti.unipd.it)  

**Margarita Shnaider**  
University of Padua  
[margarita.shnaider@studenti.unipd.it](mailto:margarita.shnaider@studenti.unipd.it)
