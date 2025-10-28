# Autonomous Driving Car project

This is the repository for the Autonomous Driving project of the Reinforcement Learning course.

The goal of your agent will be to drive an Autonomous Vehicle through an highway, taking into consideration the presence of other vehicles. For this project you will use the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) library.

<img src="https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway_fast_dqn.gif"/>

This project implements and analyzes different **Reinforcement Learning (RL)** approaches to solve the `highway-v0` environment from the `gymnasium` package.  
The goal is to evaluate and compare various agents in terms of **reward**, **success rate**, and **stability** within a simulated autonomous driving scenario.

---

## Project Overview

The study includes:
- A **Baseline** control model
- Two Deep Q-Network (DQN) variants:
  - Standard DQN
  - DQN with manual feature selection
- A **Tabular Q-Learning** agent

Each model is trained and evaluated under consistent conditions to ensure fair comparison.

---
## Repository Structure
```
├── training/
│   ├── DQN.py                    # Deep Q-Network agent implementation
│   ├── DQN_feat_select.py        # DQN with manual feature selection
│   └── utilities/                # Helper functions and utilities
│       ├── get_bins_state.py     # State space discretization
│       ├── map2States.py         # State mapping functions
│       ├── ttc.py                # Time-to-collision calculations
│       └── __init__.py
│
├── baseline.py                   # Baseline model implementation
├── evaluate.py                   # Evaluation and comparison script
├── Q_learn.py                    # Tabular Q-Learning agent
├── results/                      # Evaluation results + weight's model
└── autonomous_driving.pdf        # Technical paper 
```
A comprehensive explanation of the technical intricacies can be found in the associated paper: **RL for autonomous driving.pdf**

---

## Dependencies

### Core Requirements
```bash
pip install numpy gymnasium torch matplotlib tqdm
```

### Recommended Tools
```bash
pip install seaborn pandas
```

**Minimum Versions:**
- Python 3.8+
- PyTorch 1.10+
- Gymnasium 0.26+
- NumPy 1.21+

---

## Usage

### Quick Start

To evaluate all trained models and generate comparison results:
```bash
python evaluate.py
```

### Training Individual Agents

**Deep Q-Network (DQN):**
```bash
python training/DQN.py
```

**DQN with Feature Selection:**
```bash
python training/DQN_feat_select.py
```

**Tabular Q-Learning:**
```bash
python Q_learn.py
```

**Baseline Model:**
```bash
python baseline.py
```

All trained models and performance logs are automatically saved in the `results/` directory.

---

## Results and Evaluation

The project provides comprehensive quantitative comparison of all implemented agents based on:

- **Average Episode Reward** - Cumulative reward per episode
- **Success Rate** - Percentage of successful task completions
- **Training Stability** - Convergence behavior across epochs
- **Sample Efficiency** - Learning speed and data requirements

Detailed experimental analysis, methodology, and performance plots are available in the accompanying technical paper: **`autonomous_driving.pdf`**

---

## Environment specifications

### State space:
The state space consists in a `V x F` array that describes a list of `V = 5` vehicles by a set of features of size 
`F = 5`.

The features for each vehicle are:
- Presence (boolean value)
- Normalized position along the x axis w.r.t. the ego-vehicle
- Normalized position along the y axis w.r.t. the ego-vehicle
- Normalized velocity along the x axis w.r.t. the ego-vehicle
- Normalized velocity along the y axis w.r.t. the ego-vehicle

***Note:*** the first row contains the features of the ego-vehicle, which are the only ones referred to the absolute reference frame.

### Action space
The action space is discrete, and it contains 5 possible actions:
  - Change lane to the left
  - Idle
  - Change lane to the right
  - Go faster
  - Go slower

### Reward function
The reward function is a composition of various terms:
- Bonus term for progressing quickly on the road
- Bonus term for staying on the rightmost lane
- Penalty term for collisions

## Baselines
For this project, you will need to compare the performances of your agent against the baseline you define