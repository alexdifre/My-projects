import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import os
from utilities.get_bins_state import ego_lane,states_x_ax_pos, states_x_ax_neg, states_y_ax_pos, states_y_ax_neg, time_collision
from utilities.map2States import map2States

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# (in, 128)-(128-128)-(128, out) con Dropout
GAMMA = 0.95  # Slightly lower for faster convergence
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 30_000  # steps
MAX_STEPS = int(40_000)
alpha = 0.1  # learning rate
all_bins = [
np.array(ego_lane()),
np.array([0]+ states_x_ax_pos()),
np.array([0]+ states_x_ax_neg()),
np.array(states_x_ax_neg()+[0]+states_x_ax_pos()),
np.array(states_x_ax_neg()+[0]+states_x_ax_pos()),
np.array([0]+ states_y_ax_pos()),
np.array(states_y_ax_neg()+[0]),
np.array(states_y_ax_neg()+[0]+states_y_ax_pos()),
np.array(states_y_ax_neg()+[0]+states_y_ax_pos()),
np.array([0]+ time_collision()),
np.array([0]+ time_collision())
]
def epsilon_by_step(step):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY)
    
def configuration(none):
    config = {
        
    'action': {
        'type': 'DiscreteMetaAction'
    },

    'observation': {
        "type": "Kinematics",
        "vehicles_count": 10,
        "lanes_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
                "x": [-100, 100],
                "y": [-50, 50],
        },
        "absolute": False,
    },
    
        'off_road_terminal': False,      # Termina se l'agente esce di strada.

        # PESI DELLA RICOMPENSA
        'weights': {
        'collision': -5.0,     # penalità forte per incidenti
    },
        
        # GENERALI
        'duration': 40,
        'lanes_count': 5,
        "vehicles_count": 10
}
    return config

# 🔹 Directory per salvataggi
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Parametri
GAMMA = 0.95
ALPHA = 0.1  # Learning rate per Q-learning
MAX_STEPS = 20_000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10_000

# Metriche
episode_rewards = []
episode_lengths = []
epsilons = []

# Inizializzazione ambiente
env = gymnasium.make("highway-v0", config=configuration(None))
obs, _ = env.reset()
dt = 1 / env.unwrapped.config["simulation_frequency"]
obj = map2States(env)

state_continuous = obj.features_extraction(obs, env, dt)
d = 11
# Q-table: converti stato multidimensionale in indice unico
num_bins_per_feature = obj.num_bins_per_feature
num_actions = env.action_space.n
num_states = np.prod(num_bins_per_feature)
Q = np.zeros((num_states, num_actions))

# Estrai feature iniziali

# Training loop
episode = 1
episode_steps = 0
episode_return = 0
done = False
truncated = False


state_visit_counts = np.zeros(num_states)


episode_rewardsss = []
episode_lengthsss = []
successes = []
for t in range(MAX_STEPS+50):
    epsilon = epsilon_by_step(t)
    
    done, truncated = False, False
    total_reward, steps = 0, 0
    success = 1   # assumiamo successo, diventa 0 se collisione
    
    bin_list = [0] * len(state_continuous)  # inizializza la lista dei bin
    for i in range(len(all_bins)):
        bin_list[i] = obj.map_to_bins(state_continuous[i], all_bins[i])
    state_index = obj.combine_bins(bin_list)
    
    # Epsilon-greedy action selection
    if np.random.rand() < epsilon and t < MAX_STEPS:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state_index])
    
    # Step nell'ambiente
    next_obs, reward, done, truncated, info = env.step(action)
    
    # Estrai feature del next state
    next_state_continuous = map2States.features_extraction(next_obs, env, dt)
    next_bin_list = [0] * len(state_continuous)  
    for i in range(len(all_bins)):
        next_bin_list[i] = obj.map_to_bins(next_state_continuous[i], all_bins[i])
    next_state_index = obj.combine_bins(next_bin_list)
    
    
    state_visit_counts[state_index] += 1
    
    
    # Q-learning update
    best_next_action = np.argmax(Q[next_state_index])
    td_target = reward + GAMMA * Q[next_state_index, best_next_action]
    td_error = td_target - Q[state_index, action]
    Q[state_index, action] += ALPHA * td_error
    
    # Aggiorna stato e metriche
    state_continuous = next_state_continuous
    episode_return += reward
    episode_steps += 1
    
    # Fine episodio
    if done or truncated:
        episode_rewards.append(episode_return)
        episode_lengths.append(episode_steps)
        epsilons.append(epsilon)
        
        print(f"Step {t} | Ep {episode} | Steps {episode_steps} | "
              f"Return {episode_return:.2f} | Eps {epsilon:.3f}")
        
        # Reset ambiente
        obs, _ = env.reset()
        state_continuous = map2States.features_extraction(obs, env, dt)
        episode += 1
        episode_steps = 0
        episode_return = 0
        done = False
        truncated = False

    if t > MAX_STEPS:

        total_reward += reward
        steps += 1
        state = next_state_index

        # 👇 se c’è collisione, fallimento
        if "crashed" in info and info["crashed"]:
            success = 0

        episode_rewardsss.append(total_reward)
        episode_lengthsss.append(steps)
        successes.append(success)
  


df = pd.DataFrame({
    "episode": range(1, len(episode_rewards) + 1),
    "reward": episode_rewards,
    "length": episode_lengths,
    "epsilon": epsilons
})


# 🔹 Plot metriche MIGLIORATI
plt.figure(figsize=(20, 10))

# Subplot 1: Rewards
plt.subplot(2, 3, 1)
plt.plot(df["episode"], df["reward"], alpha=0.7, label="Episode Return")
# Media mobile per smoothing
window = min(50, len(df) // 10)
if window > 1:
    rolling_mean = df["reward"].rolling(window=window, center=True).mean()
    plt.plot(df["episode"], rolling_mean, 'r-', linewidth=2, label=f'Moving Average ({window})')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Rewards")
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: distribuzione esplorazione degli stati in training
plt.subplot(2, 3, 2)
num_visited = np.sum(state_visit_counts > 0)
num_unvisited = num_states - num_visited
percentages = [num_visited / num_states * 100, num_unvisited / num_states * 100]
labels = [f'Esplorati\n({num_visited})', f'Non esplorati\n({num_unvisited})']
colors = ['#2ecc71', '#e74c3c']

plt.pie(percentages, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title(f'Copertura spazio degli stati\n(Tot: {num_states} stati)')
plt.title("distribuzione esplorazione degli stati")
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 3: Epsilon decay
plt.subplot(2, 3, 3)
plt.plot(df["episode"], df["epsilon"], label="Epsilon", color='orange')
plt.xlabel("Episode")
plt.ylabel("ε")
plt.title("Epsilon Decay")
plt.grid(True, alpha=0.3)
plt.legend()

print(f"\n{'='*50}")
print(f"STATISTICHE ESPLORAZIONE STATI")
print(f"{'='*50}")
print(f"Stati totali:              {num_states}")
print(f"Stati visitati:            {num_visited} ({percentages[0]:.2f}%)")
print(f"Stati non visitati:        {num_unvisited} ({percentages[1]:.2f}%)")
print(f"Visite medie per stato:    {state_visit_counts.mean():.2f}")
print(f"Stato più visitato:        {state_visit_counts.max():.0f} visite")
print(f"{'='*50}\n")


plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_plots.png"), dpi=300, bbox_inches='tight')
plt.show()
print("✅ Grafici migliorati salvati in results/training_plots.png")


# 🔹 Analisi statistica
mean_episode = np.mean(episode_lengthsss)
std_episode = np.std(episode_lengthsss)

import scipy.stats as stats
mean_reward = np.mean(episode_rewardsss)
std_reward = np.std(episode_rewardsss)
se = std_reward / np.sqrt(50)                # errore standard
# intervallo di confidenza 95%
ci = stats.norm.interval(0.95, loc=mean_reward, scale=se)

success_rate = np.mean(successes) * 100

print("\n📊 Risultati valutazione:")
print(f"episode medio: {mean_episode:.2f} ± {std_episode:.2f}")
print(f"Reward medio: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"Success rate: {success_rate:.1f}%")
print(f"Reward medio: {mean_reward:.2f},  "
      f"(IC 95%: {ci[0]:.2f} – {ci[1]:.2f})")



env.close()

Risultati valutazione:
episode medio: 1.00 ± 0.00
Reward medio: 0.81 ± 0.15
Success rate: 98.0%
Reward medio: 0.81,  (IC 95%: 0.77 – 0.85)