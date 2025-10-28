import gymnasium
import highway_env
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from training.DQN_feat_select import QNetwork
from training.DQN_feat_select import configuration
import random
from utilities.map2States import map2States

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

MODEL_PATH = "results/dqn_highway.pth"
N_EPISODES_TEST = 50   # numero episodi di valutazione
CSV_PATH = "results/evaluation_metrics.csv"

config = configuration(None)
# 🔹 Crea env con rendering (puoi anche mettere "rgb_array" per registrare video)
env = gymnasium.make("highway-v0",render_mode="human",config = config)

# 🔹 Carica modello
state, _ = env.reset()
state_dim = 11  # Numero di feature dopo l'estrazione
n_actions = env.action_space.n

policy_net = QNetwork(state_dim, n_actions)
policy_net.load_state_dict(torch.load(MODEL_PATH))
policy_net.eval()

# 🔹 Liste metriche
episode_rewards = []
episode_lengths = []
successes = []
dt = 1 / env.unwrapped.config["simulation_frequency"]

# 🔹 Loop di test
for ep in range(N_EPISODES_TEST):
    state, _ = env.reset()
    state = map2States.features_extraction(state, env, dt)

    done, truncated = False, False
    total_reward, steps = 0, 0
    success = 1   # assumiamo successo, diventa 0 se collisione
    

    base_env = env.unwrapped  # accesso diretto a vehicle e road
    obs, info = base_env.reset()


    while not (done or truncated):
        with torch.no_grad():
            q_values = policy_net(torch.FloatTensor(state))
            action = q_values.argmax().item()

        next_state, reward, done, truncated, info = env.step(action)
        next_state = map2States.features_extraction(next_state, env, dt)

    
        total_reward += reward
        steps += 1
        state = next_state

        # 👇 se c’è collisione, fallimento
        if "crashed" in info and info["crashed"]:
            success = 0

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    successes.append(success)

    print(f"[TEST] Ep {ep+1}/{N_EPISODES_TEST} | Reward: {total_reward:.2f} | Steps: {steps} | Success: {success}")

env.close()


# 🔹 Analisi statistica
mean_episode = np.mean(episode_lengths)
std_episode = np.std(episode_lengths)

mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
se = std_reward / np.sqrt(N_EPISODES_TEST)                # errore standard
# intervallo di confidenza 95%
ci = stats.norm.interval(0.95, loc=mean_reward, scale=se)

success_rate = np.mean(successes) * 100

print("\n📊 Risultati valutazione:")
print(f"episode medio: {mean_episode:.2f} ± {std_episode:.2f}")
print(f"Reward medio: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"Success rate: {success_rate:.1f}%")
print(f"Reward medio: {mean_reward:.2f},  "
      f"(IC 95%: {ci[0]:.2f} – {ci[1]:.2f})")

