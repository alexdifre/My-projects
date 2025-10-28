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

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done))
    def __len__(self):
        return len(self.buffer)
    
# 🔹 Epsilon-greedy decay
def epsilon_by_step(step):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY)
    
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, d=128):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
        
            nn.Linear(state_dim, d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d, n_actions)
        )
    
    def forward(self, x):
        return self.fc(x)

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
        "absolute": False,
    },
    
        # PARAMETRI DI CONTROLLO VEICOLI
        
        # PARAMETRI DI RICOMPENSA
        'reward_speed_range': [22, 23], # Gamma di velocità in cui la ricompensa è massima.
        'success_goal_reward': 0.1,     # Piccola ricompensa per aver completato lo scenario.
        'off_road_terminal': True,      # Termina se l'agente esce di strada.

        # PESI DELLA RICOMPENSA
        'weights': {
        'collision': -10.0,     # penalità forte per incidenti
        'on_road': 0.3,         # premia stare in carreggiata
        'high_speed': 8.0,      # grande incentivo ad andare veloce
        'acceleration': 1.5,    # premia l'uso dell'acceleratore
        'jerk': -0.5,           # penalizza cambi di accelerazione troppo bruschi
        'right_lane': 0.05,     # piccolo premio per stare a destra
        'lane_change': -0.2,    # penalità per zig-zag
    },
        
        # GENERALI
        'duration': 40,
        'lanes_count': 3,
        "vehicles_count": 5
}
    return config
    
if __name__ == "__main__":

    TRAIN_FREQ = 4  # Allenarsi ogni N step invece che ogni step
    LOSS_WINDOW = 100  # Finestra per smoothing della loss

    # (in, 128)-(128-128)-(128, out) con Dropout
    GAMMA = 0.95  # Slightly lower for faster convergence
    LR = 5e-4     # Reduced learning rate
    BATCH_SIZE = 32  # Smaller batch for more frequent updates
    BUFFER_SIZE = 50_000
    TARGET_UPDATE = 1000  # Much more frequent target updates
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 5_000  # steps
    MAX_STEPS = int(20_000)


    env = gymnasium.make("highway-v0",config= configuration(None)) 
    state, _ = env.reset()
    state_dim = state.reshape(-1).shape[0]
    n_actions = env.action_space.n


    # 🔹 Initialize networks and optimizer
    policy_net = QNetwork(state_dim, n_actions)
    target_net = QNetwork(state_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)


    # 🔹 Directory per salvataggi
    SAVE_DIR = "results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 🔹 Metriche MIGLIORATE
    episode_rewards = []
    episode_lengths = []
    episode_losses = []  # Loss media per episodio, calcolata correttamente
    epsilons = []
    all_losses = []  # Tutte le loss per smoothing
    smoothed_losses = []  # Loss con media mobile

    # 🔹 Training Loop
    state = state.reshape(-1)
    done, truncated = False, False
    episode, episode_steps, episode_return = 1, 0, 0
    episode_loss_sum = 0.0
    episode_training_steps = 0

    for t in range(MAX_STEPS):
        episode_steps += 1
        epsilon = epsilon_by_step(t)

        # Epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state))
                action = q_values.argmax().item()

        # Step
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = next_state.reshape(-1)

        # Replay buffer
        replay_buffer.push(state, action, reward, next_state, done or truncated)

        state = next_state
        episode_return += reward

        # Training con frequenza ridotta
        if len(replay_buffer) > BATCH_SIZE and t % TRAIN_FREQ == 0:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            
            # Forward pass
            q_values = policy_net(states)
            state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target values con Double DQN (opzionale miglioramento)
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0]
                expected_values = rewards + GAMMA * next_q_values * (1 - dones)

            # Loss con clipping per stabilità
            loss = nn.MSELoss()(state_action_values, expected_values)
            
            # Gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
            optimizer.step()
            
            # Raccolta metriche corrette
            current_loss = loss.item()
            all_losses.append(current_loss)
            episode_loss_sum += current_loss
            episode_training_steps += 1
            
            # Smoothed loss con finestra mobile
            if len(all_losses) >= LOSS_WINDOW:
                smoothed_loss = np.mean(all_losses[-LOSS_WINDOW:])
                smoothed_losses.append(smoothed_loss)

        # Update target network
        if t % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Fine episodio
        if done or truncated:
            # Calcola loss media CORRETTA per questo episodio
            if episode_training_steps > 0:
                episode_mean_loss = episode_loss_sum / episode_training_steps
            else:
                episode_mean_loss = 0.0
                
            episode_rewards.append(episode_return)
            episode_lengths.append(episode_steps)
            episode_losses.append(episode_mean_loss)
            epsilons.append(epsilon)

            print(f"Step {t} | Ep {episode} | Steps {episode_steps} | Return {episode_return:.2f} "
                f"| Loss {episode_mean_loss:.4f} | Eps {epsilon:.3f} | Training Steps: {episode_training_steps}")

            # Reset per nuovo episodio
            state, _ = env.reset()
            state = state.reshape(-1)
            episode += 1
            episode_steps, episode_return = 0, 0
            episode_loss_sum = 0.0
            episode_training_steps = 0

    # 🔹 Salvataggio finale dei pesi
    torch.save(policy_net.state_dict(), os.path.join(SAVE_DIR, "dqn_highway.pth"))
    print("✅ Pesi salvati in results/dqn_highway.pth")

    # 🔹 Salvataggio metriche in CSV
    df = pd.DataFrame({
        "episode": range(1, len(episode_rewards) + 1),
        "reward": episode_rewards,
        "length": episode_lengths,
        "loss": episode_losses,
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

    # Subplot 2: Loss per episodio (corretta)
    plt.subplot(2, 3, 2)
    valid_losses = [l for l in df["loss"] if l > 0]  # Escludi episodi senza training
    if valid_losses:
        plt.plot(range(1, len(valid_losses) + 1), valid_losses, alpha=0.7, label="Loss per Episode")
        if len(valid_losses) > 10:
            window_loss = min(20, len(valid_losses) // 5)
            rolling_loss = pd.Series(valid_losses).rolling(window=window_loss, center=True).mean()
            plt.plot(range(1, len(valid_losses) + 1), rolling_loss, 'r-', linewidth=2, label=f'Moving Average ({window_loss})')
    plt.xlabel("Episode (with training)")
    plt.ylabel("Mean Loss")
    plt.title("Training Loss per Episode")
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


    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_plots.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Grafici migliorati salvati in results/training_plots.png")

    env.close()