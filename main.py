import gymnasium as gym
import numpy as np
import os
import math
import random
import argparse
from collections import deque, namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


# --- parametros ---
replay_size=100_000
seed = 42
alpha = 0.1 # learning rate
gamma = 0.99 # discount factor
epsilon = 0.2 # exploration rate
num_episodes = 1000
max_steps_per_episode = 1000
learning_rate=1e-3
eps_start=1.0
eps_end=0.05
eps_decay_steps=50_000
max_steps_per_ep=1000
batch_size=128
target_update_hard=2000
target_update_tau=0.005
eval_episodes=300
eval_render_mode="human"

#
def seed_everything():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))


#Neural Network
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.net(state)

def soft_update(target_net, online_net, tau=1.0):
    #mezclar los parametros de las dos redes
    for tgt_p, src_p in zip(target_net.parameters(), online_net.parameters()):
        tgt_p.data.copy_(tau * src_p.data + (1.0 - tau) * tgt_p.data)

def optimize(qnet, tgt, opt, batch, gamma, device):
    # batch: Transition of tuples
    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    
    q_values = qnet(states).gather(1, actions)

    with torch.no_grad():
        next_max_q = tgt(next_states).max(1, keepdim=True)[0]
        target = rewards + (1.0 - dones) * gamma * next_max_q

    loss = nn.SmoothL1Loss()(q_values, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
    opt.step()
    return loss.item()


#Training
def make_env(render_mode=None, seed=42):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    env.reset(seed=seed)
    return env

#select the max action
def select_action(qnet, state, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = qnet(s)
        return int(q.argmax(dim=1).item())

def moving_average(x, w):
    if len(x) < 1:
        return np.array([])
    w = max(1, w)
    return np.convolve(x, np.ones(w), 'valid') / w 

def evaluate(env, qnet, episodes=10, device="cpu"):
    rewards = []
    for ep in range(episodes):
        s, _ = env.reset()
        ep_r = 0.0
        done = False
        steps = 0
        while not done and steps < 2000:
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(qnet(s_t).argmax(dim=1).item())
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_r += r
            steps += 1
        rewards.append(ep_r)
    return float(np.mean(rewards))

def train():
    seed_everything()
    #entrenar sin render
    train_env = make_env(render_mode=None, seed=seed)
    obs_dim = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.n
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    qnet = QNetwork(obs_dim, n_actions).to(device) #entrenamiento politica
    tgt = QNetwork(obs_dim, n_actions).to(device) #entrenamiento politica
    soft_update(tgt, qnet, tau=1.0)

    opt = optim.Adam(qnet.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(replay_size)
    
    eps_decay_steps = max(1, 50_000)

    total_steps = 0
    rewards_hist = []
    losses_hist = []

    pbar = trange(num_episodes, desc="Entrenando", ncols=100)
    for ep in pbar:
        state, _ = train_env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        steps_ep = 0

        while not done and steps_ep < max_steps_per_ep:
            epsilon = eps_end + (eps_start - eps_end) * math.exp(-1.0 * total_steps / eps_decay_steps)
            action = select_action(qnet, state, epsilon, n_actions, device)

            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            ep_reward += reward
            total_steps += 1
            steps_ep += 1

            # Entrenamiento
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                loss = optimize(qnet, tgt, opt, batch, gamma, device)
                losses_hist.append(loss)

            # Actualización de la red objetivo (dura cada N pasos o soft cada paso)
            if target_update_hard > 0:
                if total_steps % target_update_hard == 0:
                    soft_update(tgt, qnet, tau=1.0)
            else:
                soft_update(tgt, qnet, tau=target_update_tau)

        rewards_hist.append(ep_reward)
        avg100 = np.mean(rewards_hist[-100:]) if len(rewards_hist) >= 10 else np.mean(rewards_hist)
        pbar.set_postfix(env="LunarLander-v3", R=ep_reward, avg100=round(avg100, 1), eps=round(epsilon, 3))

    train_env.close()

     # Guardar modelo
    torch.save({"model_state_dict": qnet.state_dict(),
                "obs_dim": obs_dim,
                "n_actions": n_actions}, "dqn_lunarlander.pt")

    plt.figure(figsize=(8, 4.5))
    plt.plot(rewards_hist, label="Reward por episodio")
    if len(rewards_hist) >= 50:
        plt.plot(range(50-1, len(rewards_hist)),
                 moving_average(np.array(rewards_hist), 50),
                 label=f"Media móvil (w={50})")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Convergencia DQN - LunarLander")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)

    # Evaluación opcional
    if eval_episodes > 0:
        eval_render = eval_render_mode  # "human" local; "rgb_array" en Colab
        eval_env = make_env(render_mode=eval_render, seed=42 + 999)
        avg_eval = evaluate(eval_env, qnet, 10, device=device)
        eval_env.close()
        print(f"Reward promedio en evaluación ({10} eps): {avg_eval:.2f}")

train()