# train_knapsack_ddqn.py
"""
Reproduce RL experiments for Knapsack (0-1 and bounded) using Dueling DQN + Noisy layers,
plus baselines (DP optimal, greedy). Outputs training logs and evaluation metrics.

Usage:
    python train_knapsack_ddqn.py
"""

import math
import time
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm

# -------------------------
# Config / Hyperparameters
# -------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 100  # episodes
EVAL_EPISODES = 200  # number of test instances for evaluation (to compute accuracy avg)

# Environment parameters (matches paper style)
TRAIN_EPISODES = 4000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
REPLAY_SIZE = 50000
TARGET_UPDATE_FREQ = 1000  # steps
MIN_REPLAY_SIZE = 1000
MAX_STEPS_PER_EP = 1000

# Network params
HIDDEN_DIM = 256

# For experiments: change N_ITEMS etc.
N_ITEMS = 20  # default for training; can test on 10,20,30
CAPACITY_FACTOR_01 = 30  # paper: cw = n*30 for baseline 0-1
CAPACITY_FACTOR_BOUNDED = 80  # paper for 2D-BKP: cw = n*80
# -------------------------

# -------------------------
# Utilities: DP & Greedy
# -------------------------
def dp_2d_knapsack(values, weights, volumes, cw, cv, bounded_counts=None):
    """
    DP for 2D knapsack (pseudo-polynomial). If bounded_counts is None => 0-1 knapsack.
    Returns optimal value.
    NOTE: This is O(n * cw * cv). Use only for small n or small capacities.
    """
    n = len(values)
    # We'll implement DP by iterating items and capacities; to save memory we use 2D table
    # But beware large cw*cv -> memory explosion.
    W = int(cw)
    V = int(cv)
    # initialize 2D array
    dp = np.zeros((W+1, V+1), dtype=np.float32)
    for i in range(n):
        val = values[i]
        w = int(weights[i])
        vol = int(volumes[i])
        if bounded_counts is None:
            # 0-1: iterate backwards
            for a in range(W, w-1, -1):
                row = dp[a - w]
                # update dp[a][b] for all b
                # vectorized over volume dimension
                dp_a = dp[a]
                candidate = row.copy()
                # shift along volume
                candidate_shift = np.zeros_like(candidate)
                if vol <= V:
                    candidate_shift[vol:] = row[:-vol]
                    candidate_shift += val
                    dp[a] = np.maximum(dp_a, candidate_shift)
        else:
            # bounded: treat like multiple 0-1 by binary splitting or naive loop of count (simple)
            count = int(bounded_counts[i])
            # naive repeated 0-1 for count times
            for _ in range(count):
                for a in range(W, w-1, -1):
                    row = dp[a - w]
                    dp_a = dp[a]
                    candidate_shift = np.zeros_like(dp_a)
                    if vol <= V:
                        candidate_shift[vol:] = row[:-vol]
                        candidate_shift += val
                        dp[a] = np.maximum(dp_a, candidate_shift)
    return float(dp.max())

def greedy_knapsack(values, weights):
    """
    Greedy for 1D knapsack ratio by value/weight (ignore volume here).
    Returns selected total value.
    """
    idx = np.argsort(- (np.array(values) / (np.array(weights) + 1e-9)))
    total_w = 0
    total_v = 0
    for i in idx:
        # For greedy baseline we don't consider volume in simplest variant; but we can check capacity outside
        pass
    # We'll not use this function directly; below in env we implement greedy that respects cw and cv
    return None

# -------------------------
# Environment
# -------------------------
class KnapsackEnv:
    """
    Deterministic sequential environment: at step t, agent picks an index among remaining items.
    State: (n+1) x 4 matrix flattened:
      - selection status xi (0/1)
      - normalized value p'_i = p_i / sum(p)
      - normalized weight w'_i = w_i / cw
      - normalized volume v'_i = v_i / cv
    final row is remaining capacity (weight_ratio, volume_ratio, 0,0) for convenience.
    Action: pick index i (0..n-1). If pick infeasible -> reward 0 and state unchanged (or we can mark as invalid).
    Episode ends when no more feasible picks or all items selected.
    """
    def __init__(self, n_items, cw, cv, bounded_counts=None):
        self.n = n_items
        self.cw = cw
        self.cv = cv
        self.bounded = bounded_counts is not None
        self.bounded_counts = None if bounded_counts is None else bounded_counts.copy()
        # placeholders
        self.weights = None
        self.volumes = None
        self.values = None
        self.total_value = None
        self.selected = None
        self.rem_w = None
        self.rem_v = None

    def sample_instance(self, value_dist=(1,100), weight_dist=(1,100), volume_dist=(1,100), bounded_max=5):
        # generate random instance
        self.values = np.random.randint(value_dist[0], value_dist[1]+1, size=self.n).astype(np.float32)
        self.weights = np.random.randint(weight_dist[0], weight_dist[1]+1, size=self.n).astype(np.int32)
        self.volumes = np.random.randint(volume_dist[0], volume_dist[1]+1, size=self.n).astype(np.int32)
        if self.bounded:
            if self.bounded_counts is None:
                self.bounded_counts = np.random.randint(1, bounded_max+1, size=self.n).astype(np.int32)
            else:
                # if provided from constructor, keep it
                pass
        else:
            self.bounded_counts = None
        self.total_value = float(self.values.sum())
        self.selected = np.zeros(self.n, dtype=np.int8)
        self.rem_w = int(self.cw)
        self.rem_v = int(self.cv)
        return self._get_state()

    def _get_state(self):
        # Build (n+1) x 4
        state = np.zeros((self.n+1, 4), dtype=np.float32)
        # selection status
        state[:self.n,0] = self.selected
        # normalized value
        if self.total_value <= 0:
            state[:self.n,1] = 0.0
        else:
            state[:self.n,1] = self.values / (self.total_value + 1e-9)
        # normalized weight/volume
        state[:self.n,2] = self.weights / (self.cw + 1e-9)
        state[:self.n,3] = self.volumes / (self.cv + 1e-9)
        # last row: remaining capacity ratios
        state[self.n,0] = 0
        state[self.n,1] = 0
        state[self.n,2] = self.rem_w / (self.cw + 1e-9)
        state[self.n,3] = self.rem_v / (self.cv + 1e-9)
        return state.flatten()

    def valid_actions_mask(self):
        mask = np.zeros(self.n, dtype=np.bool_)
        for i in range(self.n):
            if self.bounded:
                if self.bounded_counts[i] <= 0:
                    mask[i] = False
                    continue
            else:
                if self.selected[i] == 1:
                    mask[i] = False
                    continue
            # check capacity
            if self.weights[i] <= self.rem_w and self.volumes[i] <= self.rem_v:
                mask[i] = True
            else:
                mask[i] = False
        return mask

    def step(self, action):
        """
        action: index to select
        returns: next_state, reward, done, info
        """
        mask = self.valid_actions_mask()
        if action < 0 or action >= self.n:
            # invalid -> no-op
            return self._get_state(), 0.0, True, {}
        if not mask[action]:
            # infeasible pick: reward 0, but we keep going; in paper reward=0 for infeasible
            return self._get_state(), 0.0, False, {}
        # feasible: add item
        reward = float(self.values[action])
        # update capacities and selected
        self.rem_w -= int(self.weights[action])
        self.rem_v -= int(self.volumes[action])
        if self.bounded:
            self.bounded_counts[action] -= 1
            # selection status not just binary: we keep counts as selected times (but state uses binary xi per paper)
            # we'll still keep selected mark to 1 if any taken
            if self.bounded_counts[action] < 0:
                self.bounded_counts[action] = 0
            if self.bounded_counts[action] == 0:
                self.selected[action] = 1
        else:
            self.selected[action] = 1
        done = not mask.any()  # no more feasible actions
        return self._get_state(), reward, done, {}

    def greedy_solution_value(self):
        # Greedy that respects both weight and volume: use value/(w+v) ratio
        idx = np.argsort(- (self.values / (self.weights + self.volumes + 1e-9)))
        rem_w = int(self.cw)
        rem_v = int(self.cv)
        total = 0.0
        counts = None if not self.bounded else self.bounded_counts.copy()
        for i in idx:
            if self.bounded:
                while counts[i] > 0 and self.weights[i] <= rem_w and self.volumes[i] <= rem_v:
                    total += float(self.values[i])
                    rem_w -= int(self.weights[i])
                    rem_v -= int(self.volumes[i])
                    counts[i] -= 1
            else:
                if self.weights[i] <= rem_w and self.volumes[i] <= rem_v:
                    total += float(self.values[i])
                    rem_w -= int(self.weights[i])
                    rem_v -= int(self.volumes[i])
        return total

# -------------------------
# Noisy linear layer
# -------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1. / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(input, weight, bias)

# -------------------------
# Dueling DQN Network with Noisy layers
# -------------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, n_actions, use_noisy=True, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.use_noisy = use_noisy
        lin = NoisyLinear if use_noisy else nn.Linear
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # value stream
        self.value_layer = nn.Sequential(
            lin(hidden_dim, hidden_dim),
            nn.ReLU(),
            lin(hidden_dim, 1)
        )
        # advantage stream
        self.adv_layer = nn.Sequential(
            lin(hidden_dim, hidden_dim),
            nn.ReLU(),
            lin(hidden_dim, n_actions)
        )

    def forward(self, x):
        x = self.fc(x)
        v = self.value_layer(x)
        a = self.adv_layer(x)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        if not self.use_noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

# -------------------------
# Replay buffer
# -------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Agent: Dueling DQN with Noisy exploration
# -------------------------
class Agent:
    def __init__(self, state_dim, n_actions, use_noisy=True, lr=LR):
        self.n_actions = n_actions
        self.net = DuelingDQN(state_dim, n_actions, use_noisy=use_noisy).to(DEVICE)
        self.target = DuelingDQN(state_dim, n_actions, use_noisy=use_noisy).to(DEVICE)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.use_noisy = use_noisy
        self.step_count = 0

    def act(self, state, mask=None):
        # state: numpy array flattened
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            q = self.net(s).cpu().numpy()[0]  # raw Qs
        self.net.train()
        # apply mask: set invalid actions to very low q
        if mask is not None:
            q = np.array(q)
            invalid = ~mask
            q[invalid] = -1e9
        action = int(np.argmax(q))
        return action

    def update(self, batch: Transition):
        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # compute current Q
        q_values = self.net(states).gather(1, actions)  # (batch,1)

        # Double-DQN target
        with torch.no_grad():
            # online selects action
            next_q_online = self.net(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states).gather(1, next_actions)
            target_q = rewards + GAMMA * next_q_target * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # reset noise if used
        if self.use_noisy:
            self.net.reset_noise()
            self.target.reset_noise()
        return loss.item()

    def sync_target(self):
        self.target.load_state_dict(self.net.state_dict())

# -------------------------
# Training & Evaluation loops
# -------------------------
def train(env_generator, n_items=N_ITEMS, cw=None, cv=None, bounded=False,
          train_episodes=TRAIN_EPISODES, use_noisy=True):
    state_dim = (n_items + 1) * 4
    n_actions = n_items
    agent = Agent(state_dim, n_actions, use_noisy=use_noisy)
    buffer = ReplayBuffer()
    total_steps = 0
    losses = []
    reward_history = []

    # prefill replay
    while len(buffer) < MIN_REPLAY_SIZE:
        # sample random episode
        env = env_generator()
        s = env.sample_instance()
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_EP:
            mask = env.valid_actions_mask()
            # choose random valid action or random action
            valid_idxs = np.where(mask)[0]
            if len(valid_idxs) == 0:
                break
            a = int(np.random.choice(valid_idxs))
            ns, r, done, _ = env.step(a)
            buffer.push(s, a, r, ns, float(done))
            s = ns
            steps += 1
        # allow multiple random episodes to fill
    print(f"Replay buffer prefilled: {len(buffer)}")

    pbar = trange(train_episodes)
    for ep in pbar:
        env = env_generator()
        state = env.sample_instance()
        ep_reward = 0.0
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_EP:
            mask = env.valid_actions_mask()
            # either pick via agent, but occasionally random for decorrelation
            if random.random() < 0.05:
                valid_idxs = np.where(mask)[0]
                if len(valid_idxs) == 0:
                    a = 0
                else:
                    a = int(np.random.choice(valid_idxs))
            else:
                a = agent.act(state, mask=mask)
                # if chosen invalid (rare due to mask handled), pick random valid
                if not mask[a]:
                    valid_idxs = np.where(mask)[0]
                    if len(valid_idxs) == 0:
                        a = 0
                    else:
                        a = int(np.random.choice(valid_idxs))
            next_state, reward, done, _ = env.step(a)
            buffer.push(state, a, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            steps += 1
            total_steps += 1

            # training step
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = agent.update(batch)
                losses.append(loss)

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.sync_target()

        reward_history.append(ep_reward)
        if (ep+1) % PRINT_EVERY == 0:
            avg_recent = np.mean(reward_history[-PRINT_EVERY:])
            pbar.set_description(f"ep {ep+1} avgR {avg_recent:.2f}")
    return agent

def evaluate(agent, env_generator, n_items, cw, cv, bounded=False, episodes=EVAL_EPISODES, compare_dp=True):
    accs = []
    greedy_vals = []
    agent_vals = []
    dp_vals = []
    runtimes_dp = []
    runtimes_agent = []
    for _ in range(episodes):
        env = env_generator()
        state = env.sample_instance()
        # compute optimal via DP (may be expensive)
        values = env.values.copy()
        weights = env.weights.copy()
        volumes = env.volumes.copy()
        bounded_counts = None if not bounded else env.bounded_counts.copy()
        if compare_dp:
            t0 = time.time()
            try:
                opt = dp_2d_knapsack(values, weights, volumes, env.cw, env.cv, bounded_counts=bounded_counts)
            except Exception as e:
                # DP failed (too heavy) -> skip dp
                opt = None
            t1 = time.time()
            runtimes_dp.append(t1-t0)
        else:
            opt = None
        # greedy
        greedy_val = env.greedy_solution_value()
        greedy_vals.append(greedy_val)
        # agent rollout (greedy by Q)
        t0 = time.time()
        total = 0.0
        max_steps = 1000
        steps = 0
        while steps < max_steps:
            mask = env.valid_actions_mask()
            if not mask.any():
                break
            a = agent.act(env._get_state(), mask=mask)
            ns, r, done, _ = env.step(a)
            total += r
            steps += 1
            if done:
                break
        t1 = time.time()
        runtimes_agent.append(t1-t0)
        agent_vals.append(total)
        if opt is None or opt <= 0:
            acc = float(total) / (greedy_val + 1e-9)  # fallback
        else:
            acc = float(total) / (opt + 1e-9)
        accs.append(acc)
        if opt is not None:
            dp_vals.append(opt)
    results = {
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'agent_mean_value': float(np.mean(agent_vals)),
        'agent_std_value': float(np.std(agent_vals)),
        'greedy_mean_value': float(np.mean(greedy_vals)),
        'dp_mean_value': float(np.mean(dp_vals)) if len(dp_vals)>0 else None,
        'runtime_agent_mean': float(np.mean(runtimes_agent)),
        'runtime_dp_mean': float(np.mean(runtimes_dp)) if len(runtimes_dp)>0 else None
    }
    return results

# -------------------------
# Helper env generator closure
# -------------------------
def make_env_generator(n_items, cw, cv, bounded=False):
    def _gen():
        return KnapsackEnv(n_items=n_items, cw=cw, cv=cv, bounded_counts=(None if not bounded else None))
    return _gen

# -------------------------
# Main runnable example
# -------------------------
if __name__ == "__main__":
    # Choose scenario: standard 0-1 knapsack or bounded
    scenario = "0-1"  # choose from "0-1" or "bounded"
    n_items = 20
    if scenario == "0-1":
        cw = cv = n_items * CAPACITY_FACTOR_01
        bounded = False
    else:
        cw = cv = n_items * CAPACITY_FACTOR_BOUNDED
        bounded = True

    print("Device:", DEVICE)
    env_gen = make_env_generator(n_items, cw, cv, bounded=bounded)

    # Train agent (DDQN-Noisy)
    print("Training Dueling DQN with Noisy layers...")
    agent = train(env_gen, n_items=n_items, cw=cw, cv=cv, bounded=bounded,
                  train_episodes=TRAIN_EPISODES, use_noisy=True)

    # Evaluate
    print("Evaluating agent vs greedy and DP...")
    results = evaluate(agent, env_gen, n_items, cw, cv, bounded=bounded, episodes=EVAL_EPISODES, compare_dp=True)

    print("Results summary:")
    for k,v in results.items():
        print(f"  {k}: {v}")
