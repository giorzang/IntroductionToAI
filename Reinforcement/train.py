import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import time
import json
import pickle
from tqdm import tqdm

# Fix all random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

print("="*80)
print("DQN KNAPSACK TRAINING ON 10,000 RANDOM DATASETS")
print("="*80)

# ---------- 1) Environment ----------
class Knapsack1DEnv:
    def __init__(self, weights, values, capacity):
        self.weights = np.array(weights, dtype=float)
        self.values  = np.array(values,  dtype=float)
        self.capacity = float(capacity)
        self.n_items = len(weights)
        self.total_value = np.sum(self.values)
        self.reset()

    def reset(self):
        self.selected = np.zeros(self.n_items, dtype=int)
        self.remain_w = self.capacity
        return self._get_state()

    def _get_state(self):
        norm_w = self.weights / self.capacity
        norm_v = self.values  / self.total_value
        mat = np.zeros((self.n_items+1, 3), dtype=float)
        for i in range(self.n_items):
            mat[i] = [self.selected[i], norm_v[i], norm_w[i]]
        mat[self.n_items] = [1.0, 1.0, self.remain_w/self.capacity]
        return mat.flatten()

    def step(self, action):
        reward = 0.0
        if self.selected[action]==0 and self.remain_w>=self.weights[action]:
            self.selected[action] = 1
            self.remain_w -= self.weights[action]
            reward = float(self.values[action])
        
        next_state = self._get_state()
        can_fit = False
        for i in range(self.n_items):
            if self.selected[i] == 0 and self.remain_w >= self.weights[i]:
                can_fit = True
                break
        done = not can_fit
        return next_state, reward, done, {}

    def optimal_value(self):
        W = int(self.capacity)
        if W <= 0 or W > 10000:  # Prevent memory issues
            return 0
        dp = [0]*(W+1)
        for i in range(self.n_items):
            w, v = int(self.weights[i]), int(self.values[i])
            if w <= W and w > 0:
                for cap in range(W, w-1, -1):
                    dp[cap] = max(dp[cap], dp[cap-w] + v)
        return dp[W]

# ---------- 2) Dataset Generator ----------
def generate_random_datasets(n_datasets=10000, n_items_range=(10, 20)):
    """Generate diverse random knapsack datasets"""
    print(f"Generating {n_datasets} random datasets...")
    datasets = []
    
    for i in tqdm(range(n_datasets), desc="Generating datasets"):
        # Random number of items
        n_items = np.random.randint(n_items_range[0], n_items_range[1] + 1)
        
        # Random weights and values with different distributions
        if np.random.random() < 0.5:
            # Uniform distribution
            weights = np.random.randint(1, 101, size=n_items)
            values = np.random.randint(10, 201, size=n_items)
        else:
            # Normal distribution (clipped to positive)
            weights = np.clip(np.random.normal(30, 15, n_items).astype(int), 1, 100)
            values = np.clip(np.random.normal(100, 40, n_items).astype(int), 10, 200)
        
        # Capacity varies from 40% to 80% of total weight
        capacity_ratio = np.random.uniform(0.4, 0.8)
        capacity = int(np.sum(weights) * capacity_ratio)
        
        datasets.append({
            'weights': weights.tolist(),
            'values': values.tolist(), 
            'capacity': capacity,
            'n_items': n_items
        })
    
    print(f"Generated {len(datasets)} datasets")
    return datasets

# ---------- 3) Replay Buffer ----------
Transition = namedtuple('Transition', ('s','a','r','s2','done'))
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# ---------- 4) DQN Network ----------
class DQN(nn.Module):
    def __init__(self, state_dim, max_actions, hidden_size=128, n_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, max_actions))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ---------- 5) Training Function ----------
def train_dqn_on_datasets(datasets, episodes_per_dataset=5, max_items=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Network parameters
    max_state_dim = (max_items + 1) * 3  # Max possible state dimension
    model = DQN(max_state_dim, max_items, hidden_size=128, n_layers=3).to(device)
    target_model = DQN(max_state_dim, max_items, hidden_size=128, n_layers=3).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=50000)
    
    # Training parameters
    gamma = 0.99
    batch_size = 128
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = len(datasets) * episodes_per_dataset // 3
    
    steps_done = 0
    all_rewards = []
    dataset_rewards = []
    losses = []
    
    print(f"Training parameters:")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Episodes per dataset: {episodes_per_dataset}")
    print(f"  Total episodes: {len(datasets) * episodes_per_dataset}")
    print(f"  Epsilon decay: {eps_decay}")
    
    def get_epsilon(step):
        return eps_end + (eps_start - eps_end) * np.exp(-1. * step / eps_decay)
    
    def pad_state(state, target_dim):
        """Pad state to match network input dimension"""
        if len(state) < target_dim:
            padded = np.zeros(target_dim)
            padded[:len(state)] = state
            return padded
        return state[:target_dim]
    
    start_time = time.time()
    total_episodes = len(datasets) * episodes_per_dataset
    
    for dataset_idx, dataset in enumerate(tqdm(datasets, desc="Processing datasets")):
        env = Knapsack1DEnv(dataset['weights'], dataset['values'], dataset['capacity'])
        n_items = dataset['n_items']
        
        dataset_episode_rewards = []
        
        for episode in range(episodes_per_dataset):
            state = env.reset()
            state = pad_state(state, max_state_dim)
            total_reward = 0.0
            step_count = 0
            
            while step_count < n_items * 2:
                eps = get_epsilon(steps_done)
                
                if random.random() < eps:
                    action = random.randrange(n_items)
                else:
                    with torch.no_grad():
                        s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_vals = model(s_tensor)
                        # Only consider valid actions for this dataset
                        action = q_vals[0, :n_items].argmax().item()
                
                next_state, reward, done, _ = env.step(action)
                next_state = pad_state(next_state, max_state_dim)
                
                replay_buffer.push(state, action, reward, next_state, float(done))
                
                state = next_state
                total_reward += reward
                steps_done += 1
                step_count += 1
                
                # Training step
                if len(replay_buffer) >= batch_size and steps_done % 4 == 0:
                    transitions = replay_buffer.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    
                    states = torch.FloatTensor(batch.s).to(device)
                    actions = torch.LongTensor(batch.a).unsqueeze(1).to(device)
                    rewards = torch.FloatTensor(batch.r).unsqueeze(1).to(device)
                    next_states = torch.FloatTensor(batch.s2).to(device)
                    dones = torch.FloatTensor(batch.done).unsqueeze(1).to(device)
                    
                    q_values = model(states).gather(1, actions)
                    next_q = target_model(next_states).max(1)[0].detach().unsqueeze(1)
                    q_target = rewards + gamma * next_q * (1 - dones)
                    
                    loss = nn.MSELoss()(q_values, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    losses.append(loss.item())
                
                if done:
                    break
            
            dataset_episode_rewards.append(total_reward)
            all_rewards.append(total_reward)
        
        dataset_rewards.append(np.mean(dataset_episode_rewards))
        
        # Update target network
        if (dataset_idx + 1) % 50 == 0:
            target_model.load_state_dict(model.state_dict())
        
        # Progress reporting
        if (dataset_idx + 1) % 1000 == 0:
            current_episode = (dataset_idx + 1) * episodes_per_dataset
            avg_reward = np.mean(all_rewards[-5000:]) if len(all_rewards) >= 5000 else np.mean(all_rewards)
            avg_loss = np.mean(losses[-1000:]) if len(losses) >= 1000 else np.mean(losses) if losses else 0
            elapsed_time = time.time() - start_time
            
            print(f"\nProgress: {current_episode}/{total_episodes} episodes")
            print(f"  Avg Reward (last 5k): {avg_reward:.1f}")
            print(f"  Avg Loss (last 1k): {avg_loss:.4f}")
            print(f"  Epsilon: {get_epsilon(steps_done):.3f}")
            print(f"  Elapsed: {elapsed_time/60:.1f} min")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    return model, all_rewards, dataset_rewards, losses, training_time

# ---------- 6) Evaluation Functions ----------
def evaluate_model_on_datasets(model, test_datasets, device):
    """Evaluate trained model on test datasets"""
    model.eval()
    max_items = 20
    max_state_dim = (max_items + 1) * 3
    
    def pad_state(state, target_dim):
        if len(state) < target_dim:
            padded = np.zeros(target_dim)
            padded[:len(state)] = state
            return padded
        return state[:target_dim]
    
    results = []
    
    print("Evaluating model on test datasets...")
    for dataset in tqdm(test_datasets, desc="Evaluating"):
        env = Knapsack1DEnv(dataset['weights'], dataset['values'], dataset['capacity'])
        n_items = dataset['n_items']
        
        # DQN solution
        state = env.reset()
        state = pad_state(state, max_state_dim)
        dqn_reward = 0
        step_count = 0
        
        with torch.no_grad():
            while step_count < n_items * 2:
                s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_vals = model(s_tensor)
                action = q_vals[0, :n_items].argmax().item()
                
                state, reward, done, _ = env.step(action)
                state = pad_state(state, max_state_dim)
                dqn_reward += reward
                step_count += 1
                
                if done:
                    break
        
        # DP solution
        dp_reward = env.optimal_value()
        
        # Greedy solution
        ratios = [(dataset['values'][i]/dataset['weights'][i], i) for i in range(n_items)]
        ratios.sort(reverse=True)
        greedy_reward = 0
        total_weight = 0
        for _, idx in ratios:
            if total_weight + dataset['weights'][idx] <= dataset['capacity']:
                total_weight += dataset['weights'][idx]
                greedy_reward += dataset['values'][idx]
        
        results.append({
            'dqn': dqn_reward,
            'dp': dp_reward,
            'greedy': greedy_reward,
            'n_items': n_items,
            'capacity': dataset['capacity']
        })
    
    return results

# ---------- 7) Main Execution ----------
if __name__ == "__main__":
    # Generate training datasets
    print("Step 1: Generating training datasets...")
    train_datasets = generate_random_datasets(n_datasets=400, n_items_range=(10, 20))
    
    # Generate test datasets  
    print("\nStep 2: Generating test datasets...")
    test_datasets = generate_random_datasets(n_datasets=100, n_items_range=(10, 20))
    
    # Save datasets
    print("\nStep 3: Saving datasets...")
    with open('train_datasets.pkl', 'wb') as f:
        pickle.dump(train_datasets, f)
    with open('test_datasets.pkl', 'wb') as f:
        pickle.dump(test_datasets, f)
    print("Datasets saved to train_datasets.pkl and test_datasets.pkl")
    
    # Train model
    print("\nStep 4: Training DQN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, all_rewards, dataset_rewards, losses, training_time = train_dqn_on_datasets(
        train_datasets, episodes_per_dataset=1000
    )
    
    # Save model
    print("\nStep 5: Saving trained model...")
    model_save_path = 'dqn_knapsack_10k_datasets.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'max_state_dim': (20 + 1) * 3,
        'max_actions': 20,
        'hidden_size': 128,
        'n_layers': 3,
        'training_time': training_time,
        'total_episodes': len(all_rewards),
        'datasets_trained': len(train_datasets)
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Evaluate model
    print("\nStep 6: Evaluating model...")
    evaluation_results = evaluate_model_on_datasets(model, test_datasets, device)
    
    # Calculate statistics
    dqn_rewards = [r['dqn'] for r in evaluation_results]
    dp_rewards = [r['dp'] for r in evaluation_results]
    greedy_rewards = [r['greedy'] for r in evaluation_results]
    
    dqn_accuracy = np.array([dqn_rewards[i]/dp_rewards[i]*100 if dp_rewards[i] > 0 else 100 
                            for i in range(len(dqn_rewards))])
    greedy_accuracy = np.array([greedy_rewards[i]/dp_rewards[i]*100 if dp_rewards[i] > 0 else 100 
                               for i in range(len(greedy_rewards))])
    
    # Create comprehensive plots
    print("\nStep 7: Creating visualizations...")
    plt.figure(figsize=(20, 12))
    
    # Learning curve
    plt.subplot(2, 4, 1)
    window = 1000
    if len(all_rewards) >= window:
        ma = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        plt.plot(ma, 'b-', linewidth=2)
    plt.title(f"Learning Curve (MA {window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    # Raw rewards (sampled)
    plt.subplot(2, 4, 2)
    sample_indices = np.linspace(0, len(all_rewards)-1, 5000, dtype=int)
    plt.plot(sample_indices, [all_rewards[i] for i in sample_indices], 'b-', alpha=0.6, linewidth=0.5)
    plt.title("Raw Learning Curve (Sampled)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    # Loss curve
    plt.subplot(2, 4, 3)
    if losses:
        window_loss = 500
        if len(losses) >= window_loss:
            ma_loss = np.convolve(losses, np.ones(window_loss)/window_loss, mode='valid')
            plt.plot(ma_loss, 'r-', linewidth=2)
        plt.title(f"Training Loss (MA {window_loss})")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True)
    
    # Dataset rewards distribution
    plt.subplot(2, 4, 4)
    plt.hist(dataset_rewards, bins=50, alpha=0.7, color='blue')
    plt.title("Avg Reward per Dataset")
    plt.xlabel("Average Reward")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # Accuracy comparison
    plt.subplot(2, 4, 5)
    plt.hist(dqn_accuracy, bins=30, alpha=0.7, label='DQN', color='blue')
    plt.hist(greedy_accuracy, bins=30, alpha=0.7, label='Greedy', color='orange')
    plt.xlabel("Accuracy (% of Optimal)")
    plt.ylabel("Frequency")
    plt.title("Algorithm Performance")
    plt.legend()
    plt.grid(True)
    
    # Scatter plot: DQN vs DP
    plt.subplot(2, 4, 6)
    plt.scatter(dp_rewards, dqn_rewards, alpha=0.6, s=10)
    max_val = max(max(dp_rewards), max(dqn_rewards))
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    plt.xlabel("DP Optimal")
    plt.ylabel("DQN")
    plt.title("DQN vs DP Performance")
    plt.grid(True)
    
    # Performance by problem size
    plt.subplot(2, 4, 7)
    n_items_list = [r['n_items'] for r in evaluation_results]
    plt.scatter(n_items_list, dqn_accuracy, alpha=0.6, s=10, label='DQN')
    plt.scatter(n_items_list, greedy_accuracy, alpha=0.6, s=10, label='Greedy')
    plt.xlabel("Number of Items")
    plt.ylabel("Accuracy (%)")
    plt.title("Performance vs Problem Size")
    plt.legend()
    plt.grid(True)
    
    # Summary statistics
    plt.subplot(2, 4, 8)
    stats_data = [dqn_accuracy, greedy_accuracy]
    plt.boxplot(stats_data, labels=['DQN', 'Greedy'])
    plt.ylabel("Accuracy (%)")
    plt.title("Performance Summary")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("FINAL TRAINING AND EVALUATION RESULTS")
    print("="*80)
    print(f"Training Summary:")
    print(f"  Total datasets: {len(train_datasets)}")
    print(f"  Total episodes: {len(all_rewards)}")
    print(f"  Training time: {training_time/60:.1f} minutes")
    print(f"  Final average reward: {np.mean(all_rewards[-1000:]):.1f}")
    
    print(f"\nEvaluation Summary (on {len(test_datasets)} test datasets):")
    print(f"  DQN Performance:")
    print(f"    Mean accuracy: {np.mean(dqn_accuracy):.1f}% ± {np.std(dqn_accuracy):.1f}%")
    print(f"    Median accuracy: {np.median(dqn_accuracy):.1f}%")
    print(f"    Min accuracy: {np.min(dqn_accuracy):.1f}%")
    print(f"    Max accuracy: {np.max(dqn_accuracy):.1f}%")
    print(f"    95th percentile: {np.percentile(dqn_accuracy, 95):.1f}%")
    
    print(f"\n  Greedy Performance:")
    print(f"    Mean accuracy: {np.mean(greedy_accuracy):.1f}% ± {np.std(greedy_accuracy):.1f}%")
    print(f"    Median accuracy: {np.median(greedy_accuracy):.1f}%")
    
    print(f"\n  Performance by problem size:")
    for n in range(10, 21):
        mask = np.array(n_items_list) == n
        if np.sum(mask) > 0:
            dqn_acc_n = dqn_accuracy[mask]
            print(f"    {n} items: DQN {np.mean(dqn_acc_n):.1f}% ± {np.std(dqn_acc_n):.1f}% ({np.sum(mask)} instances)")
    
    # Save comprehensive results
    final_results = {
        'training': {
            'datasets': len(train_datasets),
            'episodes': len(all_rewards),
            'training_time_minutes': training_time/60,
            'final_avg_reward': float(np.mean(all_rewards[-1000:]))
        },
        'evaluation': {
            'test_datasets': len(test_datasets),
            'dqn_accuracy_mean': float(np.mean(dqn_accuracy)),
            'dqn_accuracy_std': float(np.std(dqn_accuracy)),
            'dqn_accuracy_median': float(np.median(dqn_accuracy)),
            'dqn_accuracy_min': float(np.min(dqn_accuracy)),
            'dqn_accuracy_max': float(np.max(dqn_accuracy)),
            'greedy_accuracy_mean': float(np.mean(greedy_accuracy)),
            'greedy_accuracy_std': float(np.std(greedy_accuracy))
        },
        'model_info': {
            'architecture': '3-layer DQN with 128 hidden units',
            'max_items': 20,
            'state_dim': (20 + 1) * 3
        }
    }
    
    with open('final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nFiles created:")
    print(f"  - {model_save_path} (trained model)")
    print(f"  - train_datasets.pkl (training data)")
    print(f"  - test_datasets.pkl (test data)")
    print(f"  - final_results.json (results summary)")
    print(f"  - dqn_training_results.png (visualization)")
    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
