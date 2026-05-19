import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    if not os.path.exists(filepath): return [], []
    with open(filepath, 'r') as f:
        data = json.load(f)
        eps = data.get('episodes', [])
        rews = data.get('rewards', [])
        # Filter for episodes <= 300
        filtered = [(e, r) for e, r in zip(eps, rews) if e <= 300]
        if not filtered: return [], []
        return zip(*filtered)

def save_plot(eps, rewards, title, out_path):
    if not rewards: return
    plt.figure(figsize=(10, 5))
    plt.scatter(eps, rewards, alpha=0.3, color='blue', label='Logged Reward', s=15)
    window = 5
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(eps[window-1:], avg, color='red', label=f'{window}-Ep Moving Avg')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 310)
    plt.savefig(out_path)
    plt.close()

# Generate all
e_lstm, r_lstm = load_data('results/baseline_lstm_rewards.json')
save_plot(e_lstm, r_lstm, 'Baseline LSTM Training (300 Episodes)', 'report/lstm_training.png')

e_trans, r_trans = load_data('results/baseline_transformer_rewards.json')
save_plot(e_trans, r_trans, 'Baseline Transformer Training (300 Episodes)', 'report/transformer_training.png')

e_d3qn, r_d3qn = load_data('results/improved_d3qn_rewards.json')
save_plot(e_d3qn, r_d3qn, 'Improved D3QN Training (300 Episodes)', 'report/d3qn_training.png')

print("Individual training plots updated for 300 episodes.")
