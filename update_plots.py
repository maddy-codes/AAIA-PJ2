import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_rewards(filepath):
    if not os.path.exists(filepath): return []
    with open(filepath, 'r') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'rewards' in data: return data['rewards']
        return data

def save_plot(rewards, title, out_path):
    if not rewards: return
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color='blue', label='Per Episode')
    window = 20
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), avg, color='red', label=f'{window}-Ep Moving Avg')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

# Generate all
save_plot(load_rewards('results/baseline_lstm_rewards.json'), 'Baseline LSTM Training', 'report/lstm_training.png')
save_plot(load_rewards('results/baseline_transformer_rewards.json'), 'Baseline Transformer Training', 'report/transformer_training.png')
save_plot(load_rewards('results/improved_d3qn_rewards.json'), 'Improved D3QN Training', 'report/d3qn_training.png')
print("Individual training plots updated.")
