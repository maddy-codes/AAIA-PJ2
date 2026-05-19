import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_rewards(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'rewards' in data:
                return data['rewards']
            return data
    except Exception:
        return []

def plot_results():
    # Use paths relative to current work dir
    lstm_rewards = load_rewards('results/baseline_lstm_rewards.json')
    transformer_rewards = load_rewards('results/baseline_transformer_rewards.json')
    d3qn_rewards = load_rewards('results/improved_d3qn_rewards.json')

    plt.figure(figsize=(12, 7))

    def rolling_avg(data, window=20):
        if not data: return []
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    if lstm_rewards:
        plt.plot(rolling_avg(lstm_rewards), label=f'Baseline LSTM ({len(lstm_rewards)} eps)', color='teal', alpha=0.8)
    if transformer_rewards:
        plt.plot(rolling_avg(transformer_rewards), label=f'Baseline Transformer ({len(transformer_rewards)} eps)', color='purple', linestyle='--', alpha=0.6)
    if d3qn_rewards:
        plt.plot(rolling_avg(d3qn_rewards), label=f'Improved D3QN ({len(d3qn_rewards)} eps)', color='red', linewidth=2)

    plt.title('Training Performance Comparison (Cooking World)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Rolling Avg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('report', exist_ok=True)
    out_path = 'report/comparison_plot.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Comparison plot saved to {out_path}")

if __name__ == "__main__":
    plot_results()
