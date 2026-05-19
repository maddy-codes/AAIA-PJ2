import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    if not os.path.exists(filepath): return [], []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            eps = data.get('episodes', [])
            rews = data.get('rewards', [])
            # Filter for episodes <= 300
            filtered = [(e, r) for e, r in zip(eps, rews) if e <= 300]
            if not filtered: return [], []
            return zip(*filtered)
    except Exception:
        return [], []

def plot_results():
    lstm_eps, lstm_rewards = load_data('results/baseline_lstm_rewards.json')
    trans_eps, trans_rewards = load_data('results/baseline_transformer_rewards.json')
    d3qn_eps, d3qn_rewards = load_data('results/improved_d3qn_rewards.json')

    plt.figure(figsize=(12, 7))

    def rolling_avg(data, window=5):
        if not data: return []
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    if lstm_rewards:
        avg = rolling_avg(lstm_rewards)
        plt.plot(lstm_eps[len(lstm_eps)-len(avg):], avg, label=f'Baseline LSTM (300 eps)', color='teal', alpha=0.8)
        plt.scatter(lstm_eps, lstm_rewards, color='teal', alpha=0.2, s=10)
        
    if trans_rewards:
        avg = rolling_avg(trans_rewards)
        plt.plot(trans_eps[len(trans_eps)-len(avg):], avg, label=f'Baseline Transformer (300 eps)', color='purple', linestyle='--', alpha=0.6)
        plt.scatter(trans_eps, trans_rewards, color='purple', alpha=0.1, s=10)
        
    if d3qn_rewards:
        avg = rolling_avg(d3qn_rewards)
        plt.plot(d3qn_eps[len(d3qn_eps)-len(avg):], avg, label=f'Improved D3QN (300 eps)', color='red', linewidth=2)
        plt.scatter(d3qn_eps, d3qn_rewards, color='red', alpha=0.3, s=15)

    plt.title('Training Performance Comparison (Cooking World - 300 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Rolling Avg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 310)
    
    os.makedirs('report', exist_ok=True)
    out_path = 'report/comparison_plot.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Comparison plot saved to {out_path}")

if __name__ == "__main__":
    plot_results()
