import re
import json
import os

def extract_rewards(log_path):
    rewards_dict = {}
    if not os.path.exists(log_path): return [], []
    with open(log_path, "r") as f:
        for line in f:
            # Match Episode number and Reward
            # Example: 2026-05-19 04:26:45,661 [INFO] Episode   300 | Reward:  0.2 | Epsilon: 0.10 | Steps: 20
            match = re.search(r"Episode\s+(\d+)\s+\|\s+Reward:\s+([\d\.]+)", line)
            if match:
                ep = int(match.group(1))
                rew = float(match.group(2))
                # If we have multiple entries for the same episode (from different runs),
                # we'll keep the last one or the one with the highest reward? 
                # Let's keep the last one as it usually represents the latest state of that episode in the log.
                rewards_dict[ep] = rew
    
    if not rewards_dict:
        return [], []
        
    sorted_eps = sorted(rewards_dict.keys())
    rewards = [rewards_dict[ep] for ep in sorted_eps]
    return sorted_eps, rewards

# Improved D3QN
eps, d3qn_rewards = extract_rewards("logs/improved_d3qn_training.log")
with open("results/improved_d3qn_rewards.json", "w") as f:
    json.dump({"episodes": eps, "rewards": d3qn_rewards}, f, indent=4)
print(f"D3QN: {len(d3qn_rewards)} points extracted, up to episode {eps[-1] if eps else 0}")

# Transformer
eps, trans_rewards = extract_rewards("logs/baseline_transformer_training.log")
with open("results/baseline_transformer_rewards.json", "w") as f:
    json.dump({"episodes": eps, "rewards": trans_rewards}, f, indent=4)
print(f"Transformer: {len(trans_rewards)} points extracted, up to episode {eps[-1] if eps else 0}")

# LSTM
eps, lstm_rewards = extract_rewards("logs/baseline_lstm_training.log")
with open("results/baseline_lstm_rewards.json", "w") as f:
    json.dump({"episodes": eps, "rewards": lstm_rewards}, f, indent=4)
print(f"LSTM: {len(lstm_rewards)} points extracted, up to episode {eps[-1] if eps else 0}")
