import re
import json
import os

def extract_rewards(log_path):
    rewards = []
    if not os.path.exists(log_path): return []
    with open(log_path, "r") as f:
        for line in f:
            match = re.search(r"Episode\s+(\d+)\s+\|\s+Reward:\s+([\d\.]+)", line)
            if match:
                rewards.append(float(match.group(2)))
    return rewards

# Improved D3QN
d3qn_rewards = extract_rewards("logs/improved_d3qn_training.log")
with open("results/improved_d3qn_rewards.json", "w") as f:
    json.dump({"episodes": list(range(1, len(d3qn_rewards) + 1)), "rewards": d3qn_rewards}, f, indent=4)
print(f"D3QN: {len(d3qn_rewards)} episodes")

# Transformer
trans_rewards = extract_rewards("logs/baseline_transformer_training.log")
with open("results/baseline_transformer_rewards.json", "w") as f:
    json.dump({"episodes": list(range(1, len(trans_rewards) + 1)), "rewards": trans_rewards}, f, indent=4)
print(f"Transformer: {len(trans_rewards)} episodes")

# LSTM
lstm_rewards = extract_rewards("logs/baseline_lstm_training.log")
with open("results/baseline_lstm_rewards.json", "w") as f:
    json.dump({"episodes": list(range(1, len(lstm_rewards) + 1)), "rewards": lstm_rewards}, f, indent=4)
print(f"LSTM: {len(lstm_rewards)} episodes")
