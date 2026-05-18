import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
from collections import deque
from textworld_express import TextWorldExpressEnv
from transformers import DistilBertModel, DistilBertTokenizer

# --- Constants & Hyperparameters ---
SEED = 42
GAMMA = 0.95
BATCH_SIZE = 8  # Keep small for Transformer inference speed
REPLAY_BUFFER_CAPACITY = 500
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.8
MAX_STEPS_PER_EPISODE = 15
MAX_SEQ_LEN = 256

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class TransformerDQN(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        # Freeze the DistilBERT weights as per requirement
        for param in self.encoder.parameters():
            param.requires_grad = False
        # DRRN projection head (768 -> 1)
        self.scorer = nn.Linear(768, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT [CLS] token hidden state
        h = outputs.last_hidden_state[:, 0, :] # (batch, 768)
        logits = self.scorer(h)
        return logits.squeeze(-1)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, next_actions, done):
        self.buffer.append((state, action, reward, next_state, next_actions, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def tokenize(texts, tokenizer, device, max_len=MAX_SEQ_LEN):
    encoding = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=max_len, 
        return_tensors="pt"
    ).to(device)
    return encoding['input_ids'], encoding['attention_mask']

def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = TextWorldExpressEnv()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    model = TransformerDQN().to(device)
    target_model = TransformerDQN().to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    
    epsilon = EPSILON_START
    all_rewards = []
    
    print("Starting training loop (50 episodes)...")
    for episode in range(1, 51):
        obs, info = env.reset(gameName="cookingworld")
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            valid_actions = info['validActions']
            
            # Form state+action strings (DRRN approach)
            sa_strings = [f"{obs} [SEP] {a}" for a in valid_actions]
            
            if random.random() < epsilon:
                action_idx = random.randint(0, len(valid_actions) - 1)
            else:
                model.eval()
                with torch.no_grad():
                    input_ids, masks = tokenize(sa_strings, tokenizer, device)
                    q_values = model(input_ids, masks)
                    action_idx = q_values.argmax().item()
                model.train()
            
            action = valid_actions[action_idx]
            next_obs, reward, done, next_info = env.step(action)
            
            buffer.push(obs, action, reward, next_obs, next_info['validActions'], done)
            
            obs = next_obs
            info = next_info
            total_reward += reward
            steps += 1
            
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                
                # Current Q
                curr_sa_strings = [f"{b[0]} [SEP] {b[1]}" for b in batch]
                ids, masks = tokenize(curr_sa_strings, tokenizer, device)
                q_values = model(ids, masks)
                
                # Target Q
                targets = []
                for b_obs, b_action, b_reward, b_next_obs, b_next_actions, b_done in batch:
                    if b_done:
                        targets.append(float(b_reward))
                    else:
                        with torch.no_grad():
                            next_sa_strings = [f"{b_next_obs} [SEP] {a}" for a in b_next_actions]
                            n_ids, n_masks = tokenize(next_sa_strings, tokenizer, device)
                            next_q_vals = target_model(n_ids, n_masks)
                            targets.append(b_reward + GAMMA * next_q_vals.max().item())
                
                targets = torch.tensor(targets, dtype=torch.float).to(device)
                loss = nn.MSELoss()(q_values, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        all_rewards.append(total_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if episode % 5 == 0:
            target_model.load_state_dict(model.state_dict())
        
        print(f"Episode {episode:2d} | Reward: {total_reward:4.1f} | Epsilon: {epsilon:.2f} | Steps: {steps}")
        
    with open("baseline_transformer_rewards.json", "w") as f:
        json.dump(all_rewards, f)
    print("Training complete. Rewards saved to baseline_transformer_rewards.json")

if __name__ == "__main__":
    train()
