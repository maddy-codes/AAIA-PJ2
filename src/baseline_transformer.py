import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
import logging
import wandb
from collections import deque
from textworld_express import TextWorldExpressEnv
from transformers import DistilBertModel, DistilBertTokenizer

# --- Constants & Hyperparameters ---
SEED = 42
GAMMA = 0.95
BATCH_SIZE = 64
REPLAY_BUFFER_CAPACITY = 10000
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
DECAY_EPISODES = 240
TARGET_UPDATE_FREQ = 50
MAX_STEPS_PER_EPISODE = 20
MAX_SEQ_LEN = 256
NUM_EPISODES = 300
CHECKPOINT_FREQ = 100

# --- Setup Logging ---
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/baseline_transformer_training.log"),
        logging.StreamHandler()
    ]
)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class TransformerDQN(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        # Freeze the DistilBERT weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.scorer = nn.Linear(768, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state[:, 0, :] # [CLS] token
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
    
    # Initialize wandb with fixed ID for consistent resuming on the same line
    wandb.init(
        project="aaia-pj2",
        name="baseline-transformer",
        id="baseline-transformer-final",
        resume="allow",
        config={
            "seed": SEED,
            "gamma": GAMMA,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "decay_episodes": DECAY_EPISODES,
            "architecture": "Transformer-DQN"
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    env = TextWorldExpressEnv()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    model = TransformerDQN().to(device)
    target_model = TransformerDQN().to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    
    epsilon = EPSILON_START
    epsilon_decay_step = (EPSILON_START - EPSILON_END) / DECAY_EPISODES
    
    start_episode = 1
    episode_rewards = []
    
    # Check for existing results to resume
    if os.path.exists("results/baseline_transformer_rewards.json"):
        try:
            with open("results/baseline_transformer_rewards.json", "r") as f:
                data = json.load(f)
                episode_rewards = data["rewards"]
                start_episode = len(episode_rewards) + 1
                logging.info(f"Resuming from episode {start_episode}")
                
                # Load latest checkpoint
                ckpt_path = f"checkpoints/baseline_transformer_ep{((start_episode-1)//CHECKPOINT_FREQ)*CHECKPOINT_FREQ}.pth"
                # Fallback to last known good if above fails
                if not os.path.exists(ckpt_path):
                    for e in range(start_episode-1, 0, -1):
                        test_path = f"checkpoints/baseline_transformer_ep{e}.pth"
                        if os.path.exists(test_path):
                            ckpt_path = test_path
                            break
                
                if os.path.exists(ckpt_path):
                    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                    target_model.load_state_dict(model.state_dict())
                    logging.info(f"Loaded checkpoint {ckpt_path}")
                
                # Adjust epsilon
                for e in range(1, start_episode):
                    if e <= DECAY_EPISODES:
                        epsilon = max(EPSILON_END, EPSILON_START - e * epsilon_decay_step)
        except Exception as e:
            logging.error(f"Error loading resume data: {e}")

    logging.info(f"Starting training for {NUM_EPISODES} episodes on 'cookingworld'...")
    
    for episode in range(start_episode, NUM_EPISODES + 1):
        obs, info = env.reset(gameName="cookingworld")
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            valid_actions = info['validActions']
            if not valid_actions: break
            
            sa_strings = [f"{obs} [SEP] {a}" for a in valid_actions]
            
            if random.random() < epsilon:
                action_idx = random.randint(0, len(valid_actions) - 1)
            else:
                model.eval()
                with torch.no_grad():
                    input_ids, masks = tokenize(sa_strings, tokenizer, device)
                    with torch.amp.autocast('cuda'):
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
                
                curr_sa_strings = [f"{b[0]} [SEP] {b[1]}" for b in batch]
                ids, masks = tokenize(curr_sa_strings, tokenizer, device)
                
                with torch.amp.autocast('cuda'):
                    q_values = model(ids, masks)
                
                with torch.no_grad():
                    targets = []
                    for b_obs, b_action, b_reward, b_next_obs, b_next_actions, b_done in batch:
                        if b_done or not b_next_actions:
                            targets.append(float(b_reward))
                        else:
                            next_sa_strings = [f"{b_next_obs} [SEP] {a}" for a in b_next_actions]
                            n_ids, n_masks = tokenize(next_sa_strings, tokenizer, device)
                            with torch.amp.autocast('cuda'):
                                next_q_vals = target_model(n_ids, n_masks)
                            targets.append(b_reward + GAMMA * next_q_vals.max().item())
                
                targets = torch.tensor(targets, dtype=torch.float).to(device)
                
                with torch.amp.autocast('cuda'):
                    loss = nn.MSELoss()(q_values, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                wandb.log({"loss": loss.item()}, commit=False)
        
        episode_rewards.append(total_reward)
        wandb.log({
            "episode": episode,
            "reward": total_reward,
            "epsilon": epsilon
        })
        
        if episode <= DECAY_EPISODES:
            epsilon = max(EPSILON_END, EPSILON_START - episode * epsilon_decay_step)
        
        if episode % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(model.state_dict())
            
        if episode % 10 == 0:
            logging.info(f"Episode {episode:5d} | Reward: {total_reward:4.1f} | Epsilon: {epsilon:.2f} | Steps: {steps}")
            
        if episode % CHECKPOINT_FREQ == 0:
            ckpt_path = f"checkpoints/baseline_transformer_ep{episode}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")
            
            output_data = {
                "episodes": list(range(1, episode + 1)),
                "rewards": episode_rewards
            }
            with open("results/baseline_transformer_rewards.json", "w") as f:
                json.dump(output_data, f, indent=4)

    output_data = {
        "episodes": list(range(1, NUM_EPISODES + 1)),
        "rewards": episode_rewards
    }
    with open("results/baseline_transformer_rewards.json", "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info("Training complete. Rewards saved to results/baseline_transformer_rewards.json")
    wandb.finish()

if __name__ == "__main__":
    train()
