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

# --- Constants & Hyperparameters ---
SEED = 42
GAMMA = 0.95
BATCH_SIZE = 128
REPLAY_BUFFER_CAPACITY = 50000
LEARNING_RATE = 1e-3
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
        logging.FileHandler("logs/baseline_lstm_training.log"),
        logging.StreamHandler()
    ]
)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CharacterVocabulary:
    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2char = ["<PAD>", "<UNK>"]
        for i in range(32, 127):
            char = chr(i)
            self.char2idx[char] = len(self.idx2char)
            self.idx2char.append(char)
            
    def encode(self, text):
        return [self.char2idx.get(c, 1) for c in text]
    
    def __len__(self):
        return len(self.idx2char)

class LSTMDQN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)
        
        mask = attention_mask.unsqueeze(-1).float()
        masked_outputs = outputs * mask
        sum_outputs = masked_outputs.sum(dim=1)
        count_outputs = mask.sum(dim=1).clamp(min=1)
        h = sum_outputs / count_outputs
        
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

def tokenize(texts, vocab, max_len=MAX_SEQ_LEN):
    batch_ids = []
    batch_masks = []
    for text in texts:
        ids = vocab.encode(text)[:max_len]
        mask = [1] * len(ids)
        padding = [0] * (max_len - len(ids))
        ids += padding
        mask += padding
        batch_ids.append(ids)
        batch_masks.append(mask)
    return torch.tensor(batch_ids), torch.tensor(batch_masks)

def train():
    set_seed(SEED)
    
    # Initialize wandb with fixed ID for consistent resuming on the same line
    wandb.init(
        project="aaia-pj2",
        name="baseline-lstm",
        id="baseline-lstm-final",
        resume="allow",
        config={
            "seed": SEED,
            "gamma": GAMMA,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "decay_episodes": DECAY_EPISODES,
            "architecture": "LSTM-DQN"
        }
    )
    
    # Enable MPS hardware acceleration if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    logging.info(f"Using device: {device}")
    
    env = TextWorldExpressEnv()
    vocab = CharacterVocabulary()
    
    model = LSTMDQN(len(vocab)).to(device)
    target_model = LSTMDQN(len(vocab)).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    
    epsilon = EPSILON_START
    epsilon_decay_step = (EPSILON_START - EPSILON_END) / DECAY_EPISODES
    
    start_episode = 1
    episode_rewards = []
    
    # Check for existing results to resume rewards
    if os.path.exists("results/baseline_lstm_rewards.json"):
        try:
            with open("results/baseline_lstm_rewards.json", "r") as f:
                data = json.load(f)
                episode_rewards = data["rewards"]
                start_episode = len(episode_rewards) + 1
                logging.info(f"Resuming from episode {start_episode}")
                
                # Load latest checkpoint if resuming
                ckpt_path = f"checkpoints/baseline_lstm_ep{((start_episode-1)//CHECKPOINT_FREQ)*CHECKPOINT_FREQ}.pth"
                if os.path.exists(ckpt_path):
                    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                    target_model.load_state_dict(model.state_dict())
                    logging.info(f"Loaded checkpoint {ckpt_path}")
                else:
                    # Try ep600 or others if freq changed
                    back_ckpt = f"checkpoints/baseline_lstm_ep600.pth"
                    if os.path.exists(back_ckpt):
                        model.load_state_dict(torch.load(back_ckpt, weights_only=True))
                        target_model.load_state_dict(model.state_dict())
                        logging.info(f"Loaded back-checkpoint {back_ckpt}")

                # Adjust epsilon for resume
                for e in range(1, start_episode):
                    if e <= DECAY_EPISODES:
                        epsilon = max(EPSILON_END, EPSILON_START - e * epsilon_decay_step)
        except Exception as e:
            logging.error(f"Error loading resume data: {e}")
    
    logging.info(f"Starting training loop ({NUM_EPISODES} episodes) on 'cookingworld'...")
    
    for episode in range(start_episode, NUM_EPISODES + 1):
        obs, info = env.reset(gameName="cookingworld")
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            valid_actions = info['validActions']
            if not valid_actions:
                break
                
            sa_strings = [f"{obs} [SEP] {a}" for a in valid_actions]
            
            if random.random() < epsilon:
                action_idx = random.randint(0, len(valid_actions) - 1)
            else:
                model.eval()
                with torch.no_grad():
                    input_ids, masks = tokenize(sa_strings, vocab)
                    with torch.amp.autocast('cuda'):
                        q_values = model(input_ids.to(device), masks.to(device))
                    action_idx = q_values.argmax().item()
                model.train()
            
            action = valid_actions[action_idx]
            next_obs, reward, done, next_info = env.step(action)
            next_valid_actions = next_info['validActions']
            
            buffer.push(obs, action, reward, next_obs, next_valid_actions, done)
            
            obs = next_obs
            info = next_info
            total_reward += reward
            steps += 1
            
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                
                curr_sa_strings = [f"{b[0]} [SEP] {b[1]}" for b in batch]
                ids, masks = tokenize(curr_sa_strings, vocab)
                
                with torch.amp.autocast('cuda'):
                    q_values = model(ids.to(device), masks.to(device))
                
                targets = []
                for b_obs, b_action, b_reward, b_next_obs, b_next_actions, b_done in batch:
                    if b_done or not b_next_actions:
                        targets.append(float(b_reward))
                    else:
                        with torch.no_grad():
                            next_sa_strings = [f"{b_next_obs} [SEP] {a}" for a in b_next_actions]
                            n_ids, n_masks = tokenize(next_sa_strings, vocab)
                            with torch.amp.autocast('cuda'):
                                next_q_vals = target_model(n_ids.to(device), n_masks.to(device))
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
        
        # Linear Epsilon Decay
        if episode <= DECAY_EPISODES:
            epsilon = max(EPSILON_END, EPSILON_START - episode * epsilon_decay_step)
        
        # Target Network Update
        if episode % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(model.state_dict())
            
        if episode % 10 == 0:
            logging.info(f"Episode {episode:5d} | Reward: {total_reward:4.1f} | Epsilon: {epsilon:.2f} | Steps: {steps}")
            
        # Checkpointing
        if episode % CHECKPOINT_FREQ == 0:
            ckpt_path = f"checkpoints/baseline_lstm_ep{episode}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")
            
            # Save intermediate results
            output_data = {
                "episodes": list(range(1, episode + 1)),
                "rewards": episode_rewards
            }
            with open("results/baseline_lstm_rewards.json", "w") as f:
                json.dump(output_data, f, indent=4)
        
    # Final Save
    output_data = {
        "episodes": list(range(1, NUM_EPISODES + 1)),
        "rewards": episode_rewards
    }
    with open("results/baseline_lstm_rewards.json", "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info("Training complete. Rewards saved to results/baseline_lstm_rewards.json")
    wandb.finish()

if __name__ == "__main__":
    train()
