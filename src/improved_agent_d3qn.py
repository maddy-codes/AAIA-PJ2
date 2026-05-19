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
REPLAY_BUFFER_CAPACITY = 20000
LEARNING_RATE = 5e-5
EPSILON_START = 1.0
EPSILON_END = 0.1
DECAY_EPISODES = 240
TARGET_UPDATE_FREQ = 50
MAX_STEPS_PER_EPISODE = 20
MAX_SEQ_LEN = 128
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
        logging.FileHandler("logs/improved_d3qn_training.log"),
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

class D3QNTransformer(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        
        # Partial Unfreezing: Freeze layers 0-4, unfreeze Layer 5
        for i, layer in enumerate(self.encoder.transformer.layer):
            if i < 5:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
            
        self.value_head = nn.Linear(768, 1)
        self.advantage_head = nn.Linear(768, 1)
        
    def forward(self, obs_list, actions_list, tokenizer, device):
        # 1. Encode all observations to get V(s)
        obs_enc = tokenizer(obs_list, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
        v_h = self.encoder(**obs_enc).last_hidden_state[:, 0, :] # [CLS] token
        v = self.value_head(v_h) # Shape: (B, 1)
        
        # 2. Flatten all (obs, action) pairs to get A(s, a)
        flat_sa_texts = []
        num_actions_per_state = []
        for obs, actions in zip(obs_list, actions_list):
            num_actions_per_state.append(len(actions))
            for a in actions:
                flat_sa_texts.append(f"{obs} [SEP] {a}")
        
        if not flat_sa_texts:
            return [torch.tensor([]).to(device) for _ in range(len(obs_list))]

        # 3. Encode all (obs, action) pairs in one batch
        sa_enc = tokenizer(flat_sa_texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
        a_h = self.encoder(**sa_enc).last_hidden_state[:, 0, :]
        a = self.advantage_head(a_h).squeeze(-1) # Shape: (TotalActionsInBatch)
        
        # 4. Compute Q = V + (A - mean(A)) for each state
        qs = []
        start = 0
        for i, n in enumerate(num_actions_per_state):
            state_v = v[i]
            state_a = a[start:start+n]
            if n > 0:
                state_q = state_v + (state_a - state_a.mean())
            else:
                state_q = state_v
            qs.append(state_q)
            start += n
        return qs

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, valid_actions, next_valid_actions, done):
        self.buffer.append((state, action, reward, next_state, valid_actions, next_valid_actions, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train():
    set_seed(SEED)
    
    # Initialize wandb with fixed ID for consistent resuming on the same line
    wandb.init(
        project="aaia-pj2",
        name="improved-d3qn",
        id="improved-d3qn-final",
        resume="allow",
        config={
            "seed": SEED,
            "gamma": GAMMA,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "decay_episodes": DECAY_EPISODES,
            "architecture": "D3QN-DistilBERT"
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
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    online_model = D3QNTransformer().to(device)
    target_model = D3QNTransformer().to(device)
    target_model.load_state_dict(online_model.state_dict())
    
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, online_model.parameters()), lr=LEARNING_RATE)
    buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    
    epsilon = EPSILON_START
    epsilon_decay_step = (EPSILON_START - EPSILON_END) / DECAY_EPISODES
    
    start_episode = 1
    episode_rewards = []
    
    # Check for existing results to resume
    if os.path.exists("results/improved_d3qn_rewards.json"):
        try:
            with open("results/improved_d3qn_rewards.json", "r") as f:
                data = json.load(f)
                episode_rewards = data["rewards"]
                start_episode = len(episode_rewards) + 1
                logging.info(f"Resuming from episode {start_episode}")
                
                # Load latest checkpoint
                ckpt_path = f"checkpoints/improved_d3qn_ep{((start_episode-1)//CHECKPOINT_FREQ)*CHECKPOINT_FREQ}.pth"
                if not os.path.exists(ckpt_path):
                    for e in range(start_episode-1, 0, -1):
                        test_path = f"checkpoints/improved_d3qn_ep{e}.pth"
                        if os.path.exists(test_path):
                            ckpt_path = test_path
                            break

                if os.path.exists(ckpt_path):
                    online_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                    target_model.load_state_dict(online_model.state_dict())
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
            if not valid_actions:
                break
                
            # Action selection
            if random.random() < epsilon:
                action_idx = random.randint(0, len(valid_actions) - 1)
                action = valid_actions[action_idx]
            else:
                online_model.eval()
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        q_values = online_model([obs], [valid_actions], tokenizer, device)[0]
                    action_idx = q_values.argmax().item()
                    action = valid_actions[action_idx]
                online_model.train()
            
            next_obs, reward, done, next_info = env.step(action)
            next_valid_actions = next_info['validActions']
            
            buffer.push(obs, action, reward, next_obs, valid_actions, next_valid_actions, done)
            
            obs = next_obs
            info = next_info
            total_reward += reward
            steps += 1
            
            # Training Step
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                
                b_obs = [x[0] for x in batch]
                b_actions = [x[1] for x in batch]
                b_rewards = torch.tensor([x[2] for x in batch], dtype=torch.float).to(device)
                b_next_obs = [x[3] for x in batch]
                b_valid_actions = [x[4] for x in batch]
                b_next_valid_actions = [x[5] for x in batch]
                b_dones = torch.tensor([x[6] for x in batch], dtype=torch.float).to(device)
                
                with torch.amp.autocast('cuda'):
                    qs_online_current = online_model(b_obs, b_valid_actions, tokenizer, device)
                
                curr_q_list = []
                for i in range(BATCH_SIZE):
                    try:
                        action_idx = b_valid_actions[i].index(b_actions[i])
                        curr_q_list.append(qs_online_current[i][action_idx])
                    except ValueError:
                        curr_q_list.append(torch.tensor(0.0).to(device))
                curr_q = torch.stack(curr_q_list)
                
                with torch.no_grad():
                    target_q = b_rewards.clone()
                    non_terminal_indices = [i for i, d in enumerate(b_dones) if not d]
                    
                    if non_terminal_indices:
                        nt_next_obs = [b_next_obs[i] for i in non_terminal_indices]
                        nt_next_valid_actions = [b_next_valid_actions[i] for i in non_terminal_indices]
                        
                        with torch.amp.autocast('cuda'):
                            qs_next_online = online_model(nt_next_obs, nt_next_valid_actions, tokenizer, device)
                            qs_next_target = target_model(nt_next_obs, nt_next_valid_actions, tokenizer, device)
                        
                        for idx, nt_idx in enumerate(non_terminal_indices):
                            if len(nt_next_valid_actions[idx]) > 0:
                                best_action_idx = qs_next_online[idx].argmax().item()
                                target_val = b_rewards[nt_idx] + GAMMA * qs_next_target[idx][best_action_idx]
                                target_q[nt_idx] = target_val
                
                with torch.amp.autocast('cuda'):
                    loss = nn.MSELoss()(curr_q, target_q)
                
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
            target_model.load_state_dict(online_model.state_dict())
            
        if episode % 10 == 0:
            logging.info(f"Episode {episode:5d} | Reward: {total_reward:4.1f} | Epsilon: {epsilon:.2f} | Steps: {steps}")
            
        # Checkpointing
        if episode % CHECKPOINT_FREQ == 0:
            ckpt_path = f"checkpoints/improved_d3qn_ep{episode}.pth"
            torch.save(online_model.state_dict(), ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")
            
            # Save intermediate results
            output_data = {
                "episodes": list(range(1, episode + 1)),
                "rewards": episode_rewards
            }
            with open("results/improved_d3qn_rewards.json", "w") as f:
                json.dump(output_data, f, indent=4)

    # Final Save
    output_data = {
        "episodes": list(range(1, NUM_EPISODES + 1)),
        "rewards": episode_rewards
    }
    with open("results/improved_d3qn_rewards.json", "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info("Training complete. Rewards saved to results/improved_d3qn_rewards.json")
    wandb.finish()

if __name__ == "__main__":
    train()
