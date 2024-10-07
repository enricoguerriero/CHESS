# run like this: python3 architectures/nn_torch_2.py --model-path './models/chess_nn_2.pth'


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import argparse
import wandb
from tqdm import trange

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='DQN Chess Agent Training')
parser.add_argument('--model-path', type=str, default='chess_dqn_model.pth',
                    help='Path to save/load the model')
args = parser.parse_args()
model_path = args.model_path

# State representation
def board_to_tensor(board):
    piece_planes = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = piece_planes[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6  # Offset for black pieces
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[plane, row, col] = 1
    return tensor

# Action encoding and decoding
def encode_action(move):
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion if move.promotion else 0  # 0 means no promotion
    return from_square * 64 * 7 + to_square * 7 + promotion

def decode_action(index):
    from_square = index // (64 * 7)
    to_square = (index % (64 * 7)) // 7
    promotion = (index % (64 * 7)) % 7
    promotion = promotion if promotion != 0 else None
    return chess.Move(from_square, to_square, promotion=promotion)

ACTION_SIZE = 64 * 64 * 7

# Environment class
class ChessEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = chess.Board()
        self.done = False
        self.previous_material = self.calculate_material_balance()
        return self.get_state()

    def get_state(self):
        return board_to_tensor(self.board)

    def step(self, action_idx):
        move = decode_action(action_idx)
        if move in self.board.legal_moves:
            self.board.push(move)
            reward = self.get_reward()
            done = self.board.is_game_over()
            next_state = self.get_state()
            return next_state, reward, done, {}
        else:
            # Illegal move
            reward = -10  # Penalty for illegal move
            done = True
            next_state = self.get_state()
            return next_state, reward, done, {}

    def get_reward(self):
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return -100  # Loss
            else:
                return 100   # Win
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_draw():
            return 0  # Draw
        else:
            # Intermediate rewards
            current_material = self.calculate_material_balance()
            reward = current_material - self.previous_material
            self.previous_material = current_material
            return reward

    def calculate_material_balance(self):
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        white_material = sum(
            piece_values.get(piece.piece_type, 0) for piece in self.board.piece_map().values() if piece.color == chess.WHITE
        )
        black_material = sum(
            piece_values.get(piece.piece_type, 0) for piece in self.board.piece_map().values() if piece.color == chess.BLACK
        )
        return black_material - white_material

# DQN model
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
STATE_SHAPE = (12, 8, 8)
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000
LR = 1e-4
MEMORY_CAPACITY = 100000
TARGET_UPDATE = 1000
num_episodes = 10000

# Initialize components
env = ChessEnv()
policy_net = DQN(ACTION_SIZE).to(device)
target_net = DQN(ACTION_SIZE).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_CAPACITY)

# Initialize wandb
wandb.init(project='dqn-chess-agent', name='training-run', config={
    'batch_size': BATCH_SIZE,
    'gamma': GAMMA,
    'epsilon_start': EPSILON_START,
    'epsilon_end': EPSILON_END,
    'epsilon_decay': EPSILON_DECAY,
    'learning_rate': LR,
    'memory_capacity': MEMORY_CAPACITY,
    'target_update': TARGET_UPDATE,
    'num_episodes': num_episodes,
})
wandb.watch(policy_net, log='all')

# Check for existing model and load it
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    checkpoint = torch.load(model_path)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode_start = checkpoint['episode'] + 1
    epsilon = checkpoint['epsilon']
    memory = checkpoint['memory']
    losses = checkpoint['losses']
    episode_rewards = checkpoint['episode_rewards']
    win_rates = checkpoint['win_rates']
    steps_done = checkpoint['steps_done']
    print(f"Resuming training from episode {episode_start}")
else:
    print("No existing model found. Starting training from scratch.")
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    episode_start = 0
    epsilon = EPSILON_START
    steps_done = 0
    # Initialize performance metrics
    episode_rewards = []
    win_rates = []
    losses = []
    wins = 0
    draws = 0
    losses_count = 0

# Action selection
def select_action(state, board, epsilon):
    sample = random.random()
    legal_moves = list(board.legal_moves)
    legal_move_indices = [encode_action(move) for move in legal_moves]

    if sample < epsilon:
        action_idx = random.choice(legal_move_indices)
    else:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            q_values = q_values.cpu().detach().numpy()[0]
            mask = np.full(ACTION_SIZE, -np.inf)
            mask[legal_move_indices] = q_values[legal_move_indices]
            action_idx = np.argmax(mask)
    return action_idx

# Training loop with tqdm progress bar
for episode in trange(episode_start, num_episodes, desc='Training'):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_idx = select_action(state, env.board, epsilon)
        next_state, reward, done, _ = env.step(action_idx)
        total_reward += reward

        memory.push((state, action_idx, reward, next_state, done))
        state = next_state

        steps_done += 1
        if epsilon > EPSILON_END:
            epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY

        if len(memory) > BATCH_SIZE:
            experiences = memory.sample(BATCH_SIZE)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*experiences)

            batch_state = torch.from_numpy(np.array(batch_state)).to(device)
            batch_action = torch.LongTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).to(device)
            batch_next_state = torch.from_numpy(np.array(batch_next_state)).to(device)
            batch_done = torch.FloatTensor(batch_done).to(device)

            q_values = policy_net(batch_state).gather(1, batch_action)
            next_q_values = target_net(batch_next_state).max(1)[0].detach()
            expected_q_values = batch_reward + (GAMMA * next_q_values * (1 - batch_done))

            loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss to wandb
            wandb.log({'loss': loss.item(), 'step': steps_done})

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)
    result = env.board.result()
    if result == '0-1':
        wins += 1
    elif result == '1/2-1/2':
        draws += 1
    else:
        losses_count += 1

    # Log metrics at each episode
    wandb.log({
        'episode': episode + 1,
        'epsilon': epsilon,
        'total_reward': total_reward,
        'steps_done': steps_done,
    })

    if (episode + 1) % 100 == 0:
        win_rate = wins / 100
        win_rates.append(win_rate)
        avg_loss = np.mean(losses[-100:]) if losses else 0
        print(f"Episode {episode + 1}, Epsilon: {epsilon:.4f}, Win Rate: {win_rate:.2f}, Avg Loss: {avg_loss:.4f}")
        wins = 0
        draws = 0
        losses_count = 0

        # Save checkpoint
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon,
            'memory': memory,
            'losses': losses,
            'episode_rewards': episode_rewards,
            'win_rates': win_rates,
            'steps_done': steps_done
        }
        torch.save(checkpoint, model_path)
        print(f"Checkpoint saved at episode {episode + 1}")

        # Log metrics to wandb
        wandb.log({
            'win_rate': win_rate,
            'avg_loss': avg_loss,
            'episode': episode + 1,
        })

# Save the final model
torch.save({
    'episode': num_episodes - 1,
    'policy_net_state_dict': policy_net.state_dict(),
    'target_net_state_dict': target_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epsilon': epsilon,
    'memory': memory,
    'losses': losses,
    'episode_rewards': episode_rewards,
    'win_rates': win_rates,
    'steps_done': steps_done
}, model_path)
print("Final model saved.")

# Finish wandb run
wandb.finish()

# Plotting performance metrics (Optional, since wandb logs these)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(win_rates)) * 100, win_rates)
plt.title('Win Rates')
plt.xlabel('Episode')
plt.ylabel('Win Rate')

plt.show()

# Testing the trained agent
def test_agent(num_games=10):
    agent_wins = 0
    opponent_wins = 0
    draws = 0

    for _ in range(num_games):
        env = ChessEnv()
        state = env.reset()
        done = False

        while not done:
            if env.board.turn == chess.BLACK:
                action_idx = select_action(state, env.board, epsilon=0)
                next_state, reward, done, _ = env.step(action_idx)
                state = next_state
            else:
                legal_moves = list(env.board.legal_moves)
                move = random.choice(legal_moves)
                env.board.push(move)
                state = env.get_state()
                done = env.board.is_game_over()

        result = env.board.result()
        if result == '0-1':
            agent_wins += 1
        elif result == '1/2-1/2':
            draws += 1
        else:
            opponent_wins += 1

    print(f"Agent Wins: {agent_wins}, Opponent Wins: {opponent_wins}, Draws: {draws}")

    # Log test results to wandb
    wandb.log({
        'test_agent_wins': agent_wins,
        'test_opponent_wins': opponent_wins,
        'test_draws': draws,
        'test_num_games': num_games,
    })

test_agent(num_games=100)
