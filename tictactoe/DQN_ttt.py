import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import wandb
import torch.optim.lr_scheduler as lr_scheduler
import collections

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)  # Flattened 3x3 board
        self.current_player = 1
        self.done = False
        return self.board.copy()

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        if self.board[action] != 0 or self.done:
            return self.board.copy(), -10, True, {}
        self.board[action] = self.current_player
        reward = self.check_winner()
        self.current_player *= -1
        return self.board.copy(), reward, self.done, {}

    def check_winner(self):
        b = self.board.reshape(3, 3)
        lines = [b[i, :] for i in range(3)] + [b[:, i] for i in range(3)] + \
                [b.diagonal(), np.fliplr(b).diagonal()]
        for line in lines:
            if sum(line) == 3:
                self.done = True
                return 1  # Player 1 wins
            elif sum(line) == -3:
                self.done = True
                return -1  # Player -1 wins
        if 0 not in self.board:
            self.done = True
            return 0  # Draw
        return 0  # Continue

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)
    
def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    return model
    
def training(num_episodes, env, dqn, optimizer, criterion, gamma, epsilon, epsilon_min, epsilon_decay, action_size, save_path, scheduler=None):
    rewards = []
    for episode in tqdm(range(num_episodes), unit='episodes', unit_scale=True, desc='Training'):
        state = env.reset()
        total_reward = 0
        while True:
            # Both players use the DQN to select actions
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if random.random() <= epsilon:
                action = random.choice(env.available_actions())
            else:
                with torch.no_grad():
                    q_values = dqn(state_tensor).numpy()[0]
                # Mask invalid actions
                q_values = [q_values[i] if i in env.available_actions() else -np.inf for i in range(action_size)]
                action = int(np.argmax(q_values))
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Prepare for DQN update
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                next_q = dqn(next_state_tensor).max().item()
            target = reward + gamma * next_q * (1 - done)
            output = dqn(state_tensor)[0, action]
            loss = criterion(output, torch.tensor(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            state = next_state
            if done:
                break
        rewards.append(total_reward)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        if (episode+1) % 500 == 0:
            torch.save(dqn.state_dict(), save_path)
        wandb.log({"total_reward": total_reward,
                   "epsilon": epsilon
        })
    return dqn, rewards

def test(dqn, num_games=50):
    env = TicTacToe()
    dqn.eval()  # Set DQN to evaluation mode
    results_vs_self = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0}
    
    with torch.no_grad():
        # Play against itself
        for _ in tqdm(range(num_games), desc='Testing vs Itself'):
            state = env.reset()
            while not env.done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = dqn(state_tensor).numpy()[0]
                available = env.available_actions()
                q_values = [q_values[i] if i in available else -np.inf for i in range(9)]
                action = int(np.argmax(q_values))
                state, reward, done, _ = env.step(action)
            
            if reward == 1:
                results_vs_self['player1_wins'] += 1
            elif reward == -1:
                results_vs_self['player2_wins'] += 1
            else:
                results_vs_self['draws'] += 1
            wandb.log({"vs_self": results_vs_self})
    
    # Print results
    print("\nTest Results:")
    print("\nAgainst Itself:")
    print(f"  Player 1 Wins: {results_vs_self['player1_wins']}")
    print(f"  Player 2 Wins: {results_vs_self['player2_wins']}")
    print(f"  Draws: {results_vs_self['draws']}")

def print_results(rewards, window=100):
    plt.figure(figsize=(12, 6))
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg, label=f'Rewards (MA {window})')
    plt.title('Total Rewards over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards (Moving Average)')
    plt.legend()
    plt.show()
    
    # Count occurrences of 1, 0, and -1
    reward_counts = collections.Counter(rewards)

    # Extract counts for each reward type
    labels = ['1', '0', '-1']
    counts = [reward_counts[1], reward_counts[0], reward_counts[-1]]

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['green', 'blue', 'red'])
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.show()


def main():
    
    run_name = 'LAMADONNA'
    
    wandb.login()
    wandb.init(project='tictactoe', name=run_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    env = TicTacToe()
    state_size = 9
    action_size = 9
    dqn = DQN(state_size, action_size).to(device)
    learning_rate = 0.1  
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    num_episodes = 100000
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.7
    epsilon_decay = 0.99995
    save_path = './tictactoe/models/' + run_name + '.pth'
    gamma_sc = 0.99
    step_size = 1000
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_sc)
    wandb.log({
        "optimizer": type(optimizer).__name__,
        "criterion": type(criterion).__name__,
        "num_episodes": num_episodes,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "load": True if os.path.exists(save_path) else False
    })

    dqn = load_model(dqn, save_path)

    dqn, rewards = training(num_episodes, env, dqn, optimizer, criterion, gamma, epsilon,
                            epsilon_min, epsilon_decay, action_size, save_path, None)  

    print_results(rewards)

    test(dqn)

if __name__ == '__main__':
    main()
