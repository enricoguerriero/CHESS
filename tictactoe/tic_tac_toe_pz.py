import matplotlib.pyplot as plt
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.classic import tictactoe_v3
from pettingzoo.utils.wrappers import TerminateIllegalWrapper
import os
from tqdm import tqdm

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# Create directories for saving models if they don't exist
os.makedirs('models', exist_ok=True)


class DQN(nn.Module):
    """Deep Q-Network Model."""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        """Add a new experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)


class DQNAgent:
    """Interacts with and learns from the environment using DQN."""

    def __init__(self, state_dim, action_dim, device, 
                 learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_min=0.000001, 
                 epsilon_decay=0.995, memory_capacity=5000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Initialize the policy and target networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(capacity=memory_capacity)

        # Discount factor
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Parameters for target network updates
        self.target_update_freq = 1000  # Update target network every 1000 steps
        self.steps_done = 0

    def select_action(self, state, action_mask):
        """Select an action using epsilon-greedy policy."""
        legal_actions = [i for i, mask in enumerate(action_mask) if mask]
        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                # Mask illegal actions by setting their Q-values to -inf
                q_values[0][~torch.tensor(action_mask, dtype=torch.bool).to(self.device)] = -float('inf')
                action = q_values.argmax().item()
        return action

    def store_experience(self, experience):
        """Store experience in replay memory."""
        self.memory.add(experience)

    def update_epsilon(self):
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, batch_size=64):
        """Sample a batch of experiences and perform a learning step."""
        if len(self.memory) < 1000:
            return  # Do not learn until sufficient experiences are gathered

        batch = self.memory.sample(batch_size)
        batch_observations, batch_actions, batch_rewards, batch_next_obs, batch_dones = zip(*batch)

        # Convert to tensors
        batch_observations = torch.FloatTensor(np.array(batch_observations)).to(self.device)
        batch_actions = torch.LongTensor(np.array(batch_actions)).to(self.device)
        batch_rewards = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
        batch_next_obs = torch.FloatTensor(np.array(batch_next_obs)).to(self.device)
        batch_dones = torch.FloatTensor(np.array(batch_dones)).to(self.device)

        # Current Q-values
        q_values = self.policy_net(batch_observations).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

        # Next Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_obs).max(1)[0]
            q_targets = batch_rewards + (self.gamma * next_q_values * (1 - batch_dones))

        # Compute loss
        loss = self.criterion(q_values, q_targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network if needed
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path='models/model_dqn_tictactoe.pth'):
        """Save the policy network's state_dict."""
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path='models/model_dqn_tictactoe.pth'):
        """Load the policy network's state_dict."""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


class TicTacToeEnv:
    """Encapsulates the Tic-Tac-Toe environment."""

    def __init__(self, terminate_illegal=False, illegal_reward=-1, seed=42):
        env = tictactoe_v3.env()
        if terminate_illegal:
            env = TerminateIllegalWrapper(env, illegal_reward=illegal_reward)
        self.env = env
        self.env.reset(seed=seed)
        self.possible_agents = self.env.possible_agents

    def reset(self, seed=42):
        """Reset the environment."""
        self.env.reset(seed=seed)

    def get_observation_space_dim(self):
        """Get the flattened observation space dimension."""
        first_agent = self.possible_agents[0]
        obs_space = self.env.observation_space(first_agent)["observation"]
        return np.prod(obs_space.shape)  # 3x3x2 = 18

    def get_action_space_dim(self):
        """Get the number of possible actions."""
        first_agent = self.possible_agents[0]
        return self.env.action_space(first_agent).n  # 9 for Tic-Tac-Toe

    def agent_iter(self):
        """Get the agent iterator."""
        return self.env.agent_iter()

    def last_observation(self, agent):
        """Get the last observation for the current agent."""
        return self.env.last()

    def step(self, action):
        """Take a step in the environment."""
        self.env.step(action)


class Trainer:
    """Coordinates the training process."""

    def __init__(self, env, agent, num_episodes=10000):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.episode_rewards = {agent: [] for agent in self.env.possible_agents}

    def train(self):
        """Run the training loop."""
        for episode in tqdm(range(1, self.num_episodes + 1), ascii=True, unit='episode', desc='Training'):
            self.env.reset()
            total_rewards = {agent: 0 for agent in self.env.possible_agents}
            steps = 0
            last_observations = {}
            last_actions = {}

            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.env.last()
                steps += 1

                # Flatten the observation
                observation_board = observation["observation"].flatten()

                # Accumulate rewards
                total_rewards[agent] += reward

                # Store experience from the previous step
                if agent in last_observations:
                    prev_observation = last_observations[agent]
                    prev_action = last_actions[agent]

                    # Append experience: (state, action, reward, next_state, done)
                    self.agent.store_experience((prev_observation, prev_action, reward,
                                                observation_board.copy(), termination))

                    # Perform learning step
                    self.agent.learn()

                if termination or truncation:
                    action = None
                else:
                    action_mask = observation["action_mask"]
                    action = self.agent.select_action(observation_board, action_mask)

                    # Store current observation and action for the next step
                    if action is not None:
                        last_observations[agent] = observation_board.copy()
                        last_actions[agent] = action

                # Take the action in the environment
                self.env.step(action)

            # Decay epsilon after each episode
            self.agent.update_epsilon()

            # Record rewards
            for agent_name in self.env.possible_agents:
                self.episode_rewards[agent_name].append(total_rewards[agent_name])

            # # Print episode summary every 100 episodes
            # if episode % 100 == 0 or episode == 1:
            #     print(f'Episode {episode}/{self.num_episodes}:')
            #     for agent_name in self.env.possible_agents:
            #         print(f'  {agent_name} Total Reward: {total_rewards[agent_name]}')
            #     print(f'  Steps: {steps}')
            #     print(f'  Epsilon: {self.agent.epsilon:.6f}\n')

        # Save the trained model
        self.agent.save_model()

        return self.episode_rewards


def plot_rewards(rewards, agents, window=100):
    """Plot the rewards over episodes for each agent with moving average."""
    plt.figure(figsize=(12, 6))
    for agent in agents:
        rewards_array = np.array(rewards[agent])
        moving_avg = np.convolve(rewards_array, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label=f'{agent} (MA {window})')
    plt.title('Total Rewards over Episodes for Each Agent')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards (Moving Average)')
    plt.legend()
    plt.show()


def main():
    # Initialize environment
    env = TicTacToeEnv(terminate_illegal=False, seed=42)

    # Get state and action dimensions
    state_dim = env.get_observation_space_dim()
    action_dim = env.get_action_space_dim()
    print("State Dimension:", state_dim)
    print("Action Dimension:", action_dim)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize agent
    agent = DQNAgent(state_dim=state_dim,
                     action_dim=action_dim,
                     device=device,
                     learning_rate=0.001,
                     gamma=0.99,
                     epsilon_start=1.0,
                     epsilon_min=0.1,
                     epsilon_decay=0.9999,
                     memory_capacity=5000)

    # Initialize trainer
    trainer = Trainer(env=env, agent=agent, num_episodes=100000)

    # Start training
    rewards = trainer.train()

    # Plot the rewards
    plot_rewards(rewards, env.possible_agents)


if __name__ == "__main__":
    main()
