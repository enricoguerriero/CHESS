import matplotlib.pyplot as plt
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.classic import tictactoe_v3
from pettingzoo.utils.wrappers import TerminateIllegalWrapper

# Initialize the environment
env = tictactoe_v3.env()
#env = TerminateIllegalWrapper(env, illegal_reward=-1)
env.reset(seed=42)

# Save the list of agents
possible_agents = env.possible_agents

# Define the DQN model for Tic-Tac-Toe
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN training function for Tic-Tac-Toe
def train_dqn(env, num_episodes=10000, gamma=0.99, epsilon_start=1.0, epsilon_min=0.000001,
              epsilon_decay=0.995, learning_rate=0.001, memory_size=5000):
    env.reset()

    # Get observation space dimensions (3x3 grid with 2 planes)
    first_agent = possible_agents[0]
    observation_dim = env.observation_space(first_agent)["observation"].shape[0] * \
                      env.observation_space(first_agent)["observation"].shape[1] * \
                      env.observation_space(first_agent)["observation"].shape[2]
    print("Observation dimension:", observation_dim)

    action_dim = env.action_space(first_agent).n  # Number of actions (9 actions for Tic-Tac-Toe)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN(observation_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    epsilon = epsilon_start
    memory = deque(maxlen=memory_size)
    episode_rewards = {agent: [] for agent in possible_agents}  # Use saved possible_agents

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in possible_agents}  # Track total rewards for each agent
        steps = 0
        last_observations = {}
        last_actions = {}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            steps += 1

            # Flatten the 3x3x2 observation board for the model
            observation_board = observation["observation"].flatten()

            total_rewards[agent] += reward  # Update total rewards for this agent

            # Store the experience from the previous turn, if available
            if agent in last_observations:
                prev_observation = last_observations[agent]
                prev_action = last_actions[agent]

                # Store experience in memory
                memory.append((prev_observation, prev_action, reward,
                               observation_board.copy(), termination))

                # Update the model
                if len(memory) > 1000:  # Only start training after some experience is gathered
                    batch = random.sample(memory, 64)
                    batch_observations, batch_actions, batch_rewards, batch_next_obs, batch_dones = zip(*batch)

                    batch_observations = torch.FloatTensor(batch_observations).to(device)
                    batch_actions = torch.LongTensor(batch_actions).to(device)
                    batch_rewards = torch.FloatTensor(batch_rewards).to(device)
                    batch_next_obs = torch.FloatTensor(batch_next_obs).to(device)
                    batch_dones = torch.FloatTensor(batch_dones).to(device)

                    # Get predicted Q-values for current states
                    q_values = model(batch_observations).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                    # Compute Q-targets for next states
                    with torch.no_grad():
                        next_q_values = model(batch_next_obs).max(1)[0]
                        q_targets = batch_rewards + (gamma * next_q_values * (1 - batch_dones))

                    loss = criterion(q_values, q_targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if termination or truncation:
                action = None
            else:
                action_mask = observation["action_mask"]
                legal_actions = [i for i, mask in enumerate(action_mask) if mask]

                if not legal_actions:
                    action = None
                else:
                    if random.random() < epsilon:
                        action = random.choice(legal_actions)
                    else:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(observation_board).unsqueeze(0).to(device)
                            q_values = model(state_tensor)
                            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).to(device)
                            q_values[0][~action_mask_tensor] = -float('inf')  # Mask illegal actions
                            action = q_values.argmax().item()

                # Store current observation and action for the next step
                last_observations[agent] = observation_board.copy()
                last_actions[agent] = action

            env.step(action)

        # Epsilon decay after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Track episode rewards
        for agent in possible_agents:
            episode_rewards[agent].append(total_rewards[agent])

        # Print episode summary
        print(f'Episode {episode + 1}/{num_episodes}:')
        for agent in possible_agents:
            print(f'  {agent} Total Reward: {total_rewards[agent]}')
        print(f'  Steps: {steps}')
        print(f'  Epsilon: {epsilon:.4f}')
        print()

    # Save the model
    torch.save(model.state_dict(), 'models/model_dqn_tictactoe.pth')
    return model, episode_rewards

# Train the DQN for Tic-Tac-Toe
model, rewards = train_dqn(env)

# Plot rewards over episodes for each agent
plt.figure(figsize=(12, 6))
plt.plot(rewards['player_1'], label=f'Rewards for Player 1')
plt.title('Total Rewards over Episodes for Each Agent')
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.legend()
plt.show()
