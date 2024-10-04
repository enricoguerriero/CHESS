import os
import chess
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, optimizers
from collections import deque

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus, 'GPU')
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)

# Define global variables
ACTION_SPACE_SIZE = 4672  # Placeholder; will be updated based on action mapping
MAX_GAME_LENGTH = 200     # Maximum number of moves per game

# Path to Stockfish (if used)
STOCKFISH_PATH = "./stockfish/stockfish-ubuntu-x86-64-avx2"

# Step 1: Create Action Mapping
def create_action_mapping():
    move_to_index = {}
    index_to_move = {}
    idx = 0
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            # Normal moves
            move = chess.Move(from_square, to_square)
            uci = move.uci()
            if uci not in move_to_index:
                move_to_index[uci] = idx
                index_to_move[idx] = move
                idx += 1
            # Promotion moves
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion)
                uci = move.uci()
                if uci not in move_to_index:
                    move_to_index[uci] = idx
                    index_to_move[idx] = move
                    idx += 1
    return move_to_index, index_to_move

move_to_index, index_to_move = create_action_mapping()
ACTION_SPACE_SIZE = len(move_to_index)
print(f"Total action space size: {ACTION_SPACE_SIZE}")

# Step 2: Helper functions for board representation
def split_dims(board):
    board3d = np.zeros((14, 8, 8), dtype=np.int8)
    
    # Piece positions
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # Turn
    board3d[13][:, :] = int(board.turn)
    
    return board3d

# Step 3: Load the pre-trained model and extend it
pretrained_model_path = './models/model_01_15.h5'
if os.path.exists(pretrained_model_path):
    pretrained_model = load_model(pretrained_model_path)
    print("Pre-trained model loaded successfully.")
    pretrained_model.summary()
else:
    raise FileNotFoundError(f"Pre-trained model not found at {pretrained_model_path}")

# Inspect layer names and output shapes
print("\nPre-trained model layer names and output shapes:")
for idx, layer in enumerate(pretrained_model.layers):
    try:
        print(f"Layer {idx}: {layer.name} - {layer.output.shape}")
    except AttributeError:
        print(f"Layer {idx}: {layer.name} - No output shape available")

# Identify the last shared layer (assumed to be 'dense')
try:
    shared_layer = pretrained_model.get_layer('dense').output
except ValueError:
    # If 'dense' layer not found, adjust accordingly
    shared_layer = pretrained_model.layers[-2].output  # Assuming last layer is output
    print("Layer 'dense' not found. Using the second last layer as shared_layer.")

# Add Policy Head
policy_dense = layers.Dense(256, activation='relu', name='policy_dense')(shared_layer)
policy_output = layers.Dense(ACTION_SPACE_SIZE, activation='softmax', name='policy_output')(policy_dense)

# Add Value Head
value_output = layers.Dense(1, activation='tanh', name='value_output')(shared_layer)

# Create the new RL model
rl_model = Model(inputs=pretrained_model.input, outputs=[policy_output, value_output])

# Compile the model
rl_model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss={
        'policy_output': 'categorical_crossentropy',
        'value_output': 'mean_squared_error'
    },
    loss_weights={
        'policy_output': 1.0,
        'value_output': 0.5
    }
)

print("\nRL model with Policy and Value heads created successfully.")
rl_model.summary()

# Step 4: Implement Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        state_batch, policy_batch, value_batch = map(np.array, zip(*batch))
        return state_batch, policy_batch, value_batch
    
    def __len__(self):
        return len(self.buffer)

# Step 5: Define the Memory class for self-play
class Memory:
    def __init__(self):
        self.state_memory = []
        self.policy_memory = []
        self.value_memory = []
    
    def store(self, state, policy, value):
        self.state_memory.append(state)
        self.policy_memory.append(policy)
        self.value_memory.append(value)
    
    def clear(self):
        self.state_memory = []
        self.policy_memory = []
        self.value_memory = []

# Step 6: Implement Masking of Illegal Moves
def mask_illegal_moves(policy_output, board):
    legal_moves = list(board.legal_moves)
    mask = np.zeros(ACTION_SPACE_SIZE)
    for move in legal_moves:
        if move.uci() in move_to_index:
            mask[move_to_index[move.uci()]] = 1
    masked_policy = policy_output * mask
    masked_policy_sum = np.sum(masked_policy)
    if masked_policy_sum > 0:
        masked_policy /= masked_policy_sum
    else:
        masked_policy = mask / np.sum(mask)
    return masked_policy

# Step 7: Implement Move Selection
def select_move(model, board):
    state = split_dims(board)  # Shape: (14, 8, 8)
    state_for_prediction = np.expand_dims(state, axis=0)  # Shape: (1, 14, 8, 8)
    policy_pred, _ = model.predict(state_for_prediction, verbose=0)
    policy_pred = policy_pred[0]  # Shape: (ACTION_SPACE_SIZE,)
    masked_policy = mask_illegal_moves(policy_pred, board)
    move_index = np.random.choice(range(ACTION_SPACE_SIZE), p=masked_policy)
    move = index_to_move.get(move_index, None)
    if move is None or move not in board.legal_moves:
        # Fallback to random legal move if selected move is illegal
        move = random.choice(list(board.legal_moves))
    return move

# Step 8: Define Reward Function
def get_reward(board):
    result = board.result()
    if result == '1-0':
        return 1  # White wins
    elif result == '0-1':
        return -1  # Black wins
    else:
        return 0  # Draw or ongoing game

# Step 9: Implement Self-Play Mechanism
def self_play(model, num_games):
    memory = Memory()
    for game_num in range(1, num_games + 1):
        board = chess.Board()
        game_memory = []
        state_list = []
        while not board.is_game_over() and board.fullmove_number <= MAX_GAME_LENGTH:
            state = split_dims(board)  # Shape: (14, 8, 8)
            state_list.append(state)

            # Batch prediction
            if len(state_list) >= 32 or board.is_game_over():
                state_batch = np.array(state_list)
                policy_preds, value_preds = model.predict(state_batch, verbose=0)
                for i in range(len(state_batch)):
                    state = state_batch[i]
                    policy_pred = policy_preds[i]  # Shape: (ACTION_SPACE_SIZE,)
                    value_pred = value_preds[i]
                    # Select move based on policy
                    masked_policy = mask_illegal_moves(policy_pred, board)
                    move_index = np.random.choice(range(ACTION_SPACE_SIZE), p=masked_policy)
                    move = index_to_move.get(move_index, None)
                    if move is None or move not in board.legal_moves:
                        move = random.choice(list(board.legal_moves))
                    board.push(move)
                    game_memory.append((state, policy_pred, value_pred))
                state_list = []

        # Get the game result
        reward = get_reward(board)
        for state, policy, value in game_memory:
            memory.store(state, policy, reward)
        print(f"Completed self-play game {game_num}/{num_games} with reward {reward}.")
    return memory

# Step 10: Training Loop
def train_rl_model(model, num_iterations, games_per_iteration, batch_size):
    replay_buffer = ReplayBuffer()
    for iteration in range(1, num_iterations + 1):
        print(f"\n=== Iteration {iteration}/{num_iterations} ===")
        # Self-play to generate training data
        memory = self_play(model, games_per_iteration)
        # Push data to replay buffer
        for state, policy, reward in zip(memory.state_memory, memory.policy_memory, memory.value_memory):
            replay_buffer.push(state, policy, reward)
        print(f"Replay buffer size: {len(replay_buffer)}")
        # Sample a batch and train
        if len(replay_buffer) < batch_size:
            print("Not enough samples in replay buffer to train.")
            continue
        state_batch, policy_batch, value_batch = replay_buffer.sample(batch_size)
        model.fit(state_batch, {'policy_output': policy_batch, 'value_output': value_batch},
                  batch_size=batch_size, epochs=1, verbose=1)
        # Save the model periodically
        model_save_path = f'./models/checkpoints/model_rl_iteration_{iteration}.weights.h5'
        model.save_weights(model_save_path)
        print(f"Saved model weights to {model_save_path}.")
    print("RL Training completed.")

# Step 11: Execute Training
if __name__ == '__main__':
    # Define training parameters
    NUM_ITERATIONS = 10
    GAMES_PER_ITERATION = 10
    BATCH_SIZE = 64
    
    # Start training
    train_rl_model(rl_model, NUM_ITERATIONS, GAMES_PER_ITERATION, BATCH_SIZE)
    
    print("All training iterations completed.")
