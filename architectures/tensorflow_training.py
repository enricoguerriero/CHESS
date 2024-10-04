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

# Load the pre-trained model or create a new one
pretrained_model_path = './models/model_01_15_2.h5'
if os.path.exists(pretrained_model_path):
    with tf.device('/GPU:0'):
        pretrained_model = load_model(pretrained_model_path, compile=False)
    print("Pre-trained model loaded successfully.")
    pretrained_model.summary()
else:
    print("Pre-trained model not found. Creating a new model.")
    input_layer = layers.Input(shape=(14, 8, 8))
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    output_layer = layers.Dense(ACTION_SPACE_SIZE, activation='softmax')(x)
    pretrained_model = Model(inputs=input_layer, outputs=output_layer)
    pretrained_model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy')
    print("New model created successfully.")
    pretrained_model.summary()

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
    board3d = tf.zeros((14, 8, 8), dtype=tf.int8)
    for piece in chess.PIECE_TYPES:
        # White pieces
        squares = board.pieces(piece, chess.WHITE)
        for square in squares:
            idx = divmod(square, 8)
            idx = (7 - idx[0], idx[1])
            indices = tf.constant([[piece - 1, idx[0], idx[1]]])
            updates = tf.constant([1], dtype=tf.int8)
            board3d = tf.tensor_scatter_nd_update(board3d, indices, updates)
        # Black pieces
        squares = board.pieces(piece, chess.BLACK)
        for square in squares:
            idx = divmod(square, 8)
            idx = (7 - idx[0], idx[1])
            indices = tf.constant([[piece + 5, idx[0], idx[1]]])
            updates = tf.constant([1], dtype=tf.int8)
            board3d = tf.tensor_scatter_nd_update(board3d, indices, updates)
    return board3d

# Step 3: Implement Masking of Illegal Moves
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

# Step 4: Implement Move Selection
def select_move(model, board):
    state = split_dims(board)  # Shape: (14, 8, 8)
    state_for_prediction = tf.expand_dims(state, axis=0)  # Shape: (1, 14, 8, 8)
    with tf.device('/GPU:0'):
        policy_pred = model(state_for_prediction, training=False).numpy()
    policy_pred = policy_pred[0]  # Shape: (ACTION_SPACE_SIZE,)
    masked_policy = mask_illegal_moves(policy_pred, board)
    move_index = np.random.choice(range(ACTION_SPACE_SIZE), p=masked_policy)
    move = index_to_move.get(move_index, None)
    if move is None or move not in board.legal_moves:
        # Fallback to random legal move if selected move is illegal
        move = random.choice(list(board.legal_moves))
    return move

# Step 5: Define Reward Function
def get_reward(board):
    result = board.result()
    if result == '1-0':
        return 1  # White wins
    elif result == '0-1':
        return -1  # Black wins
    else:
        return 0  # Draw or ongoing game

# Step 6: Implement Self-Play Mechanism
def self_play(model, num_games):
    memory = []
    for game_num in range(1, num_games + 1):
        board = chess.Board()
        game_memory = []
        while not board.is_game_over() and board.fullmove_number <= MAX_GAME_LENGTH:
            state = split_dims(board)  # Shape: (14, 8, 8)
            policy_pred = model(tf.expand_dims(state, axis=0), training=False).numpy()[0]
            # Select move based on policy
            masked_policy = mask_illegal_moves(policy_pred, board)
            move_index = np.random.choice(range(ACTION_SPACE_SIZE), p=masked_policy)
            move = index_to_move.get(move_index, None)
            if move is None or move not in board.legal_moves:
                # Fallback to random legal move if selected move is illegal
                move = random.choice(list(board.legal_moves))
            board.push(move)
            game_memory.append((state, policy_pred))
        # Get the game result
        reward = get_reward(board)
        # Assign rewards to all moves in the game
        for state, policy in game_memory:
            memory.append((state, policy, reward))
        print(f"Completed self-play game {game_num}/{num_games} with reward {reward}.")
    return memory

# Step 7: Execute Self-Play and Training
if __name__ == '__main__':
    # Define training parameters
    NUM_GAMES = 10
    memory = self_play(pretrained_model, NUM_GAMES)
    print("Self-play completed.")