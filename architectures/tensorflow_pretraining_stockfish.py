import chess
import chess.engine
import random
import numpy as np
import tensorflow as tf

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def random_board(max_depth=200):
    board = chess.Board()
    depth = random.randrange(0, max_depth)
    
    for _ in range(depth):
        random_move = random.choice(list(board.legal_moves))
        board.push(random_move)
        if board.is_game_over():
            break
    return board

def stockfish(board, depth=0):
    with chess.engine.SimpleEngine.popen_uci("./stockfish-ubuntu-x86-64-avx2") as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result["score"].white().score()
        return score

square_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}

def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), square_index[letter[0]]

def split_dims(board):
    # Use TensorFlow tensors instead of NumPy arrays
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
    
    # Legal moves
    aux = board.turn 
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        indices = tf.constant([[13, i, j]])
        updates = tf.constant([1], dtype=tf.int8)
        board3d = tf.tensor_scatter_nd_update(board3d, indices, updates)
    board.turn = aux
    
    return board3d

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))
    
    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3,
                          padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=board3d, outputs=x)

def build_model_residual(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))
    
    x = layers.Conv2D(filters=conv_size, kernel_size=3,
                      padding='same')(board3d)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    for _ in range(conv_depth):
        previous = x
        x = layers.Conv2D(filters=conv_size, kernel_size=3,
                          padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=conv_size, kernel_size=3,
                          padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, previous])
        x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=board3d, outputs=x)

import tensorflow.keras.callbacks as callbacks
from tqdm import tqdm

def get_dataset():
    num_samples = 10000
    boards = []
    values = []
    for _ in tqdm(range(num_samples)):
        board = random_board(max_depth=200)
        value = stockfish(board, depth=1)
        if value is None:
            continue  # Skip if Stockfish evaluation is None
        # Convert Stockfish evaluation to a probability between 0 and 1
        prob = 1 / (1 + 10 ** (-value / 400))
        boards.append(split_dims(board))
        values.append(prob)
    # Convert lists to tensors and place them on the GPU
    with tf.device('/GPU:0'):
        b = tf.stack(boards)
        v = tf.stack(values)
    return b, v

def minmax_eval(board):
    board3d = split_dims(board)
    board3d = tf.expand_dims(board3d, axis=0)
    # Ensure prediction is done on GPU
    with tf.device('/GPU:0'):
        return model.predict(board3d)[0][0]

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return minmax_eval(board)
    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_ai_move(board, depth=3):
    best_move = None
    max_eval = -np.inf
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -np.inf, np.inf, False)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            best_move = move
    return best_move

if __name__ == '__main__':

    x_train, y_train = get_dataset()
    print(x_train.shape, y_train.shape)
    
    model_path = './models/model_01_15.h5'
    
    try:
        # Load the model before building a new one
        with tf.device('/GPU:0'):
            model = models.load_model(model_path)
        print("Model loaded successfully.")
    except (OSError, IOError):
        print("Saved model not found. Building a new model.")
        with tf.device('/GPU:0'):
            model = build_model_residual(32, 4)
        model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
        model.summary()
    
    # Ensure training happens on GPU
    with tf.device('/GPU:0'):
        model.fit(x_train, y_train, 
                  batch_size=2048, 
                  verbose=1,
                  epochs=100, 
                  validation_split=0.1,
                  callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                             callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])
    
    model.save(model_path)
    
    board = chess.Board()

    with chess.engine.SimpleEngine.popen_uci("./stockfish/stockfish-ubuntu-x86-64-avx2") as engine:
        while True:
            move = get_ai_move(board, 1)
            board.push(move)
            print(f'\n{board}')
            if board.is_game_over():
                break
            
            move = engine.analyse(board, chess.engine.Limit(time=1))['pv'][0]
            board.push(move)
            print(f'\n{board}')
            if board.is_game_over():
                break
