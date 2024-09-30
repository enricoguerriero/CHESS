import torch
from torch.utils.data import Dataset
import requests
import json
import chess
import chess.pgn
import io
import numpy as np
import time

class LichessDataset(Dataset):
    def __init__(self, usernames, max_games_per_user=10):
        self.games = []  # List to store game data
        self._fetch_and_parse_games(usernames, max_games_per_user)
        
    def _fetch_and_parse_games(self, usernames, max_games):
        for username in usernames:
            games = self._fetch_user_games(username, max_games)
            for game_data in games:
                self._parse_game(game_data)
                
    def _fetch_user_games(self, username, max_games):
        url = f'https://lichess.org/api/games/user/{username}'
        headers = {
            'Accept': 'application/x-ndjson'
        }
        params = {
            'max': max_games,
            'clocks': 'true',
            'evals': 'true',
            'opening': 'true',
            'moves': 'true'
        }
        response = requests.get(url, headers=headers, params=params)
        
        # Rate limiting
        time.sleep(1)
        
        if response.status_code == 200:
            games = response.text.strip().split('\n')
            try:
                return [json.loads(game) for game in games if game]
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                print(f"Response content: {response.text}")
                return []
        elif response.status_code == 404:
            print(f"User '{username}' not found. Please check the username.")
            return []
        else:
            print(f"Failed to fetch games for user {username}. Status code: {response.status_code}")
            return []
    
    def _parse_game(self, game_data):
        pgn_text = game_data.get('pgn', '')
        if not pgn_text:
            return
        
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        result = game.headers.get('Result', '1/2-1/2')
        label = self._get_label(result)
        moves = []
        board = game.board()
        
        for move in game.mainline_moves():
            board.push(move)
            position = self._board_to_tensor(board)
            moves.append({
                'position': position,
                'move': move.uci(),    # Move in UCI notation
                'fen': board.fen()     # Board state in FEN notation
            })
        
        # Store game data
        self.games.append({
            'username': game.headers.get('White') if label == 1 else game.headers.get('Black'),
            'opponent': game.headers.get('Black') if label == 1 else game.headers.get('White'),
            'result': result,
            'label': label,
            'moves': moves,
            'pgn': pgn_text,
            'game_data': game_data
        })
    
    def _get_label(self, result):
        if result == '1-0':
            return 1   # White wins
        elif result == '0-1':
            return -1  # Black wins
        else:
            return 0   # Draw
    
    def _board_to_tensor(self, board):
        piece_map = board.piece_map()
        board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square, piece in piece_map.items():
            piece_type = piece.piece_type
            color = int(piece.color)  # 1 for White, 0 for Black
            idx = (piece_type - 1) + (6 * color)
            row = 7 - (square // 8)
            col = square % 8
            board_tensor[idx, row, col] = 1
        return board_tensor
    
    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, idx):
        return self.games[idx]  # Return the entire game data


if __name__ == '__main__':
    usernames = ['GMHikaru', 'magnuscarlsen']
    dataset = LichessDataset(usernames, max_games_per_user=5)

    # Access the first game
    game = dataset[0]

    print("Game between:", game['username'], "and", game['opponent'])
    print("Result:", game['result'])
    print("Label:", game['label'])
    print("Number of moves:", len(game['moves']))
    print("PGN:\n", game['pgn'])
    print(len(dataset))
    # print(dataset[0])