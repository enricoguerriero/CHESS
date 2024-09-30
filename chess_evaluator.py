import chess

# Piece values
piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0  # King's value is set to zero as it's invaluable
}

def get_game_phase(board):
    """
    Returns the current game phase: 'opening', 'middlegame', or 'endgame'.
    """
    total_material = sum(
        piece_values[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.piece_type != chess.KING
    )

    if total_material > 3200:
        return 'opening'
    elif total_material > 2000:
        return 'middlegame'
    else:
        return 'endgame'


# Pawns positional values

central_squares = [
    chess.D4, chess.E4, chess.D5, chess.E5,
    chess.C4, chess.F4, chess.C5, chess.F5
]

def pawn_advancement_score(square, is_white):
    rank = chess.square_rank(square)
    if is_white:
        return (rank - 1) * 10  # Ranks 2 to 7 (index 1 to 6)
    else:
        return (6 - rank) * 10  # Ranks 7 to 2 for black

def is_passed_pawn(board, square, is_white):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    pawn_direction = 1 if is_white else -1
    opponent_pawns = board.pieces(chess.PAWN, not is_white)

    # Files to check: current file and adjacent files
    files_to_check = [file]
    if file > 0:
        files_to_check.append(file - 1)
    if file < 7:
        files_to_check.append(file + 1)

    # Squares ahead of the pawn
    for f in files_to_check:
        for r in range(rank + pawn_direction, 8 if is_white else -1, pawn_direction):
            sq = chess.square(f, r)
            if sq in opponent_pawns:
                return False
    return True

def is_isolated_pawn(board, square, is_white):
    file = chess.square_file(square)
    friendly_pawns = board.pieces(chess.PAWN, is_white)

    # Check adjacent files for friendly pawns
    adjacent_files = []
    if file > 0:
        adjacent_files.append(file - 1)
    if file < 7:
        adjacent_files.append(file + 1)

    for f in adjacent_files:
        for r in range(8):
            sq = chess.square(f, r)
            if sq in friendly_pawns:
                return False
    return True

def count_doubled_pawns(board, is_white):
    pawn_files = [0] * 8  # One entry per file
    for square in board.pieces(chess.PAWN, is_white):
        file = chess.square_file(square)
        pawn_files[file] += 1
    doubled_pawns = sum(1 for count in pawn_files if count > 1)
    return doubled_pawns

def is_backward_pawn(board, square, is_white):
    # Simplified detection
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    pawn_direction = 1 if is_white else -1
    friendly_pawns = board.pieces(chess.PAWN, is_white)
    opponent_pawns = board.pieces(chess.PAWN, not is_white)

    # Check adjacent files
    adjacent_files = []
    if file > 0:
        adjacent_files.append(file - 1)
    if file < 7:
        adjacent_files.append(file + 1)

    for f in adjacent_files:
        # Check if there's a friendly pawn ahead
        for r in range(rank + pawn_direction, 8 if is_white else -1, pawn_direction):
            sq = chess.square(f, r)
            if sq in friendly_pawns:
                return False
    # Check if advancing the pawn is safe (no opponent pawn controls)
    next_square = chess.square(file, rank + pawn_direction)
    if board.is_attacked_by(not is_white, next_square):
        return True
    return False

def is_protected_passed_pawn(board, square, is_white):
    if not is_passed_pawn(board, square, is_white):
        return False
    # Check if the pawn is protected by another pawn
    pawn_direction = 1 if is_white else -1
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    friendly_pawns = board.pieces(chess.PAWN, is_white)

    # Diagonally backward squares
    for f in [file - 1, file + 1]:
        if 0 <= f <= 7:
            sq = chess.square(f, rank - pawn_direction)
            if sq in friendly_pawns:
                return True
    return False

def is_pawn_in_chain(board, square, is_white):
    pawn_direction = 1 if is_white else -1
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    friendly_pawns = board.pieces(chess.PAWN, is_white)

    # Check diagonally backward squares
    for f in [file - 1, file + 1]:
        if 0 <= f <= 7:
            sq = chess.square(f, rank - pawn_direction)
            if sq in friendly_pawns:
                return True
    return False

def evaluate_pawns(board):
    evaluation = 0

    # Constants (to tune)
    central_pawn_bonus = 20
    passed_pawn_bonus = [0, 10, 20, 30, 50, 80, 130, 0]
    isolated_pawn_penalty = 25
    doubled_pawn_penalty = 15
    backward_pawn_penalty = 20
    protected_passed_pawn_bonus = 30
    pawn_chain_bonus = 10

    for is_white in [True, False]:
        factor = 1 if is_white else -1
        pawns = board.pieces(chess.PAWN, is_white)
        opponent_pawns = board.pieces(chess.PAWN, not is_white)

        # Doubled pawns
        pawn_files = [0] * 8
        for square in pawns:
            file = chess.square_file(square)
            pawn_files[file] += 1

        doubled_pawns = sum(1 for count in pawn_files if count > 1)
        evaluation -= factor * doubled_pawn_penalty * doubled_pawns

        for square in pawns:
            file = chess.square_file(square)
            rank = chess.square_rank(square)

            # Central pawns
            if square in central_squares:
                evaluation += factor * central_pawn_bonus

            # Pawn advancement
            evaluation += factor * pawn_advancement_score(square, is_white)

            # Isolated pawns
            if is_isolated_pawn(board, square, is_white):
                evaluation -= factor * isolated_pawn_penalty

            # Backward pawns
            if is_backward_pawn(board, square, is_white):
                evaluation -= factor * backward_pawn_penalty

            # Passed pawns
            if is_passed_pawn(board, square, is_white):
                evaluation += factor * passed_pawn_bonus[rank if is_white else 7 - rank]

                # Protected passed pawn
                if is_protected_passed_pawn(board, square, is_white):
                    evaluation += factor * protected_passed_pawn_bonus

            # Pawn chain
            if is_pawn_in_chain(board, square, is_white):
                evaluation += factor * pawn_chain_bonus

    return evaluation


# Knights positional values

knight_table = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

def is_knight_outpost(board, square, is_white):
    pawn_direction = 1 if is_white else -1
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    opponent_pawns = board.pieces(chess.PAWN, not is_white)
    friendly_pawns = board.pieces(chess.PAWN, is_white)

    # Check if square cannot be attacked by opponent pawns
    for df in [-1, 1]:
        f = file + df
        r = rank - pawn_direction
        if 0 <= f <= 7 and 0 <= r <= 7:
            sq = chess.square(f, r)
            if sq in opponent_pawns:
                return False

    # Check if square is protected by a friendly pawn
    for df in [-1, 1]:
        f = file + df
        r = rank + pawn_direction
        if 0 <= f <= 7 and 0 <= r <= 7:
            sq = chess.square(f, r)
            if sq in friendly_pawns:
                return True

    return False

def knight_mobility(board, square):
    return len(board.attacks(square))

def evaluate_knights(board):
    evaluation = 0
    knight_outpost_bonus = 50
    mobility_bonus = 5  # per possible move

    for is_white in [True, False]:
        factor = 1 if is_white else -1
        knights = board.pieces(chess.KNIGHT, is_white)

        for square in knights:
            # Centralization using piece-square table
            if is_white:
                evaluation += factor * knight_table[square]
            else:
                evaluation += factor * knight_table[chess.square_mirror(square)]

            # Outpost bonus
            if is_knight_outpost(board, square, is_white):
                evaluation += factor * knight_outpost_bonus

            # Mobility bonus
            mobility = knight_mobility(board, square)
            evaluation += factor * mobility_bonus * mobility

    return evaluation


# Bishops positional values

bishop_table = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
def square_color(square):
    """
    Returns True if the square is light, False if the square is dark.
    A chessboard square is light if the sum of the file and rank is even.
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return (rank + file) % 2 == 0  # True for light, False for dark

def has_bishop_pair(board, is_white):
    return len(board.pieces(chess.BISHOP, is_white)) >= 2

def is_bad_bishop(board, square, is_white):
    bishop_is_on_light_square = square_color(square)
    own_pawns = board.pieces(chess.PAWN, is_white)

    # Check if any of the pawns are on the same color squares as the bishop
    for pawn_square in own_pawns:
        if square_color(pawn_square) == bishop_is_on_light_square:
            return True  # Pawn is blocking the bishop on the same color squares

    return False


def bishop_mobility(board, square):
    return len(board.attacks(square))

def evaluate_bishops(board):
    evaluation = 0
    bishop_pair_bonus = 50
    bad_bishop_penalty = 25
    mobility_bonus = 5  # per possible move

    for is_white in [True, False]:
        factor = 1 if is_white else -1
        bishops = board.pieces(chess.BISHOP, is_white)

        # Bishop pair bonus
        if has_bishop_pair(board, is_white):
            evaluation += factor * bishop_pair_bonus

        for square in bishops:
            # Centralization
            if is_white:
                evaluation += factor * bishop_table[square]
            else:
                evaluation += factor * bishop_table[chess.square_mirror(square)]

            # Bad bishop penalty
            if is_bad_bishop(board, square, is_white):
                evaluation -= factor * bad_bishop_penalty

            # Mobility bonus
            mobility = bishop_mobility(board, square)
            evaluation += factor * mobility_bonus * mobility

    return evaluation


# Rooks positional values

rook_table = [
     0,   0,   5,  10,  10,   5,   0,   0,
     0,   0,   5,  10,  10,   5,   0,   0,
     0,   0,   5,  10,  10,   5,   0,   0,
     0,   0,   5,  10,  10,   5,   0,   0,
     0,   0,   5,  10,  10,   5,   0,   0,
     0,   0,   5,  10,  10,   5,   0,   0,
    25,  25,  25,  25,  25,  25,  25,  25,
     0,   0,   5,  10,  10,   5,   0,   0,
]

def rook_file_status(board, square, is_white):
    file = chess.square_file(square)
    own_pawns = board.pieces(chess.PAWN, is_white)
    opponent_pawns = board.pieces(chess.PAWN, not is_white)

    own_pawn_in_file = any(chess.square_file(pawn) == file for pawn in own_pawns)
    opponent_pawn_in_file = any(chess.square_file(pawn) == file for pawn in opponent_pawns)

    if not own_pawn_in_file and not opponent_pawn_in_file:
        return 'open'
    elif not own_pawn_in_file and opponent_pawn_in_file:
        return 'semi-open'
    else:
        return 'closed'

def are_rooks_connected(board, is_white):
    rooks = list(board.pieces(chess.ROOK, is_white))
    if len(rooks) >= 2:
        rook1, rook2 = rooks[:2]
        return board.is_attacked_by(is_white, rook1) and board.is_attacked_by(is_white, rook2)
    return False

def is_on_seventh_rank(square, is_white):
    rank = chess.square_rank(square)
    return rank == (6 if is_white else 1)

def rook_mobility(board, square):
    return len(board.attacks(square))


def evaluate_rooks(board):
    evaluation = 0
    open_file_bonus = 20
    semi_open_file_bonus = 10
    connected_rook_bonus = 15
    seventh_rank_bonus = 20
    mobility_bonus = 2  # per possible move

    for is_white in [True, False]:
        factor = 1 if is_white else -1
        rooks = board.pieces(chess.ROOK, is_white)

        # Connected rooks bonus
        if are_rooks_connected(board, is_white):
            evaluation += factor * connected_rook_bonus

        for square in rooks:
            # Centralization
            if is_white:
                evaluation += factor * rook_table[square]
            else:
                evaluation += factor * rook_table[chess.square_mirror(square)]

            # Open or semi-open file bonus
            status = rook_file_status(board, square, is_white)
            if status == 'open':
                evaluation += factor * open_file_bonus
            elif status == 'semi-open':
                evaluation += factor * semi_open_file_bonus

            # Seventh rank bonus
            if is_on_seventh_rank(square, is_white):
                evaluation += factor * seventh_rank_bonus

            # Mobility bonus
            mobility = rook_mobility(board, square)
            evaluation += factor * mobility_bonus * mobility

    return evaluation


# Queens positional values

queen_table = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   5,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

def queen_mobility(board, square):
    return len(board.attacks(square))

def queen_early_development_penalty(board, is_white):
    queen_square = list(board.pieces(chess.QUEEN, is_white))[0]
    starting_square = chess.D1 if is_white else chess.D8
    if queen_square != starting_square:
        minor_pieces = board.pieces(chess.KNIGHT, is_white) | board.pieces(chess.BISHOP, is_white)
        starting_squares = {chess.B1, chess.G1, chess.C1, chess.F1} if is_white else {chess.B8, chess.G8, chess.C8, chess.F8}
        undeveloped_minors = minor_pieces & starting_squares
        if undeveloped_minors:
            return 20  # Penalty
    return 0

def evaluate_queens(board):
    evaluation = 0
    mobility_bonus = 1  # per possible move

    for is_white in [True, False]:
        factor = 1 if is_white else -1
        queens = board.pieces(chess.QUEEN, is_white)
        if not queens:
            continue  # Queen may have been captured

        square = list(queens)[0]

        # Centralization
        if is_white:
            evaluation += factor * queen_table[square]
        else:
            evaluation += factor * queen_table[chess.square_mirror(square)]

        # Mobility bonus
        mobility = queen_mobility(board, square)
        evaluation += factor * mobility_bonus * mobility

        # Early development penalty
        penalty = queen_early_development_penalty(board, is_white)
        evaluation -= factor * penalty

    return evaluation


# Global evaluation factors

def evaluate_space(board):
    white_space = 0
    black_space = 0

    for square in chess.SQUARES:
        if chess.square_rank(square) >= 4:
            # Squares in black's half
            attackers = board.attackers(chess.WHITE, square)
            if attackers:
                white_space += 1
        else:
            # Squares in white's half
            attackers = board.attackers(chess.BLACK, square)
            if attackers:
                black_space += 1

    space_score = 5 * (white_space - black_space)
    return space_score

def evaluate_key_squares(board):
    key_squares = [chess.E4, chess.D4, chess.E5, chess.D5, chess.C4, chess.F4, chess.C5, chess.F5]
    white_control = 0
    black_control = 0

    for square in key_squares:
        if board.is_attacked_by(chess.WHITE, square):
            white_control += 1
        if board.is_attacked_by(chess.BLACK, square):
            black_control += 1

    control_score = 15 * (white_control - black_control)
    return control_score


def count_pawn_islands(board, is_white):
    pawns = board.pieces(chess.PAWN, is_white)
    files_with_pawns = set(chess.square_file(p) for p in pawns)
    pawn_islands = 0
    files = sorted(files_with_pawns)
    if files:
        pawn_islands = 1
        last_file = files[0]
        for f in files[1:]:
            if f != last_file + 1:
                pawn_islands += 1
            last_file = f
    return pawn_islands

def evaluate_pawn_structure(board):
    evaluation = 0
    for is_white in [True, False]:
        factor = -1 if is_white else 1  # Penalize own pawn islands
        pawn_islands = count_pawn_islands(board, is_white)
        evaluation += factor * 15 * (pawn_islands - 1)  # Penalty per extra pawn island
    return evaluation


def evaluate_piece_coordination(board):
    evaluation = 0
    for is_white in [True, False]:
        factor = 1 if is_white else -1
        pieces = board.piece_map()
        own_pieces = [sq for sq, piece in pieces.items() if piece.color == is_white]
        coordination = 0
        for sq in own_pieces:
            attackers = board.attackers(is_white, sq)
            coordination += len(attackers) - 1  # Subtract 1 to exclude the piece itself
        evaluation += factor * 5 * coordination
    return evaluation

def evaluate_mobility(board):
    white_mobility = len(list(board.legal_moves))
    board.push(chess.Move.null())  # Switch turns
    black_mobility = len(list(board.legal_moves))
    board.pop()
    mobility_score = 10 * (white_mobility - black_mobility)
    return mobility_score if board.turn else -mobility_score


def evaluate_threats(board):
    evaluation = 0
    for is_white in [True, False]:
        factor = -1 if is_white else 1  # Penalize own threats
        own_pieces = board.pieces(chess.PIECE_TYPES, is_white)
        opponent_attackers = board.attackers(not is_white)
        for square in own_pieces:
            if board.is_attacked_by(not is_white, square):
                if not board.is_attacked_by(is_white, square):
                    # Hanging piece
                    evaluation += factor * piece_values[board.piece_type_at(square)]
    return evaluation

def evaluate_pins(board):
    evaluation = 0
    for is_white in [True, False]:
        factor = -1 if is_white else 1
        king_square = list(board.pieces(chess.KING, is_white))[0]
        own_pieces = board.pieces(chess.PIECE_TYPES, is_white) - {king_square}
        for square in own_pieces:
            if board.is_pinned(is_white, square):
                piece_value = piece_values[board.piece_type_at(square)]
                evaluation += factor * 10 * (piece_value / 100)  # Adjust as needed
    return evaluation

def evaluate_checks_and_forced_moves(board):
    """
    Evaluates if the opponent is in check or has limited legal moves.
    """
    evaluation = 0
    opponent_color = not board.turn
    factor = 1 if board.turn == chess.WHITE else -1
    
    if board.is_check():
        # Bonus for delivering a check
        evaluation += factor * 30
    
    # Count opponent's legal moves
    board.push(chess.Move.null())  # Switch turns
    opponent_legal_moves = len(list(board.legal_moves))
    board.pop()
    
    if opponent_legal_moves < 10:
        # Bonus for restricting opponent's options
        evaluation += factor * (10 - opponent_legal_moves) * 2  # Adjust multiplier as needed
    
    return evaluation

def detect_forks(board, is_white):
    """
    Detects potential forks that the player can make.
    """
    evaluation = 0
    factor = 1 if is_white else -1
    own_pieces = {sq: board.piece_at(sq) for sq in board.piece_map() if board.piece_at(sq).color == is_white}
    opponent_pieces = {sq: board.piece_at(sq) for sq in board.piece_map() if board.piece_at(sq).color != is_white}
    
    for from_square, piece in own_pieces.items():
        moves = board.generate_legal_moves(from_square)
        for move in moves:
            to_square = move.to_square
            board.push(move)
            attacks = board.attacks(to_square)
            attacked_pieces = [sq for sq in attacks if sq in opponent_pieces]
            if len(attacked_pieces) >= 2:
                # Fork detected
                total_value = sum(piece_values[opponent_pieces[sq].piece_type] for sq in attacked_pieces)
                # Adjust the bonus as needed
                evaluation += factor * total_value / 20
            board.pop()
    
    return evaluation

def detect_vulnerable_to_forks(board, is_white):
    """
    Detects if the player's pieces are vulnerable to forks.
    """
    evaluation = 0
    factor = -1 if is_white else 1  # Penalize own vulnerability
    own_pieces = {sq: board.piece_at(sq) for sq in board.piece_map() if board.piece_at(sq).color == is_white}
    opponent_pieces = {sq: board.piece_at(sq) for sq in board.piece_map() if board.piece_at(sq).color != is_white}
    
    # For each of opponent's pieces, check if they can fork our pieces
    for from_square, piece in opponent_pieces.items():
        moves = board.generate_legal_moves(from_square)
        for move in moves:
            to_square = move.to_square
            board.push(move)
            attacks = board.attacks(to_square)
            attacked_pieces = [sq for sq in attacks if sq in own_pieces]
            if len(attacked_pieces) >= 2:
                # Our pieces are vulnerable to a fork
                total_value = sum(piece_values[own_pieces[sq].piece_type] for sq in attacked_pieces)
                evaluation += factor * total_value / 20
            board.pop()
    
    return evaluation

def detect_skewers(board, is_white):
    """
    Detects potential skewers that the player can make.
    """
    evaluation = 0
    factor = 1 if is_white else -1
    own_pieces = {sq: board.piece_at(sq) for sq in board.piece_map() if board.piece_at(sq).color == is_white}
    opponent_pieces = {sq: board.piece_at(sq) for sq in board.piece_map() if board.piece_at(sq).color != is_white}
    
    line_pieces = [chess.BISHOP, chess.ROOK, chess.QUEEN]
    
    for from_square, piece in own_pieces.items():
        if piece.piece_type in line_pieces:
            moves = board.generate_legal_moves(from_square)
            for move in moves:
                to_square = move.to_square
                board.push(move)
                # Check if the move creates a skewer
                skewered_pieces = detect_skewer_on_line(board, to_square, is_white)
                if skewered_pieces:
                    # Skewer detected
                    front_piece_value = piece_values[skewered_pieces[0].piece_type]
                    back_piece_value = piece_values[skewered_pieces[1].piece_type]
                    evaluation += factor * (back_piece_value - front_piece_value) / 20  # Adjust as needed
                board.pop()
    
    return evaluation

def detect_skewer_on_line(board, square, is_white):
    """
    Checks for skewers along lines from a given square.
    Returns a tuple of (front_piece, back_piece) if a skewer is detected.
    """
    enemy_color = not is_white
    directions = [chess.NORTH, chess.SOUTH, chess.EAST, chess.WEST,
                  chess.NORTH_EAST, chess.NORTH_WEST, chess.SOUTH_EAST, chess.SOUTH_WEST]
    
    for direction in directions:
        squares_in_line = chess.SquareSet(chess.ray(square, direction))
        pieces_in_line = []
        for sq in squares_in_line:
            piece = board.piece_at(sq)
            if piece:
                if piece.color == enemy_color:
                    pieces_in_line.append(piece)
                else:
                    break  # Blocked by own piece
            if len(pieces_in_line) == 2:
                # Skewer detected
                front_piece = pieces_in_line[0]
                back_piece = pieces_in_line[1]
                if piece_values[front_piece.piece_type] > piece_values[back_piece.piece_type]:
                    return (front_piece, back_piece)
                else:
                    break  # Not a skewer if front piece is less valuable
    return None

def detect_vulnerable_to_skewers(board, is_white):
    """
    Detects if the player's pieces are vulnerable to skewers.
    """
    evaluation = 0
    factor = -1 if is_white else 1
    own_pieces = {sq: board.piece_at(sq) for sq in board.piece_map() if board.piece_at(sq).color == is_white}
    opponent_pieces = {sq: board.piece_at(sq) for sq in board.piece_map() if board.piece_at(sq).color != is_white}
    
    line_pieces = [chess.BISHOP, chess.ROOK, chess.QUEEN]
    
    for from_square, piece in opponent_pieces.items():
        if piece.piece_type in line_pieces:
            moves = board.generate_legal_moves(from_square)
            for move in moves:
                to_square = move.to_square
                board.push(move)
                skewered_pieces = detect_skewer_on_line(board, to_square, not is_white)
                if skewered_pieces:
                    front_piece_value = piece_values[skewered_pieces[0].piece_type]
                    back_piece_value = piece_values[skewered_pieces[1].piece_type]
                    evaluation += factor * (back_piece_value - front_piece_value) / 20
                board.pop()
    
    return evaluation



def evaluate_initiative(board):
    """
    Evaluates the initiative based on the number and value of threats made.
    """
    evaluation = 0
    current_color = board.turn
    opponent_color = not current_color
    factor = 1 if current_color == chess.WHITE else -1
    
    # Get all opponent's pieces
    opponent_pieces = board.piece_map()
    opponent_pieces = {sq: piece for sq, piece in opponent_pieces.items() if piece.color == opponent_color}
    
    # For each of our pieces, check if it's attacking any opponent's piece
    for our_square in board.piece_map():
        our_piece = board.piece_at(our_square)
        if our_piece.color == current_color:
            attacks = board.attacks(our_square)
            for attacked_square in attacks:
                if attacked_square in opponent_pieces:
                    attacked_piece = opponent_pieces[attacked_square]
                    # Assign a bonus based on the value of the threatened piece
                    threat_value = piece_values[attacked_piece.piece_type]
                    evaluation += factor * threat_value / 10  # Adjust the divisor as needed
    
    # Subtract opponent's threats against us
    opponent_evaluation = 0
    for their_square in board.piece_map():
        their_piece = board.piece_at(their_square)
        if their_piece.color == opponent_color:
            attacks = board.attacks(their_square)
            for attacked_square in attacks:
                if board.piece_at(attacked_square) and board.piece_at(attacked_square).color == current_color:
                    attacked_piece = board.piece_at(attacked_square)
                    threat_value = piece_values[attacked_piece.piece_type]
                    opponent_evaluation += factor * threat_value / 10
    
    # Net initiative evaluation
    net_evaluation = evaluation - opponent_evaluation
    return net_evaluation

def get_game_phase(board):
    """
    Returns the current game phase: 'opening', 'middlegame', or 'endgame'.
    """
    total_material = sum(
        piece_values[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.piece_type != chess.KING
    )

    if total_material > 3200:
        return 'opening'
    elif total_material > 2000:
        return 'middlegame'
    else:
        return 'endgame'
    
def count_attackers_near_king(board, king_square, is_white):
    """
    Counts the number of enemy pieces attacking squares around the king.
    """
    enemy_color = not is_white
    attackers = 0
    squares_around_king = chess.SquareSet(chess.square_ring(king_square, 1))

    for square in squares_around_king:
        if board.is_attacked_by(enemy_color, square):
            attackers += 1

    return attackers

def is_file_open(board, file_index, is_white):
    """
    Checks if a file is open (no pawns) or semi-open (no own pawns).
    """
    own_pawns = board.pieces(chess.PAWN, is_white)
    opponent_pawns = board.pieces(chess.PAWN, not is_white)
    own_pawns_in_file = any(chess.square_file(sq) == file_index for sq in own_pawns)
    opponent_pawns_in_file = any(chess.square_file(sq) == file_index for sq in opponent_pawns)

    if not own_pawns_in_file and not opponent_pawns_in_file:
        return 'open'
    elif not own_pawns_in_file and opponent_pawns_in_file:
        return 'semi-open'
    else:
        return 'closed'

def count_weak_squares_around_king(board, king_square, is_white):
    """
    Counts the number of weak squares (not defended by pawns) around the king.
    """
    own_pawns = board.pieces(chess.PAWN, is_white)
    squares_around_king = chess.SquareSet(chess.square_ring(king_square, 1))
    weak_squares = 0

    for square in squares_around_king:
        if square not in own_pawns:
            if not any(
                board.piece_at(sq).piece_type == chess.PAWN and board.piece_at(sq).color == is_white
                for sq in board.attacks(square)
            ):
                weak_squares += 1

    return weak_squares

def evaluate_pawn_shield(board, king_square, is_white, game_phase):
    """
    Evaluates the pawn shield in front of the king.
    """
    penalty = 0
    pawn_shield_value = 20 if game_phase == 'middlegame' else 10
    pawn_direction = 1 if is_white else -1
    own_pawns = board.pieces(chess.PAWN, is_white)

    # Determine the shield squares based on king position
    shield_rank = chess.square_rank(king_square) + pawn_direction
    shield_files = [chess.square_file(king_square) + i for i in [-1, 0, 1] if 0 <= chess.square_file(king_square) + i <= 7]

    shield_squares = [chess.square(f, shield_rank) for f in shield_files if 0 <= shield_rank <= 7]

    # Count missing pawns in the shield
    missing_pawns = sum(1 for sq in shield_squares if sq not in own_pawns)
    penalty += pawn_shield_value * missing_pawns

    return penalty

def evaluate_open_lines_to_king(board, king_square, is_white):
    """
    Evaluates penalties for open files and diagonals leading to the king.
    """
    penalty = 0
    own_pawns = board.pieces(chess.PAWN, is_white)
    file = chess.square_file(king_square)
    rank = chess.square_rank(king_square)

    # Check for open or semi-open files
    file_status = is_file_open(board, file, is_white)
    if file_status == 'open':
        penalty += 30
    elif file_status == 'semi-open':
        penalty += 15

    # Check for open diagonals
    # Define directions for diagonals
    directions = [chess.DIAGONAL_ATTACKS[chess.NE], chess.DIAGONAL_ATTACKS[chess.NW],
                  chess.DIAGONAL_ATTACKS[chess.SE], chess.DIAGONAL_ATTACKS[chess.SW]]

    for direction in directions:
        for sq in direction[king_square]:
            piece = board.piece_at(sq)
            if piece:
                if piece.color != is_white and piece.piece_type in [chess.BISHOP, chess.QUEEN]:
                    penalty += 20
                break
            else:
                continue  # Keep checking along the diagonal

    return penalty

def evaluate_attackers_near_king(board, king_square, is_white, game_phase):
    """
    Evaluates penalties based on enemy pieces attacking squares near the king.
    """
    penalty = 0
    enemy_color = not is_white
    attackers = 0
    attack_value = 10 if game_phase == 'middlegame' else 5
    squares_around_king = chess.SquareSet(chess.square_ring(king_square, 1))

    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        enemy_pieces = board.pieces(piece_type, enemy_color)
        for piece_square in enemy_pieces:
            attacks = board.attacks(piece_square)
            if attacks & squares_around_king:
                attackers += 1
                penalty += attack_value * piece_values[piece_type] / 100  # Adjust value

    return penalty


def evaluate_weak_squares(board, king_square, is_white):
    """
    Evaluates penalties for weak squares (holes) around the king.
    """
    penalty = 0
    weak_square_penalty = 15
    squares_around_king = chess.SquareSet(chess.square_ring(king_square, 1))
    own_pawns = board.pieces(chess.PAWN, is_white)

    for square in squares_around_king:
        if not any(board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and board.piece_at(sq).color == is_white for sq in board.attackers(is_white, square)):
            penalty += weak_square_penalty

    return penalty

def evaluate_king_exposure(board, king_square, is_white):
    """
    Evaluates penalties for pawn moves that expose the king.
    """
    penalty = 0
    pawn_moves_penalty = 10
    starting_rank = 0 if is_white else 7
    own_pawns = board.pieces(chess.PAWN, is_white)

    # Pawns on the same file as the king
    file = chess.square_file(king_square)
    for rank in range(starting_rank, chess.square_rank(king_square)):
        sq = chess.square(file, rank)
        if sq not in own_pawns:
            penalty += pawn_moves_penalty

    return penalty

def evaluate_castling_status(board, is_white, game_phase):
    """
    Evaluates bonuses for castling and penalties for not castling when appropriate.
    """
    bonus = 0
    if game_phase == 'opening':
        if (is_white and not board.has_castling_rights(chess.WHITE)) or (not is_white and not board.has_castling_rights(chess.BLACK)):
            bonus += 30  # Bonus for castling
        else:
            bonus -= 20  # Penalty for not castling
    return bonus


def king_safety(board, is_white):
    """
    Evaluates the king's safety considering multiple factors.
    """
    evaluation = 0
    factor = 1 if is_white else -1
    game_phase = get_game_phase(board)
    king_square = list(board.pieces(chess.KING, is_white))[0]
    rank = chess.square_rank(king_square)
    file = chess.square_file(king_square)
    own_pawns = board.pieces(chess.PAWN, is_white)
    enemy_pieces = board.pieces(chess.PIECE_TYPES, not is_white)

    # 1. Pawn Shield Integrity
    pawn_shield_penalty = evaluate_pawn_shield(board, king_square, is_white, game_phase)
    evaluation += factor * pawn_shield_penalty

    # 2. Open Files and Diagonals
    open_line_penalty = evaluate_open_lines_to_king(board, king_square, is_white)
    evaluation += factor * open_line_penalty

    # 3. Enemy Attackers Near the King
    attackers_penalty = evaluate_attackers_near_king(board, king_square, is_white, game_phase)
    evaluation += factor * attackers_penalty

    # 4. Weak Squares Around the King
    weak_squares_penalty = evaluate_weak_squares(board, king_square, is_white)
    evaluation += factor * weak_squares_penalty

    # 5. King Exposure Due to Pawn Moves
    exposure_penalty = evaluate_king_exposure(board, king_square, is_white)
    evaluation += factor * exposure_penalty

    # 6. Castling Status
    castling_bonus = evaluate_castling_status(board, is_white, game_phase)
    evaluation += factor * castling_bonus

    # Total King Safety Evaluation
    return evaluation


def evaluate_king_activity(board, is_white):
    """
    Evaluates the king's activity in the endgame.
    """
    evaluation = 0
    factor = 1 if is_white else -1
    king_square = list(board.pieces(chess.KING, is_white))[0]
    king_mobility = len(board.attacks(king_square))

    # Encourage centralization in the endgame
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    if king_square in center_squares:
        evaluation += factor * 20

    # Mobility bonus
    evaluation += factor * 5 * king_mobility

    return evaluation

king_table_midgame = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

king_table_endgame = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]


def evaluate_king(board):
    """
    Evaluates the king's position, safety, and activity.
    """
    evaluation = 0
    game_phase = get_game_phase(board)

    for is_white in [True, False]:
        factor = 1 if is_white else -1
        king_square = list(board.pieces(chess.KING, is_white))[0]

        if game_phase in ['opening', 'middlegame']:
            # Use midgame table and king safety evaluation
            if is_white:
                evaluation += factor * king_table_midgame[king_square]
            else:
                evaluation += factor * king_table_midgame[chess.square_mirror(king_square)]
            # King safety
            evaluation += king_safety(board, is_white)
        else:
            # Use endgame table and king activity evaluation
            if is_white:
                evaluation += factor * king_table_endgame[king_square]
            else:
                evaluation += factor * king_table_endgame[chess.square_mirror(king_square)]
            # King activity
            evaluation += evaluate_king_activity(board, is_white)

    return evaluation


def evaluate_board(board):
    if board.is_checkmate():
        return -99999 if board.turn else 99999
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0

    evaluation = 0

    # Material evaluation
    for piece_type in piece_values:
        evaluation += piece_values[piece_type] * (
            len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK))
        )

    # Positional evaluations
    evaluation += evaluate_pawns(board)
    evaluation += evaluate_knights(board)
    evaluation += evaluate_bishops(board)
    evaluation += evaluate_rooks(board)
    evaluation += evaluate_queens(board)
    evaluation += evaluate_king(board)  # Includes enhanced king safety

    # Global features
    evaluation += evaluate_mobility(board)
    evaluation += evaluate_space(board)
    evaluation += evaluate_key_squares(board)
    evaluation += evaluate_checks_and_forced_moves(board)
    evaluation += evaluate_initiative(board)

    # Structures
    evaluation += evaluate_pawn_structure(board)
    evaluation += evaluate_piece_coordination(board)

    # Tactics
    evaluation += detect_forks(board, board.turn)
    evaluation += detect_vulnerable_to_forks(board, board.turn)
    evaluation += detect_skewers(board, board.turn)
    evaluation += detect_vulnerable_to_skewers(board, board.turn)
    evaluation += evaluate_threats(board)
    evaluation += evaluate_pins(board)

    return evaluation if board.turn else -evaluation
