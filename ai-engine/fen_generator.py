
import chess
import random
import json

PIECE_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
MAX_ATTEMPTS = 1000

def generate_random_endgame(max_pieces=4):
    board = chess.Board(None)  # empty board

    # Plasăm regii
    while True:
        king_white = random.randint(0, 63)
        king_black = random.randint(0, 63)
        if king_white != king_black and are_kings_non_adjacent(king_white, king_black):
            break
    board.set_piece_at(king_white, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(king_black, chess.Piece(chess.KING, chess.BLACK))

    # Plasăm până la 2 piese albe și 2 negre
    placed_squares = {king_white, king_black}
    for color in [chess.WHITE, chess.BLACK]:
        for _ in range(2):
            if len(placed_squares) >= max_pieces + 2:
                break
            piece_type = random.choice(PIECE_TYPES)
            while True:
                square = random.randint(0, 63)
                if square not in placed_squares:
                    board.set_piece_at(square, chess.Piece(piece_type, color))
                    placed_squares.add(square)
                    break

    board.turn = random.choice([chess.WHITE, chess.BLACK])
    board.clear_stack()
    return board.fen()

def generate_fens(n=100, max_pieces=4):
    fens = set()
    attempts = 0
    while len(fens) < n and attempts < MAX_ATTEMPTS:
        fen = generate_random_endgame(max_pieces)
        if chess.Board(fen).is_valid():
            fens.add(fen)
        attempts += 1
    return list(fens)


def are_kings_non_adjacent(square1, square2):
    file_diff = abs(chess.square_file(square1) - chess.square_file(square2))
    rank_diff = abs(chess.square_rank(square1) - chess.square_rank(square2))
    return max(file_diff, rank_diff) > 1


if __name__ == "__main__":
    output_file = "generated_endgames.json"
    fens = generate_fens(1000)
    with open(output_file, "w") as f:
        json.dump(fens, f, indent=2)
    print(f"✅ Salvat {len(fens)} FEN-uri în {output_file}")
