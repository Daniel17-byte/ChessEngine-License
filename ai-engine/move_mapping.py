import chess
import json

def generate_all_possible_uci_moves():
    all_moves = set()
    board = chess.Board()
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            for promo in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_sq, to_sq, promotion=promo)
                if chess.Board().is_legal(move):
                    all_moves.add(move.uci())
    return sorted(all_moves)

if __name__ == "__main__":
    moves = generate_all_possible_uci_moves()
    with open("move_mapping.json", "w") as f:
        json.dump(moves, f)