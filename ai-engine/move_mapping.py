import chess
import json

def generate_all_possible_uci_moves():
    all_moves = set()
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            move = chess.Move(from_square, to_square)
            all_moves.add(move.uci())
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promo)
                all_moves.add(move.uci())
    return sorted(all_moves)

if __name__ == "__main__":
    moves = generate_all_possible_uci_moves()
    with open("move_mapping.json", "w") as f:
        json.dump(moves, f)
    print(f"✅ Salvat {len(moves)} mutări în move_mapping.json")