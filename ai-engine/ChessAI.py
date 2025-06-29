import chess
import chess.engine
import random
import numpy as np
from typing import Optional
from ChessNet import ChessNet
import torch
import os


class ChessAI:
    def __init__(self):
        self.board = chess.Board()
        self.model = ChessNet()
        if os.path.exists("trained_model.pth"):
            self.model.load_state_dict(torch.load("trained_model.pth"))
            self.model.eval()
            print("✅ Model încărcat cu succes din trained_model.pth")
        else:
            print("⚠️ Model neantrenat — se va antrena de la zero.")
        self.model.eval()
        import json
        with open("move_mapping.json") as f:
            self.idx_to_move = json.load(f)
        self.move_to_idx = {uci: i for i, uci in enumerate(self.idx_to_move)}
        self.epsilon = 0.2

    def move_to_index(self, move_uci: str) -> int:
        if move_uci not in self.move_to_idx:
            self.move_to_idx[move_uci] = len(self.idx_to_move)
            self.idx_to_move.append(move_uci)
        return self.move_to_idx[move_uci]

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        self.board = board
        print(f"♟️  AI received board FEN: {board.fen()}")
        return self.get_best_move_from_model(board)

    def reset_board(self):
        self.board.reset()

    def calculate_target(self, board_tensor):
        # Calcul simplu al scorului poziției pe baza pieselor
        piece_values = torch.tensor([1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], dtype=torch.float32)
        # board_tensor: [1, 12, 8, 8]
        material = board_tensor.squeeze(0).reshape(12, -1).sum(dim=1)
        score = (material * piece_values).sum()
        return score.unsqueeze(0)  # Tensor [1]

    def evaluate_board(self, board: chess.Board) -> int:
        """Material + mobility evaluation."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        eval = 0
        for piece_type in values:
            eval += len(board.pieces(piece_type, chess.WHITE)) * values[piece_type]
            eval -= len(board.pieces(piece_type, chess.BLACK)) * values[piece_type]
        mobility = len(list(board.legal_moves))
        eval += 0.1 * mobility if board.turn == chess.WHITE else -0.1 * mobility
        return eval

    def get_best_move_from_model(self, board: chess.Board) -> Optional[chess.Move]:
        self.board = board
        from ChessNet import encode_fen
        board_tensor = encode_fen(board.fen()).unsqueeze(0)

        legal_moves = list(board.legal_moves)
        legal_indices = [self.move_to_index(m.uci()) for m in legal_moves]

        with torch.no_grad():
            prediction = self.model(board_tensor).squeeze(0)

        if random.random() < self.epsilon:
            best_move = random.choice(legal_moves)
        else:
            best_idx = max(legal_indices, key=lambda i: prediction[i].item())
            best_move = chess.Move.from_uci(self.idx_to_move[best_idx])
        return best_move

# Example usage:
if __name__ == "__main__":
    ai = ChessAI()
    move = ai.get_best_move_from_model(ai.board)
    print("Best move (Minimax):", move)
    ai.board.push(move)
    print(ai.board)