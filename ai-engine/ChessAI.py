import chess
import chess.engine
import random
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
        strategy = random.choices(
            ['epsilon', 'model'],
            weights=[60, 30],
            k=1
        )[0]

        if strategy == 'epsilon':
            return random.choice(list(self.board.legal_moves))
        elif strategy == 'model':
            return self.get_best_move_from_model(board)
        return None

    def get_best_move_from_model(self, board: chess.Board) -> Optional[chess.Move]:
        self.board = board
        from ChessNet import encode_fen
        board_tensor = encode_fen(board.fen()).unsqueeze(0)

        legal_moves = list(board.legal_moves)
        legal_indices = [self.move_to_index(m.uci()) for m in legal_moves]

        with torch.no_grad():
            prediction = self.model(board_tensor).squeeze(0)

        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        best_idx = max(legal_indices, key=lambda i: prediction[i].item())
        best_move = chess.Move.from_uci(self.idx_to_move[best_idx])

        if best_move not in self.board.legal_moves:
            print(f"⚠️ Predicted move {best_move} is not legal in the current board position.")
            return random.choice(legal_moves)

        return best_move