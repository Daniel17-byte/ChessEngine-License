import torch
import chess
from ChessAI import ChessAI

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.ai = ChessAI()
        self.turn = chess.WHITE  # true: white, false: black
        if self.ai:
            import torch.nn as nn
            import torch
            self.optimizer = torch.optim.Adam(self.ai.model.parameters(), lr=0.001)
            self.loss_fn = nn.CrossEntropyLoss()

    def make_move(self, move_uci=None):
        if self.board.is_game_over():
            return False, "Game is already over"

        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            return False, "Mutare ilegală"

        return True, {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_game_over": self.board.is_game_over()
        }

    def reset(self):
        self.board.reset()
        return {
            "message": "Board reset",
            "fen": self.board.fen(),
            "turn": "white",
            "is_check": False,
            "is_checkmate": False,
            "is_stalemate": False,
            "is_insufficient_material": False
        }

    def get_board_fen(self):
        return self.board.fen()

    def is_game_over(self):
        return self.board.is_game_over()

    def get_fen(self):
        return self.board.fen()

    def ai_move(self):
        if not self.ai or self.board.is_game_over():
            return None

        if self.board.turn != chess.BLACK:
            return None

        move = self.ai.select_move(self.board)
        self.board.push(move)
        return move.uci()
