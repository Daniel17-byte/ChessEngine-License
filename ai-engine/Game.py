import torch
import chess
from ChessAI import ChessAI

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.ai = ChessAI()
        self.turn = chess.WHITE
        if self.ai:
            import torch.nn as nn
            import torch
            self.optimizer = torch.optim.Adam(self.ai.model.parameters(), lr=0.001)
            self.loss_fn = nn.CrossEntropyLoss()

    def make_move(self, move_uci=None):
        if self.board.is_game_over():
            return False, "Game is already over"

        # Detectăm promovarea implicită la regină doar dacă e pion
        if len(move_uci) == 4:
            from_square = chess.parse_square(move_uci[:2])
            to_square = chess.parse_square(move_uci[2:4])
            piece = self.board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                # Verificăm dacă pionul ajunge la ultima linie pentru promovare
                if (piece.color == chess.WHITE and to_square // 8 == 0) or \
                   (piece.color == chess.BLACK and to_square // 8 == 7):
                    move_uci += 'q'  # promovare implicită la regină

        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            captured_piece = self.board.piece_at(move.to_square)
            piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
            reward = 0
            if captured_piece:
                reward += piece_values.get(captured_piece.symbol().lower(), 0)
            self.board.push(move)
            # Penalizare dacă o piesă rămâne sub atac după mutare
            for square in self.board.piece_map():
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:  # piesele celui care a mutat
                    if self.board.is_attacked_by(not self.board.turn, square):
                        reward -= piece_values.get(piece.symbol().lower(), 0) * 0.5
        else:
            return False, "Mutare ilegală"

        return True, {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_game_over": self.board.is_game_over(),
            "reward": reward
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

    def get_result(self):
        return self.board.result()
