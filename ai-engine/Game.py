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

    def make_move(self, move_uci):
        try:
            if len(move_uci) == 4:
                from_square = chess.parse_square(move_uci[:2])
                to_square = chess.parse_square(move_uci[2:])
                piece = self.board.piece_at(from_square)
                if piece and piece.piece_type == chess.PAWN:
                    rank_from = chess.square_rank(from_square)
                    rank_to = chess.square_rank(to_square)
                    if (piece.color == chess.WHITE and rank_from == 6 and rank_to == 7) or \
                       (piece.color == chess.BLACK and rank_from == 1 and rank_to == 0):
                        move_uci += 'q'

            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return False, "Invalid move format"

        if move not in self.board.legal_moves:
            return False, "Illegal move"

        self.board.push(move)
        return True, {
            "result": "Move made",
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_insufficient_material": self.board.is_insufficient_material()
        }

    def get_board_grid(self):
        grid = [[None for _ in range(8)] for _ in range(8)]
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                grid[row][col] = {
                    "type": piece.symbol().lower().replace('n', 'knight'),
                    "color": "white" if piece.color == chess.WHITE else "black"
                }
        return grid

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

    def ai_move(self):
        if not self.ai or self.board.is_game_over():
            return None

        if self.board.turn != chess.BLACK:
            print("⛔ AI tried to move on white's turn.")
            return None

        move = self.ai.select_move(self.board)
        self.board.push(move)
        return move.uci()

    def is_game_over(self):
        return self.board.is_game_over()

    def get_board_state_tensor(self):
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        tensor = torch.zeros(12, 8, 8)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                idx = piece_map.get(piece.symbol(), None)
                if idx is not None:
                    row = 7 - chess.square_rank(square)
                    col = chess.square_file(square)
                    tensor[idx][row][col] = 1
        return tensor.unsqueeze(0)

    def get_fen(self):
        return self.board.fen()

    def get_last_move_uci(self):
        if len(self.board.move_stack) == 0:
            return None
        return self.board.move_stack[-1].uci()

    def get_fen_before_last_move(self):
        if len(self.board.move_stack) == 0:
            return None
        temp_board = self.board.copy()
        temp_board.pop()
        return temp_board.fen()
    def get_game_state(self):
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_insufficient_material": self.board.is_insufficient_material(),
            "last_move": self.get_last_move_uci()
        }

    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]