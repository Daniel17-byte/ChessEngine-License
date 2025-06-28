import torch
import chess
from ChessAI import ChessAI

class Game:
    def __init__(self, vs_ai=False):
        self.board = chess.Board()
        self.vs_ai = vs_ai
        self.ai = ChessAI() if vs_ai else None
        self.turn = chess.WHITE  # true: white, false: black
        if self.ai:
            import torch.nn as nn
            import torch
            self.optimizer = torch.optim.Adam(self.ai.model.parameters(), lr=0.001)
            self.loss_fn = nn.CrossEntropyLoss()

    def make_move(self, move_uci):
        ai_move = None
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return False, "Invalid move format"
        if move not in self.board.legal_moves:
            return False, "Illegal move"
        self.board.push(move)

        if self.vs_ai and not self.board.is_game_over():
            if self.board.turn == chess.BLACK:
                ai_move = self.ai.get_best_move_minimax()
                if ai_move and ai_move in self.board.legal_moves:
                    self.board.push(ai_move)
                else:
                    print("⚠️ AI did not return a valid move.")

        if ai_move:
            return True, {"result": "Move made", "ai_move": ai_move.uci()}
        else:
            return True, {"result": "Player move accepted, no AI move"}

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