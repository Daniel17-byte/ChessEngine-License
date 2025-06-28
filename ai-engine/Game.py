import chess
from ChessAI import ChessAI

class Game:
    def __init__(self, vs_ai=False):
        self.board = chess.Board()
        self.vs_ai = vs_ai
        self.ai = ChessAI() if vs_ai else None
        self.turn = chess.WHITE  # true: white, false: black

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