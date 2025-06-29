import torch
import chess
import random
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

    def make_move(self, move_uci=None, by_ai=False):
        if self.board.is_game_over():
            return False, "Game is already over"

        if not by_ai:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in self.board.legal_moves:
                    self.board.push(move)
                else:
                    return False, "Mutare ilegală"
            except:
                return False, "Format mutare invalid"
        else:
            strategy = random.choices(
                ['epsilon', 'model', 'alphabeta', 'mcts'],
                weights=[0, 0, 100, 0],
                k=1
            )[0]

            if strategy == 'epsilon':
                move = random.choice(list(self.board.legal_moves))
            elif strategy == 'model':
                move = self.ai.select_move(self.board)
            elif strategy == 'alphabeta':
                move = self.ai_move_alphabeta()
                print("alpa")
            elif strategy == 'mcts':
                move = self.ai_move_mcts()

            self.board.push(move)

        return True, {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_game_over": self.board.is_game_over()
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

    def ai_move_alphabeta(self, depth=4):
        def evaluate_board(board):
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0
            }
            value = 0
            for piece_type in piece_values:
                value += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
                value -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
            return value

        def alphabeta(board, depth, alpha, beta, maximizing):
            if depth == 0 or board.is_game_over():
                return evaluate_board(board), None

            best_move = None
            if maximizing:
                max_eval = float('-inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval, _ = alphabeta(board, depth - 1, alpha, beta, False)
                    board.pop()
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval, best_move
            else:
                min_eval = float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval, _ = alphabeta(board, depth - 1, alpha, beta, True)
                    board.pop()
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval, best_move

        if self.board.is_game_over():
            return None

        _, best_move = alphabeta(self.board, depth, float('-inf'), float('inf'), self.board.turn == chess.WHITE)
        if best_move:
            self.board.push(best_move)
            return best_move.uci()
        return None

    def ai_move_mcts(self, simulations=10):
        import random

        def simulate_random_game(board):
            temp_board = board.copy()
            while not temp_board.is_game_over():
                legal_moves = list(temp_board.legal_moves)
                move = random.choice(legal_moves)
                temp_board.push(move)

            result = temp_board.result()
            if result == '1-0':
                return 1 if self.board.turn == chess.WHITE else 0
            elif result == '0-1':
                return 1 if self.board.turn == chess.BLACK else 0
            else:
                return 0.5

        if self.board.is_game_over():
            return None

        move_scores = {}
        for move in self.board.legal_moves:
            score = 0
            for _ in range(simulations):
                self.board.push(move)
                score += simulate_random_game(self.board)
                self.board.pop()
            move_scores[move] = score / simulations

        best_move = max(move_scores, key=move_scores.get)
        self.board.push(best_move)
        return best_move.uci()
