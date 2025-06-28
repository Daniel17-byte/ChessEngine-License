import chess
import chess.engine
import random
import numpy as np
from typing import Optional
from ChessNet import ChessNet  # <-- adăugat
import torch


class ChessAI:
    def __init__(self):
        self.board = chess.Board()
        self.model = ChessNet()
        self.move_to_idx = {}
        self.idx_to_move = []

    def move_to_index(self, move_uci: str) -> int:
        if move_uci not in self.move_to_idx:
            self.move_to_idx[move_uci] = len(self.idx_to_move)
            self.idx_to_move.append(move_uci)
        return self.move_to_idx[move_uci]

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        self.board = board
        print(f"♟️  AI received board FEN: {board.fen()}")
        return self.get_best_move_minimax()

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

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        if maximizing:
            max_eval = -np.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = np.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move_minimax(self, depth=3) -> Optional[chess.Move]:
        best_move = None
        best_value = -np.inf
        alpha = -np.inf
        beta = np.inf
        for move in self.board.legal_moves:
            self.board.push(move)
            move_value = self.minimax(self.board, depth - 1, alpha, beta, False)
            self.board.pop()
            print(f"Evaluated move {move}: {move_value}")
            if move_value > best_value:
                best_value = move_value
                best_move = move
                alpha = max(alpha, move_value)
        return best_move

    def get_random_move(self) -> chess.Move:
        return random.choice(list(self.board.legal_moves))


# Monte Carlo Tree Search (MCTS) skeleton
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_moves = list(board.legal_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        return max(self.children, key=lambda child: child.value / child.visits + c_param * np.sqrt(np.log(self.visits) / child.visits))


# Neural net placeholder
# to be replaced PyTorch model
class DummyNN:
    def predict(self, board: chess.Board):
        return np.random.rand(len(list(board.legal_moves)))


# Example usage:
if __name__ == "__main__":
    ai = ChessAI()
    move = ai.get_best_move_minimax()
    print("Best move (Minimax):", move)
    ai.board.push(move)
    print(ai.board)