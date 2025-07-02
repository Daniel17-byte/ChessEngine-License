import chess
import chess.engine
import random
from typing import Optional
from ArchiveAlpha import encode_board
from ChessNet import ChessNet
import torch
import os

import json
with open('move_mapping.json', 'r', encoding='utf-8') as fmap:
    move_list = json.load(fmap)
w2i = {m: i for i, m in enumerate(move_list)}
b2i = w2i

class ChessAI:
    def __init__(self, is_white=True, default_strategy: Optional[str] = None):
        self.is_white = is_white
        self.board = chess.Board()
        self.model = ChessNet(len(move_list))
        model_path = "trained_model_white.pth" if self.is_white else "trained_model_black.pth"
        if os.path.exists(model_path):
            try:

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                net_b = ChessNet(len(move_list))
                net_b.to(device)

                print(f"✅ Model încărcat cu succes din {model_path}")
            except RuntimeError as e:
                print(f"⚠️ Nu s-a putut încărca modelul din {model_path}: {e}")
                print("⚠️ Se va antrena de la zero.")
        else:
            print(f"⚠️ Modelul nu a fost găsit ({model_path}) — se va antrena de la zero.")

        import json
        with open("move_mapping.json") as f:
            self.idx_to_move = json.load(f)
        self.move_to_idx = {uci: i for i, uci in enumerate(self.idx_to_move)}

        self.model.eval()
        self.epsilon = 0.2
        self.default_strategy = default_strategy

    def move_to_index(self, move_uci: str) -> int:
        if move_uci not in self.move_to_idx:
            self.move_to_idx[move_uci] = len(self.idx_to_move)
            self.idx_to_move.append(move_uci)
        return self.move_to_idx[move_uci]

    def select_move(self, board: chess.Board, strategy: Optional[str] = None) -> Optional[chess.Move]:
        self.board = board
        if strategy is None:
            strategy = self.default_strategy
        if strategy is None:
            strategy = random.choices(
                ['epsilon', 'model', 'minimax'],
                weights=[30.0, 40.0, 30.0],
                k=1
            )[0]

        if strategy == 'epsilon':
            return random.choice(list(self.board.legal_moves))
        elif strategy == 'model':
            return self.get_best_move_from_model(board)
        elif strategy == 'minimax':
            return self.select_move_minimax(board)
        return None

    def get_best_move_from_model(self, board: chess.Board) -> Optional[chess.Move]:
        self.board = board
        board_tensor = encode_board(board).unsqueeze(0)

        legal_moves = list(board.legal_moves)
        legal_indices = [self.move_to_index(m.uci()) for m in legal_moves]

        with torch.no_grad():
            prediction = self.model(board_tensor).squeeze(0)

        rand_val = random.random()
        if rand_val < self.epsilon:
            return random.choice(legal_moves)

        best_idx = max(legal_indices, key=lambda i: prediction[i].item())
        best_move = chess.Move.from_uci(self.idx_to_move[best_idx])

        if best_move not in self.board.legal_moves:
            print(f"⚠️ Predicted move {best_move} is not legal in the current board position.")
            return random.choice(legal_moves)

        return best_move

    def evaluate_board(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return float('-inf') if board.turn == self.is_white else float('inf')

        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.2,
            chess.BISHOP: 3.3,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0
        }

        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        value = 0.0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                sign = 1 if piece.color == chess.WHITE else -1
                piece_type = piece.piece_type
                val = piece_values[piece_type]

                # Bonus for controlling the center
                if square in center_squares:
                    val += 0.1

                value += sign * val

        if board.is_check():
            value += 0.5 if board.turn != self.is_white else -0.5

        return value

    def select_move_minimax(self, board: chess.Board, depth: int = 2) -> Optional[chess.Move]:
        def minimax(board_, depth_, alpha, beta, maximizing_player):
            if depth_ == 0 or board_.is_game_over():
                return self.evaluate_board(board_), None

            best_move_ = None
            if maximizing_player:
                max_eval = float('-inf')
                for move in board_.legal_moves:
                    board_.push(move)
                    eval_, _ = minimax(board_, depth_ - 1, alpha, beta, False)
                    board_.pop()
                    if eval_ > max_eval:
                        max_eval = eval_
                        best_move_ = move
                    alpha = max(alpha, eval_)
                    if beta <= alpha:
                        break
                return max_eval, best_move_
            else:
                min_eval = float('inf')
                for move in board_.legal_moves:
                    board_.push(move)
                    eval_, _ = minimax(board_, depth_ - 1, alpha, beta, True)
                    board_.pop()
                    if eval_ < min_eval:
                        min_eval = eval_
                        best_move_ = move
                    beta = min(beta, eval_)
                    if beta <= alpha:
                        break
                return min_eval, best_move_

        _, best_move = minimax(board, depth, float('-inf'), float('inf'), board.turn)
        return best_move