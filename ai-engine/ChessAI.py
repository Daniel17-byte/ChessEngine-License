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
            ['epsilon', 'model', 'minimax', 'mcts'],
            weights=[40, 30, 0, 30],
            k=1
        )[0]

        # print("Strategy chose : " + strategy)

        if strategy == 'epsilon':
            return random.choice(list(self.board.legal_moves))
        elif strategy == 'model':
            return self.get_best_move_from_model(board)
        elif strategy == 'minimax':
            return self.select_move_minimax(board)
        elif strategy == 'mcts':
            return self.select_move_mcts(board)
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

    def evaluate_board(self, board: chess.Board) -> float:
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

    def select_move_minimax(self, board: chess.Board, depth: int = 5) -> Optional[chess.Move]:
        def minimax(board, depth, alpha, beta, maximizing_player):
            if depth == 0 or board.is_game_over():
                return self.evaluate_board(board), None

            best_move = None
            if maximizing_player:
                max_eval = float('-inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval, _ = minimax(board, depth - 1, alpha, beta, False)
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
                    eval, _ = minimax(board, depth - 1, alpha, beta, True)
                    board.pop()
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval, best_move

        _, best_move = minimax(board, depth, float('-inf'), float('inf'), board.turn)
        return best_move

    def select_move_mcts(self, board: chess.Board, simulations: int = 3) -> Optional[chess.Move]:
        from copy import deepcopy

        def simulate_random_game(sim_board: chess.Board) -> int:
            while not sim_board.is_game_over():
                legal_moves = list(sim_board.legal_moves)
                move = random.choice(legal_moves)
                sim_board.push(move)
            result = sim_board.result()
            if result == "1-0":
                return 1 if board.turn == chess.WHITE else -1
            elif result == "0-1":
                return -1 if board.turn == chess.WHITE else 1
            else:
                return 0  # draw

        legal_moves = list(board.legal_moves)
        move_scores = {move: 0 for move in legal_moves}

        for move in legal_moves:
            total_score = 0
            for _ in range(simulations):
                sim_board = deepcopy(board)
                sim_board.push(move)
                total_score += simulate_random_game(sim_board)
            move_scores[move] = total_score

        best_move = max(move_scores.items(), key=lambda item: item[1])[0]
        return best_move