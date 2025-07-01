import chess
import chess.engine
import random
from typing import Optional
from ChessNet import ChessNet
import torch
import os


class ChessAI:
    def __init__(self, is_white=True):
        self.is_white = is_white
        self.board = chess.Board()
        self.model = ChessNet()
        model_path = "trained_model_white.pth" if self.is_white else "trained_model_black.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"✅ Model încărcat cu succes din {model_path}")
        else:
            print(f"⚠️ Model neantrenat — se va antrena de la zero ({'alb' if self.is_white else 'negru'}).")
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
            ['epsilon', 'model', 'minimax', 'mcts', 'best_reward'],
            weights=[20.0, 40.0, 40.0, 0.0, 0.0],
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
        elif strategy == 'best_reward':
            return self.select_best_reward_path(board)
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
        if board.is_checkmate():
            return float('-inf') if board.turn == self.is_white else float('inf')
        if board.is_stalemate():
            return -0.5 if board.turn == self.is_white else 0.5

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

        return value

    def select_move_minimax(self, board: chess.Board, depth: int = 2) -> Optional[chess.Move]:
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
        from collections import defaultdict
        import math
        import time

        class MCTSNode:
            def __init__(self, board, parent=None, move=None):
                self.board = board
                self.parent = parent
                self.move = move
                self.children = []
                self.visits = 0
                self.wins = 0
                self.untried_moves = list(board.legal_moves)

            def expand(self):
                move = self.untried_moves.pop()
                next_board = self.board.copy()
                next_board.push(move)
                child_node = MCTSNode(next_board, parent=self, move=move)
                self.children.append(child_node)
                return child_node

            def is_fully_expanded(self):
                return len(self.untried_moves) == 0

            def best_child(self, c_param=1.4):
                choices_weights = [
                    (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
                    for child in self.children
                ]
                return self.children[choices_weights.index(max(choices_weights))]

            def backpropagate(self, result):
                self.visits += 1
                self.wins += result
                if self.parent:
                    self.parent.backpropagate(-result)

            def is_terminal_node(self):
                return self.board.is_game_over()

            def rollout(self):
                sim_board = self.board.copy()
                while not sim_board.is_game_over():
                    legal_moves = list(sim_board.legal_moves)
                    sim_board.push(random.choice(legal_moves))
                result = sim_board.result()
                if result == "1-0":
                    return 1 if self.board.turn == chess.WHITE else -1
                elif result == "0-1":
                    return -1 if self.board.turn == chess.WHITE else 1
                else:
                    return 0

        root = MCTSNode(board)

        for _ in range(simulations):
            node = root
            # Selection
            while not node.is_terminal_node() and node.is_fully_expanded():
                node = node.best_child()
            # Expansion
            if not node.is_terminal_node() and not node.is_fully_expanded():
                node = node.expand()
            # Simulation
            result = node.rollout()
            # Backpropagation
            node.backpropagate(result)

        best_move = max(root.children, key=lambda c: c.visits).move
        return best_move

    def select_best_reward_path(self, board: chess.Board, depth: int = 3) -> Optional[chess.Move]:
        def dfs(board, current_depth):
            if current_depth == 0 or board.is_game_over():
                return self.evaluate_board(board), []

            best_reward = float('-inf')
            best_path = []

            for move in board.legal_moves:
                board.push(move)
                reward, path = dfs(board, current_depth - 1)
                board.pop()

                if reward > best_reward:
                    best_reward = reward
                    best_path = [move] + path

            return best_reward, best_path

        _, best_move_path = dfs(board, depth)
        return best_move_path[0] if best_move_path else None