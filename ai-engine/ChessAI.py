import chess
import chess.engine
import random
from typing import Optional
from ChessNet import ChessNet
import torch
import os


class ChessAI:
    def __init__(self, is_white=True, default_strategy: Optional[str] = None):
        self.is_white = is_white
        self.board = chess.Board()
        self.model = ChessNet()
        model_path = "trained_model_white.pth" if self.is_white else "trained_model_black.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"✅ Model încărcat cu succes din {model_path}")
            print(f"📊 Sumă ponderi model: {sum(p.sum().item() for p in self.model.parameters()):.4f}")
        else:
            print(f"⚠️ Model neantrenat — se va antrena de la zero ({'alb' if self.is_white else 'negru'}).")
        self.model.eval()
        import json
        with open("move_mapping.json") as f:
            self.idx_to_move = json.load(f)
        self.move_to_idx = {uci: i for i, uci in enumerate(self.idx_to_move)}
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
                weights=[60.0, 20.0, 20.0],
                k=1
            )[0]

        # print("Strategy chose : " + strategy)

        if strategy == 'epsilon':
            return random.choice(list(self.board.legal_moves))
        elif strategy == 'model':
            return self.get_best_move_from_model(board)
        elif strategy == 'minimax':
            return self.select_move_minimax(board)
        return None

    def get_best_move_from_model(self, board: chess.Board) -> Optional[chess.Move]:
        self.board = board
        print(f"🧠 Tip model: {type(self.model)}")
        model_file = "trained_model_white.pth" if self.is_white else "trained_model_black.pth"
        print(f"📂 Fișier model încărcat: {model_file}")
        from ChessNet import encode_fen
        board_tensor = encode_fen(board.fen()).unsqueeze(0)
        # DEBUG after tensor creation
        print(f"📏 Shape tensor: {board_tensor.shape}")
        print(f"📊 Valori tensor: {board_tensor[0][:50]}")
        print("🧮 Sumă tensor intrare:", board_tensor.sum().item())

        legal_moves = list(board.legal_moves)
        legal_indices = [self.move_to_index(m.uci()) for m in legal_moves]

        with torch.no_grad():
            prediction = self.model(board_tensor).squeeze(0)
        # DEBUG after prediction
        print(f"📉 Primii 20 scoruri brute: {[round(v.item(), 2) for v in prediction[:20]]}")
        print(f"📈 Max scor total: {prediction.max().item():.2f}")
        print(f"📉 Min scor total: {prediction.min().item():.2f}")
        print(f"🧾 Medie scoruri: {prediction.mean().item():.2f}")

        print("🔢 Dimensiune prediction:", prediction.shape)
        print("📥 Predictii brute (primele 10):", [round(prediction[i].item(), 2) for i in legal_indices[:10]])

        # DEBUG after legal_indices
        print(f"🔍 Mutări legale (UCI): {[m.uci() for m in legal_moves]}")
        print(f"🔢 Indici mutări legale: {legal_indices}")
        print(f"🎯 Scoruri mutări legale: {[round(prediction[i].item(), 2) for i in legal_indices]}")

        rand_val = random.random()
        print(f"🎲 Epsilon: {self.epsilon}, Random threshold: {rand_val}")
        if rand_val < self.epsilon:
            return random.choice(legal_moves)

        best_idx = max(legal_indices, key=lambda i: prediction[i].item())
        print(f"🧠 Model alege mutarea: {self.idx_to_move[best_idx]} cu scor {prediction[best_idx].item():.2f}")
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

    def select_move_minimax(self, board: chess.Board, depth: int = 3) -> Optional[chess.Move]:
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