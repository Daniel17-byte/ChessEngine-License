import chess
import chess.engine
import random
from typing import Optional
import torch
import numpy as np
import os

# Use Cython-accelerated encoding when available
_USE_CY = os.getenv("CHESS_ENCODE_FORCE_PYTHON", "0") != "1"
try:
    if _USE_CY:
        from fastgame.board_encode import encode_board_array as _cy_encode
    else:
        _cy_encode = None
except ImportError:
    _cy_encode = None

from ArchiveAlpha import encode_board_array
from ChessNet import ChessNet
from mcts import MCTS

def _fast_encode(board):
    """Use Cython encoding if available, else fallback."""
    if _cy_encode is not None:
        return _cy_encode(board)
    return encode_board_array(board)

# Pre-allocate a reusable input buffer for single-board inference (avoids
# torch.from_numpy + unsqueeze + .to() allocation on every move).
_INFER_BUF = torch.zeros(1, 18, 8, 8, dtype=torch.float32)

import json
with open('move_mapping.json', 'r', encoding='utf-8') as fmap:
    move_list = json.load(fmap)
w2i = {m: i for i, m in enumerate(move_list)}
b2i = w2i

engine_path = "/opt/homebrew/bin/stockfish"

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}

USE_CYTHON_EVAL = os.getenv("CHESS_EVAL_FORCE_PYTHON", "0") != "1"
if USE_CYTHON_EVAL:
    try:
        from fastgame.game_eval import evaluate_board as cy_evaluate_board
        HAS_CYTHON_EVAL = True
    except ImportError:
        HAS_CYTHON_EVAL = False
else:
    HAS_CYTHON_EVAL = False


class ChessAI:
    def __init__(self, is_white=True, default_strategy: Optional[str] = None, load_model_from_disk: bool = True):
        self.is_white = is_white
        self.board = chess.Board()
        # Select device: prefer MPS on Mac, then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = ChessNet(len(move_list))
        self.model.to(self.device)
        model_path = "danibot.pth"
        if load_model_from_disk and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Model loaded successfully from {model_path}")
            except (RuntimeError, KeyError) as e:
                print(f"⚠️ Could not load model from {model_path}: {e}")
                print("Architecture mismatch — training from scratch.")
                # Remove old incompatible checkpoint
                os.rename(model_path, model_path + ".old")
                print(f"Old model backed up to {model_path}.old")
        elif load_model_from_disk:
            print(f"Model not found ({model_path}) - training from scratch.")

        # Only load Stockfish if the strategy might need it
        needs_stockfish = default_strategy in (None, 'stockfish')
        if needs_stockfish and os.path.exists(engine_path):
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        elif needs_stockfish:
            self.engine = None
            print(f"⚠️ Stockfish nu a fost găsit la {engine_path}. Strategy 'stockfish' nu va fi disponibil.")
        else:
            self.engine = None
            print(f"ℹ️ Stockfish nu e necesar pentru strategia '{default_strategy}' — nu se încarcă.")

        # Reuse already-loaded mapping to avoid duplicate file IO during init.
        self.idx_to_move = move_list
        # Keep mapping fixed to trained output space for faster inference-time lookup.
        self.move_to_idx = {uci: i for i, uci in enumerate(self.idx_to_move)}

        self.model.eval()
        # Optional PyTorch compile for faster repeated inference on supported backends.
        if os.getenv("CHESS_USE_TORCH_COMPILE", "0") == "1" and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as exc:
                print(f"⚠️ torch.compile disabled: {exc}")

        self.epsilon = 0.05
        self.default_strategy = default_strategy

        # Initialize MCTS engine for search-based play
        self.mcts = MCTS(
            model=self.model,
            device=self.device,
            move_to_idx=self.move_to_idx,
            idx_to_move=self.idx_to_move,
            encode_fn=_fast_encode,
        )

    def move_to_index(self, move_uci: str) -> int:
        return self.move_to_idx.get(move_uci, -1)

    def select_move(self, board: chess.Board, strategy: Optional[str] = None) -> Optional[chess.Move]:
        self.board = board
        if strategy is None:
            strategy = self.default_strategy

        if strategy == 'epsilon':
            return random.choice(list(self.board.legal_moves))
        elif strategy == 'model':
            return self.get_best_move_from_model(board)
        elif strategy == 'mcts':
            return self.mcts.search(board, simulations=200)
        elif strategy == 'minimax':
            return self.select_move_minimax(board)
        elif strategy == 'stockfish':
            return self.get_best_move_from_stockfish(board)
        return None

    def get_best_move_from_stockfish(self, board: chess.Board, time_limit: float = 0.01) -> Optional[chess.Move]:
        if self.engine is None:
            return random.choice(list(board.legal_moves))
        if board.is_game_over():
            return None
        result = self.engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move

    def _board_cache_key(self, board: chess.Board):
        # python-chess transposition key avoids expensive FEN string construction.
        if hasattr(board, "_transposition_key"):
            return board._transposition_key()
        return f"{board.board_fen()}|{int(board.turn)}|{board.castling_xfen()}|{board.ep_square}"

    def _predict_policy(self, board: chess.Board, cache: dict) -> torch.Tensor:
        cache_key = self._board_cache_key(board)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        board_array = _fast_encode(board)
        board_tensor = torch.from_numpy(board_array).unsqueeze(0).to(self.device, non_blocking=(self.device.type == 'cuda'))
        with torch.inference_mode():
            output = self.model(board_tensor)
            # Support both (policy, value) and policy-only models
            if isinstance(output, tuple):
                pred = output[0].squeeze(0)
            else:
                pred = output.squeeze(0)
        cache[cache_key] = pred
        return pred

    def _predict_policy_and_value(self, board: chess.Board, cache: dict):
        """Predict policy + value. Returns (policy_tensor, value_float)."""
        cache_key = ('pv', self._board_cache_key(board))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        board_array = _fast_encode(board)
        board_tensor = torch.from_numpy(board_array).unsqueeze(0).to(self.device, non_blocking=(self.device.type == 'cuda'))
        with torch.inference_mode():
            output = self.model(board_tensor)
            if isinstance(output, tuple):
                policy = output[0].squeeze(0)
                value = output[1].item()
            else:
                policy = output.squeeze(0)
                value = 0.0
        result = (policy, value)
        cache[cache_key] = result
        return result

    def _predict_policy_cpu(self, board: chess.Board, cache: dict, cpu_model) -> np.ndarray:
        """Ultra-fast policy prediction on CPU returning numpy array.

        During self-play training, MPS/CUDA transfer overhead for batch=1
        dominates inference time.  Running on CPU with a pre-copied model
        and returning numpy avoids all tensor creation / device transfer.
        """
        cache_key = self._board_cache_key(board)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        board_array = _fast_encode(board)
        # Reuse global buffer to avoid per-call allocation
        buf = _INFER_BUF
        buf_np = buf.numpy()
        buf_np[0] = board_array  # direct memory copy, no tensor creation
        with torch.inference_mode():
            pred_output = cpu_model(buf)
        # Support both (policy, value) and policy-only
        if isinstance(pred_output, tuple):
            pred = pred_output[0].squeeze(0).numpy()
        else:
            pred = pred_output.squeeze(0).numpy()
        cache[cache_key] = pred
        return pred

    def _get_legal_and_top_model_moves(self, board: chess.Board, prediction: torch.Tensor, top_n: int, cache: dict):
        cache_key = (self._board_cache_key(board), top_n)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            result = ([], [])
            cache[cache_key] = result
            return result

        output_size = prediction.shape[0]
        move_idx_pairs = []
        for move in legal_moves:
            idx = self.move_to_index(move.uci())
            if 0 <= idx < output_size:
                move_idx_pairs.append((move, idx))

        if not move_idx_pairs:
            result = (legal_moves, [])
            cache[cache_key] = result
            return result

        idx_tensor = torch.tensor([idx for _, idx in move_idx_pairs], device=prediction.device, dtype=torch.long)
        legal_scores = prediction.index_select(0, idx_tensor)
        k = min(top_n, legal_scores.shape[0])
        top_pos = torch.topk(legal_scores, k=k).indices.tolist()
        top_moves = [move_idx_pairs[pos][0] for pos in top_pos]
        result = (legal_moves, top_moves)
        cache[cache_key] = result
        return result

    def _top_legal_model_moves(self, board: chess.Board, prediction: torch.Tensor, top_n: int):
        return self._get_legal_and_top_model_moves(board, prediction, top_n, cache={})

    def get_best_move_from_model(self, board: chess.Board, top_n: int = 1, depth: int = 1) -> Optional[chess.Move]:
        self.board = board

        prediction_cache = {}
        move_cache = {}
        eval_cache = {}
        prediction = self._predict_policy(board, prediction_cache)
        legal_moves, top_moves = self._get_legal_and_top_model_moves(board, prediction, top_n, move_cache)

        if not legal_moves:
            return None
        if not top_moves:
            return random.choice(legal_moves)

        def nn_eval(b):
            """Use neural network value head for position evaluation."""
            _, val = self._predict_policy_and_value(b, prediction_cache)
            return val

        def model_minimax(board_, depth_, alpha, beta, maximizing):
            key = (self._board_cache_key(board_), depth_, maximizing)
            cached_eval = eval_cache.get(key)
            if cached_eval is not None:
                return cached_eval

            if depth_ == 0 or board_.is_game_over():
                result = (nn_eval(board_), None)
                eval_cache[key] = result
                return result

            pred_ = self._predict_policy(board_, prediction_cache)
            moves_, top_moves_ = self._get_legal_and_top_model_moves(board_, pred_, top_n, move_cache)
            if not moves_:
                result = (nn_eval(board_), None)
                eval_cache[key] = result
                return result
            if not top_moves_:
                top_moves_ = moves_

            best_eval = float('-inf') if maximizing else float('inf')
            best_move_ = None

            for move in top_moves_:
                board_.push(move)
                eval_, _ = model_minimax(board_, depth_ - 1, alpha, beta, not maximizing)
                board_.pop()

                if maximizing:
                    if eval_ > best_eval:
                        best_eval = eval_
                        best_move_ = move
                    alpha = max(alpha, best_eval)
                else:
                    if eval_ < best_eval:
                        best_eval = eval_
                        best_move_ = move
                    beta = min(beta, best_eval)

                if beta <= alpha:
                    break

            result = (best_eval, best_move_)
            eval_cache[key] = result
            return result

        maximizing = board.turn == (chess.WHITE if self.is_white else chess.BLACK)
        _, best_move = model_minimax(board, depth, float('-inf'), float('inf'), maximizing)
        if best_move and best_move in legal_moves:
            return best_move

        return top_moves[0] if top_moves else random.choice(legal_moves)

    def _evaluate_board_python(self, board: chess.Board) -> float:
        my_color = chess.WHITE if self.is_white else chess.BLACK
        if board.is_checkmate():
            return float('-inf') if board.turn == my_color else float('inf')

        value = 0.0
        for square, piece in board.piece_map().items():
            sign = 1.0 if piece.color == chess.WHITE else -1.0
            val = PIECE_VALUES[piece.piece_type]
            if square in CENTER_SQUARES:
                val += 0.1
            value += sign * val

        if board.is_check():
            value += 0.5 if board.turn != my_color else -0.5

        return value if self.is_white else -value

    def evaluate_board(self, board: chess.Board) -> float:
        if HAS_CYTHON_EVAL:
            return cy_evaluate_board(board, self.is_white)
        return self._evaluate_board_python(board)

    def select_move_minimax(self, board: chess.Board, depth: int = 2) -> Optional[chess.Move]:
        eval_cache = {}

        def move_order_key(board_ctx: chess.Board, move_: chess.Move):
            # Captures/checks first -> better alpha-beta pruning.
            score = 0
            if board_ctx.is_capture(move_):
                score += 100
            if board_ctx.gives_check(move_):
                score += 10
            return score

        def minimax(board_, depth_, alpha, beta, maximizing_player):
            key = (self._board_cache_key(board_), depth_, maximizing_player)
            cached = eval_cache.get(key)
            if cached is not None:
                return cached

            if depth_ == 0 or board_.is_game_over():
                result = (self.evaluate_board(board_), None)
                eval_cache[key] = result
                return result

            ordered_moves = sorted(board_.legal_moves, key=lambda m: move_order_key(board_, m), reverse=True)
            if not ordered_moves:
                result = (self.evaluate_board(board_), None)
                eval_cache[key] = result
                return result

            best_move_ = None
            if maximizing_player:
                max_eval = float('-inf')
                for move in ordered_moves:
                    board_.push(move)
                    eval_, _ = minimax(board_, depth_ - 1, alpha, beta, False)
                    board_.pop()
                    if eval_ > max_eval:
                        max_eval = eval_
                        best_move_ = move
                    alpha = max(alpha, max_eval)
                    if beta <= alpha:
                        break
                result = (max_eval, best_move_)
            else:
                min_eval = float('inf')
                for move in ordered_moves:
                    board_.push(move)
                    eval_, _ = minimax(board_, depth_ - 1, alpha, beta, True)
                    board_.pop()
                    if eval_ < min_eval:
                        min_eval = eval_
                        best_move_ = move
                    beta = min(beta, min_eval)
                    if beta <= alpha:
                        break
                result = (min_eval, best_move_)

            eval_cache[key] = result
            return result

        _, best_move = minimax(board, depth, float('-inf'), float('inf'), board.turn == (chess.WHITE if self.is_white else chess.BLACK))
        return best_move

    def get_fast_move_from_model(
        self,
        board: chess.Board,
        top_n: int = 3,
        sample_top_k: int = 2,
        prediction_cache: Optional[dict] = None,
        move_cache: Optional[dict] = None,
        cpu_model=None,
    ) -> Optional[chess.Move]:
        """Fast policy move for self-play throughput (no minimax search).

        When cpu_model is provided, runs inference entirely on CPU with numpy
        to avoid MPS/CUDA device-transfer overhead for batch-size=1.
        This gives ~3-10x speedup over the default device path.
        """
        prediction_cache = prediction_cache if prediction_cache is not None else {}
        move_cache = move_cache if move_cache is not None else {}

        # --- Fast CPU path (numpy) ---
        if cpu_model is not None:
            pred_np = self._predict_policy_cpu(board, prediction_cache, cpu_model)

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None

            move_to_idx = self.move_to_idx
            output_size = pred_np.shape[0]

            # Build parallel arrays of (move, score) for legal moves
            moves_arr = []
            scores_arr = []
            for move in legal_moves:
                idx = move_to_idx.get(move.uci(), -1)
                if 0 <= idx < output_size:
                    moves_arr.append(move)
                    scores_arr.append(pred_np[idx])

            if not moves_arr:
                return random.choice(legal_moves)

            # Numpy argpartition is O(n) vs O(n log n) for full sort
            scores = np.array(scores_arr, dtype=np.float32)
            k = min(top_n, len(scores))
            if k >= len(scores):
                top_indices = np.argsort(scores)[::-1][:k]
            else:
                top_indices = np.argpartition(scores, -k)[-k:]
                # Sort the top-k for consistent ordering
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            top_moves = [moves_arr[i] for i in top_indices]
            sk = min(max(sample_top_k, 1), len(top_moves))
            return random.choice(top_moves[:sk])

        # --- Standard device path (torch) ---
        prediction = self._predict_policy(board, prediction_cache)
        legal_moves, top_moves = self._get_legal_and_top_model_moves(board, prediction, top_n, move_cache)
        if not legal_moves:
            return None
        if not top_moves:
            return random.choice(legal_moves)

        k = min(max(sample_top_k, 1), len(top_moves))
        return random.choice(top_moves[:k])
