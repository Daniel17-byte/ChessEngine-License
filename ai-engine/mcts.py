"""
============================================================================
  DaniBot MCTS — Monte Carlo Tree Search
============================================================================

MCTS simplu care folosește ChessNet (policy + value) pentru a explora
linii de joc. Exact ca AlphaZero, dar mai mic/rapid.

Usage în ChessAI:
    from mcts import MCTS
    mcts = MCTS(model, device, move_to_idx, idx_to_move, encode_fn)
    best_move = mcts.search(board, simulations=200)
============================================================================
"""

import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import chess


# ── Dirichlet noise (exploration la root) ─────────────────────────────────

def _add_dirichlet_noise(priors: dict, alpha=0.3, epsilon=0.25):
    """Adaugă noise Dirichlet la root node (AlphaZero-style)."""
    if not priors:
        return priors
    moves = list(priors.keys())
    noise = np.random.dirichlet([alpha] * len(moves))
    return {
        m: (1 - epsilon) * priors[m] + epsilon * n
        for m, n in zip(moves, noise)
    }


# ── MCTS Node ─────────────────────────────────────────────────────────────

class _Node:
    """Un nod în arborele MCTS."""
    __slots__ = ('parent', 'move', 'children', 'visit_count', 'total_value',
                 'prior', 'is_expanded')

    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def q_value(self):
        """Mean value (exploitation)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct=1.5):
        """Upper Confidence Bound = Q + c_puct * P * sqrt(N_parent) / (1 + N)."""
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct=1.5):
        """Selectează copilul cu cel mai mare UCB score."""
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

    def best_move_by_visits(self):
        """Returnează mutarea cu cele mai multe vizite (mai robust decât Q)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visit_count).move


# ── MCTS Engine ───────────────────────────────────────────────────────────

class MCTS:
    """
    Monte Carlo Tree Search folosind ChessNet (policy + value).

    Args:
        model:       ChessNet cu forward() → (policy, value)
        device:      torch device
        move_to_idx: dict UCI string → index
        idx_to_move: list index → UCI string
        encode_fn:   board → numpy array [18, 8, 8]
        c_puct:      exploration constant (default 1.5)
    """

    def __init__(self, model, device, move_to_idx, idx_to_move, encode_fn,
                 c_puct=1.5):
        self.model = model
        self.device = device
        self.move_to_idx = move_to_idx
        self.idx_to_move = idx_to_move
        self.encode_fn = encode_fn
        self.c_puct = c_puct

        # Pre-allocate buffer for single-board inference
        self._buf = torch.zeros(1, 18, 8, 8, dtype=torch.float32)

    def search(self, board: chess.Board, simulations: int = 200,
               add_noise: bool = True) -> chess.Move:
        """
        Rulează MCTS cu N simulări și returnează cea mai bună mutare.

        Args:
            board:       poziția curentă
            simulations: câte simulări (noduri de explorat)
            add_noise:   adaugă Dirichlet noise la root (mai multă explorare)

        Returns:
            chess.Move — cea mai bună mutare găsită
        """
        root = _Node()

        # Evaluează root-ul și expandează
        self._expand(root, board, add_noise=add_noise)

        # Rulează simulări
        for _ in range(simulations):
            node = root
            sim_board = board.copy()

            # 1. SELECT — coboară în arbore urmărind UCB
            while node.is_expanded and node.children:
                node = node.best_child(self.c_puct)
                sim_board.push(node.move)

            # 2. EXPAND & EVALUATE
            if not sim_board.is_game_over():
                value = self._expand(node, sim_board)
            else:
                # Jocul s-a terminat — value exact
                result = sim_board.result()
                if result == '1-0':
                    value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == '0-1':
                    value = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0

            # 3. BACKPROPAGATE — propagă valoarea înapoi
            self._backpropagate(node, value, board.turn)

        # Alege mutarea cu cele mai multe vizite
        best = root.best_move_by_visits()
        if best is None:
            legal = list(board.legal_moves)
            return random.choice(legal) if legal else None
        return best

    def _expand(self, node: _Node, board: chess.Board,
                add_noise: bool = False) -> float:
        """
        Expandează un nod: evaluează cu rețeaua, creează copii.
        Returnează value-ul din perspectiva jucătorului la mutare.
        """
        # Forward pass prin rețea
        policy_probs, value = self._evaluate(board)

        # Adaugă noise la root
        if add_noise and node.parent is None:
            policy_probs = _add_dirichlet_noise(policy_probs)

        # Creează copiii
        for move, prior in policy_probs.items():
            child = _Node(parent=node, move=move, prior=prior)
            node.children.append(child)

        node.is_expanded = True
        return value

    def _evaluate(self, board: chess.Board):
        """
        Rulează rețeaua pe o poziție.
        Returnează (policy_probs: dict[Move→float], value: float).
        """
        # Encode
        arr = self.encode_fn(board)
        buf = self._buf
        buf_np = buf.numpy()
        buf_np[0] = arr

        # Inference
        self.model.eval()
        with torch.inference_mode():
            inp = buf.to(self.device)
            output = self.model(inp)
            # Suportă atât (policy, value) cât și doar policy
            if isinstance(output, tuple):
                policy_logits, value_tensor = output
                value = value_tensor.item()
            else:
                policy_logits = output
                value = 0.0  # fallback dacă nu are value head

        # Softmax pe mutările legale
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, value

        logits = policy_logits.squeeze(0).cpu()
        move_probs = {}
        total = 0.0
        for move in legal_moves:
            idx = self.move_to_idx.get(move.uci(), -1)
            if 0 <= idx < logits.shape[0]:
                # Folosim exp(logit) ca prior nenormalizat
                p = math.exp(min(logits[idx].item(), 30.0))  # clamp overflow
            else:
                p = 1e-6
            move_probs[move] = p
            total += p

        # Normalizează
        if total > 0:
            for m in move_probs:
                move_probs[m] /= total

        return move_probs, value

    def _backpropagate(self, node: _Node, value: float, root_turn):
        """
        Propagă valoarea înapoi prin arbore.
        Alternează semnul la fiecare nivel (adversar vede opusul).
        """
        # Numărăm depth-ul pentru alternarea perspectivei
        depth = 0
        n = node
        while n is not None:
            depth += 1
            n = n.parent

        current = node
        d = 0
        while current is not None:
            current.visit_count += 1
            # Alternează: nodurile la adâncime pară sunt din perspectiva root
            if d % 2 == 0:
                current.total_value += value
            else:
                current.total_value -= value
            current = current.parent
            d += 1

    def get_policy_and_value(self, board: chess.Board, simulations: int = 200):
        """
        Rulează MCTS și returnează distribuția de vizite + valoare estimată.
        Util pentru training — poți folosi visit counts ca target policy.

        Returns:
            (move_visits: dict[Move→int], root_value: float)
        """
        root = _Node()
        self._expand(root, board, add_noise=True)

        for _ in range(simulations):
            node = root
            sim_board = board.copy()

            while node.is_expanded and node.children:
                node = node.best_child(self.c_puct)
                sim_board.push(node.move)

            if not sim_board.is_game_over():
                value = self._expand(node, sim_board)
            else:
                result = sim_board.result()
                if result == '1-0':
                    value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == '0-1':
                    value = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0

            self._backpropagate(node, value, board.turn)

        # Colectează visit counts
        move_visits = {}
        total_visits = 0
        for child in root.children:
            move_visits[child.move] = child.visit_count
            total_visits += child.visit_count

        root_value = root.q_value
        return move_visits, root_value

