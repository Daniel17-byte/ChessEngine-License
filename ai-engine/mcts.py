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


def _terminal_value(board: chess.Board) -> float:
    """Valoarea unei poziții terminale din perspectiva jucătorului LA MUTARE.

    Dacă e mat, cel la mutare a pierdut => -1. Orice altă terminare e remiză.
    """
    if board.is_checkmate():
        return -1.0
    return 0.0


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
                 c_puct=1.5, cache_size=200_000, reuse_tree=False):
        self.model = model
        self.device = device
        self.move_to_idx = move_to_idx
        self.idx_to_move = idx_to_move
        self.encode_fn = encode_fn
        self.c_puct = c_puct

        # Pre-allocate buffer for single-board inference
        self._buf = torch.zeros(1, 18, 8, 8, dtype=torch.float32)

        # Profiling shows ~90% of search time is the network forward pass, and the
        # same position recurs constantly both within one search and across moves
        # of a game. Caching evaluations is the single cheapest win available.
        self._cache = {}
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

        # Optional tree reuse between consecutive moves of a game.
        self.reuse_tree = reuse_tree
        self._root = None
        self._root_key = None

        # model.eval() used to run on every evaluation, walking every submodule.
        self.model.eval()

    @staticmethod
    def _position_key(board):
        """Key covering exactly the fields the encoder reads.

        Not board.fen(): that includes move counters the network never sees, which
        would split cache entries that encode identically. Not _transposition_key()
        either — it drops ep_square when no en-passant capture is legal, but the
        encoder always writes an ep plane, so it would merge entries that differ.
        """
        return (board.pawns, board.knights, board.bishops, board.rooks,
                board.queens, board.kings,
                board.occupied_co[True], board.occupied_co[False],
                board.turn, board.castling_rights, board.ep_square)

    def clear_cache(self):
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def reset_tree(self):
        """Drop the retained tree (call when starting a new game)."""
        self._root = None
        self._root_key = None

    def advance(self, move):
        """Descend the retained tree into `move`, keeping its accumulated stats."""
        if not self.reuse_tree or self._root is None:
            return
        for child in self._root.children:
            if child.move == move:
                child.parent = None
                self._root = child
                self._root_key = None  # re-checked against the board on next search
                return
        self.reset_tree()

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
        root = self._acquire_root(board, add_noise=add_noise)

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
                value = _terminal_value(sim_board)

            # 3. BACKPROPAGATE — propagă valoarea înapoi
            self._backpropagate(node, value)

        # Alege mutarea cu cele mai multe vizite
        best = root.best_move_by_visits()
        if best is None:
            legal = list(board.legal_moves)
            return random.choice(legal) if legal else None
        return best

    def _acquire_root(self, board: chess.Board, add_noise: bool = True) -> _Node:
        """Return the root for this search, reusing the retained subtree if valid.

        After `advance(move)` the retained node already carries visit counts from
        the previous search, so those simulations are not repeated.
        """
        key = self._position_key(board)

        if self.reuse_tree and self._root is not None:
            if self._root_key is None or self._root_key == key:
                root = self._root
                self._root_key = key
                if not root.is_expanded:
                    self._expand(root, board, add_noise=add_noise)
                elif add_noise:
                    # Re-apply root exploration noise to the reused priors.
                    priors = {c.move: c.prior for c in root.children}
                    noisy = _add_dirichlet_noise(priors)
                    for c in root.children:
                        c.prior = noisy.get(c.move, c.prior)
                return root
            self.reset_tree()

        root = _Node()
        self._expand(root, board, add_noise=add_noise)
        if self.reuse_tree:
            self._root = root
            self._root_key = key
        return root

    def _expand(self, node: _Node, board: chess.Board,
                add_noise: bool = False) -> float:
        """
        Expandează un nod: evaluează cu rețeaua, creează copii.
        Returnează value-ul din perspectiva jucătorului la mutare.
        """
        # Forward pass prin rețea
        policy_probs, value = self._evaluate(board)

        # Adaugă noise la root
        if add_noise:
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
        key = self._position_key(board)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1

        # Encode
        arr = self.encode_fn(board)
        buf = self._buf
        buf_np = buf.numpy()
        buf_np[0] = arr

        # Inference
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
            result = ({}, value)
            self._store(key, result)
            return result

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

        result = (move_probs, value)
        self._store(key, result)
        return result

    def _store(self, key, result):
        if len(self._cache) >= self._cache_size:
            self._cache.clear()  # cheap bounded-memory policy; hit rate recovers fast
        self._cache[key] = result

    def _backpropagate(self, node: _Node, value: float):
        """Propagă valoarea înapoi prin arbore, alternând perspectiva.

        Convenție: X.total_value acumulează valoarea din perspectiva jucătorului
        care a mutat ÎN X (adică cel la mutare în X.parent). Doar așa are sens ca
        `best_child` să maximizeze q_value-ul copiilor: părintele își alege mutarea
        care e cea mai bună pentru EL.

        `value` vine din perspectiva jucătorului la mutare în `node`, deci pentru
        `node` însuși semnul se inversează.
        """
        current = node
        sign = -1.0
        while current is not None:
            current.visit_count += 1
            current.total_value += sign * value
            current = current.parent
            sign = -sign

    def get_policy_and_value(self, board: chess.Board, simulations: int = 200):
        """
        Rulează MCTS și returnează distribuția de vizite + valoare estimată.
        Util pentru training — poți folosi visit counts ca target policy.

        Returns:
            (move_visits: dict[Move→int], root_value: float)
        """
        root = self._acquire_root(board)

        for _ in range(simulations):
            node = root
            sim_board = board.copy()

            while node.is_expanded and node.children:
                node = node.best_child(self.c_puct)
                sim_board.push(node.move)

            if not sim_board.is_game_over():
                value = self._expand(node, sim_board)
            else:
                value = _terminal_value(sim_board)

            self._backpropagate(node, value)

        # Colectează visit counts
        move_visits = {}
        total_visits = 0
        for child in root.children:
            move_visits[child.move] = child.visit_count
            total_visits += child.visit_count

        # root.total_value e în perspectiva jucătorului care a mutat în root,
        # adică adversarul celui la mutare — inversăm ca să returnăm valoarea
        # din perspectiva jucătorului la mutare, cum se așteaptă apelantul.
        root_value = -root.q_value
        return move_visits, root_value

