"""
============================================================================
  Mirror Match — AlphaZero-style self-play training
============================================================================

Fiecare mutare din self-play e aleasă de MCTS, iar rețeaua învață:

  - policy: distribuția de vizite a MCTS (π), nu mutarea jucată. Căutarea e
    mai bună decât rețeaua brută, deci π e un target *îmbunătățit* — asta e
    ce face self-play-ul să progreseze.
  - value:  rezultatul final al partidei (z), din perspectiva jucătorului
    la mutare în poziția respectivă.

Modelul nou înlocuiește `danibot.pth` doar dacă trece un meci de gating
împotriva modelului curent. Fără gating, self-play-ul poate regresa liniștit.

Varianta veche clona mutările învingătorului (behavioral cloning). La nivelul
ăsta învingătorul câștigă pentru că adversarul dă blunder, deci învăța mutări
mediocre etichetate drept bune, iar capul de valoare nu primea niciun target.
============================================================================
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import Counter, deque

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ArchiveAlpha import encode_board_array
from ChessNet import ChessNet
from mcts import MCTS
from TrainingGame import TrainingGame

with open('move_mapping.json', 'r', encoding='utf-8') as f:
    move_list = json.load(f)
move_to_idx = {m: i for i, m in enumerate(move_list)}
N_MOVES = len(move_list)

RESULT_VALUE = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}


def print_status(epoch, max_epochs, loss, winner, stats):
    """Print status as JSON for parsing by app.py"""
    status = {
        "epoch": epoch,
        "max_epochs": max_epochs,
        "loss": round(loss, 4),
        "winner": winner,
        "white_reward": 0,
        "black_reward": 0,
        "stats": {
            "white_wins": stats.get('1-0', 0),
            "black_wins": stats.get('0-1', 0),
            "draws": stats.get('1/2-1/2', 0)
        }
    }
    print(json.dumps(status))
    sys.stdout.flush()


def emit(**payload):
    print(json.dumps(payload))
    sys.stdout.flush()


# ── model helpers ────────────────────────────────────────────────────────────

def load_model(path, device):
    model = ChessNet(N_MOVES)
    if path and os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location='cpu'))
        except (RuntimeError, KeyError) as exc:
            emit(warning=f"could not load {path}: {exc}; starting from scratch")
    return model.to(device).eval()


def value_head_is_dead(model, samples=120, tol=1e-6):
    """True when the value head returns a constant for every position.

    A network trained with all-zero value targets outputs exactly 0 everywhere.
    MCTS then evaluates every leaf as a draw and degenerates into resampling the
    policy priors, so self-play built on it cannot improve anything.
    """
    board = chess.Board()
    values = []
    with torch.inference_mode():
        for _ in range(samples):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                board = chess.Board()
                continue
            board.push(random.choice(legal))
            x = torch.from_numpy(encode_board_array(board)).unsqueeze(0)
            _, v = model(x)
            values.append(v.item())
    if len(values) < 2:
        return False
    return float(np.std(values)) < tol


def freeze_for_inference(model):
    """Return a JIT-frozen, inference-only copy of `model`.

    Measured ~1.23x over eager on this network with identical outputs (conv+BN
    fusion only). The result cannot be trained, so self-play and gating run on a
    frozen snapshot while the eager `candidate` keeps training.
    """
    snapshot = copy.deepcopy(model).cpu().eval()
    for param in snapshot.parameters():
        param.requires_grad_(False)
    try:
        with torch.inference_mode():
            example = torch.zeros(1, 18, 8, 8)
            traced = torch.jit.freeze(torch.jit.trace(snapshot, example))
            traced(example)  # warm up
        return traced
    except Exception as exc:
        emit(warning=f"JIT freeze failed, using eager model: {exc}")
        return snapshot


def make_mcts(model, reuse_tree=False):
    """Search helper.

    ~90% of search time is the forward pass, so the evaluation cache is what makes
    MCTS self-play affordable (measured 1.36x on a fixed 30-ply line).

    Tree reuse defaults to off: it is *not* a speedup. Retaining the tree pushes
    the search into deeper, fresher positions, which drops the cache hit rate from
    28% to 8.5% and costs ~20% more per move. It buys search quality (visits
    accumulate across moves) rather than throughput, so it is opt-in.
    """
    return MCTS(model, torch.device('cpu'), move_to_idx, move_list,
                encode_board_array, reuse_tree=reuse_tree)


# ── self-play ────────────────────────────────────────────────────────────────

def visits_to_policy(visits, temperature):
    """Turn MCTS visit counts into a target distribution over the move mapping."""
    idxs, counts = [], []
    for move, count in visits.items():
        idx = move_to_idx.get(move.uci())
        if idx is not None and count > 0:
            idxs.append(idx)
            counts.append(float(count))
    if not idxs:
        return None, None
    counts = np.asarray(counts, dtype=np.float64)
    if temperature <= 1e-3:
        probs = np.zeros_like(counts)
        probs[int(counts.argmax())] = 1.0
    else:
        scaled = counts ** (1.0 / temperature)
        probs = scaled / scaled.sum()
    return np.asarray(idxs, dtype=np.int64), probs.astype(np.float32)


def play_selfplay_game(mcts, game, simulations, opening_plies, temp_moves,
                       max_plies, fen_positions=None):
    """One self-play game. Returns (samples, result).

    A sample is (encoded_board, move_indices, target_probs, side_to_move).
    The value target is filled in once the game ends.
    """
    if fen_positions:
        game.reset_from_fen(random.choice(fen_positions))
    else:
        game.reset()

    mcts.reset_tree()
    samples = []
    ply = 0

    while not game.is_game_over() and ply < max_plies:
        board = game.board
        legal = list(board.legal_moves)
        if not legal:
            break

        # Random opening plies purely for diversity; not used as training targets,
        # since a random move is not something we want the policy to imitate.
        if ply < opening_plies:
            move = random.choice(legal)
            ok, _ = game.make_move_fast(move)
            if not ok:
                break
            mcts.advance(move)
            ply += 1
            continue

        visits, _ = mcts.get_policy_and_value(board, simulations=simulations)
        if not visits:
            move = random.choice(legal)
            ok, _ = game.make_move_fast(move)
            if not ok:
                break
            mcts.advance(move)
            ply += 1
            continue

        # τ=1 early keeps games diverse; τ→0 later makes play sharp.
        temperature = 1.0 if ply < opening_plies + temp_moves else 0.0
        idxs, probs = visits_to_policy(visits, temperature)
        if idxs is not None:
            samples.append((
                encode_board_array(board).copy(),
                idxs,
                probs,
                board.turn,
            ))

        # Sample the actual move from the visit counts (τ=1) or take the max (τ=0).
        moves = list(visits.keys())
        counts = np.asarray([visits[m] for m in moves], dtype=np.float64)
        if temperature > 1e-3 and counts.sum() > 0:
            move = moves[int(np.random.choice(len(moves), p=counts / counts.sum()))]
        else:
            move = moves[int(counts.argmax())]

        ok, _ = game.make_move_fast(move)
        if not ok:
            break
        mcts.advance(move)
        ply += 1

    result = game.get_result()
    if result not in RESULT_VALUE:
        result = '1/2-1/2'

    z_white = RESULT_VALUE[result]
    finished = [
        (x, idxs, probs, z_white if turn == chess.WHITE else -z_white)
        for x, idxs, probs, turn in samples
    ]
    return finished, result


# ── gating match ─────────────────────────────────────────────────────────────

def play_match_game(mcts_a, mcts_b, game, simulations, a_is_white, opening_plies,
                    max_plies):
    """One deterministic game between two searchers. Returns the result string."""
    game.reset()
    mcts_a.reset_tree()
    mcts_b.reset_tree()
    ply = 0
    while not game.is_game_over() and ply < max_plies:
        board = game.board
        legal = list(board.legal_moves)
        if not legal:
            break
        if ply < opening_plies:
            move = random.choice(legal)
        else:
            white_to_move = board.turn == chess.WHITE
            searcher = mcts_a if (white_to_move == a_is_white) else mcts_b
            move = searcher.search(board, simulations=simulations, add_noise=False)
            if move is None:
                move = random.choice(legal)
        ok, _ = game.make_move_fast(move)
        if not ok:
            break
        mcts_a.advance(move)
        mcts_b.advance(move)
        ply += 1
    result = game.get_result()
    return result if result in RESULT_VALUE else '1/2-1/2'


def gating_match(candidate, champion, games, simulations, opening_plies, max_plies,
                 reuse_tree=False):
    """Score the candidate against the champion. Returns (score, wins, draws, losses).

    Score counts a draw as half a point, so 0.5 means "no better than the
    incumbent". Colors alternate so an opening advantage cannot decide it.
    """
    mcts_c = make_mcts(freeze_for_inference(candidate), reuse_tree=reuse_tree)
    mcts_h = make_mcts(freeze_for_inference(champion), reuse_tree=reuse_tree)
    game = TrainingGame()
    wins = draws = losses = 0

    for i in range(games):
        cand_is_white = (i % 2 == 0)
        result = play_match_game(mcts_c, mcts_h, game, simulations, cand_is_white,
                                 opening_plies, max_plies)
        if result == '1/2-1/2':
            draws += 1
        elif (result == '1-0') == cand_is_white:
            wins += 1
        else:
            losses += 1
        emit(gating_progress={"played": i + 1, "of": games,
                              "w": wins, "d": draws, "l": losses})

    score = (wins + 0.5 * draws) / max(games, 1)
    return score, wins, draws, losses


# ── training ─────────────────────────────────────────────────────────────────

def train_on_buffer(model, optimizer, buffer, batch_size, device, value_weight,
                    grad_clip, epochs_over_buffer=1):
    """Policy = cross-entropy against the MCTS visit distribution; value = MSE vs z."""
    model.train()
    n = len(buffer)
    if n == 0:
        return 0.0, 0.0, 0.0, 0

    total = policy_total = value_total = 0.0
    batches = 0
    data = list(buffer)

    for _ in range(epochs_over_buffer):
        random.shuffle(data)
        for start in range(0, n, batch_size):
            chunk = data[start:start + batch_size]
            bs = len(chunk)

            x = torch.from_numpy(np.stack([c[0] for c in chunk])).to(device)
            z = torch.tensor([c[3] for c in chunk], dtype=torch.float32, device=device)

            # Dense target built from the sparse visit distribution.
            pi = torch.zeros(bs, N_MOVES, device=device)
            for row, (_, idxs, probs, _) in enumerate(chunk):
                pi[row, torch.from_numpy(idxs).to(device)] = torch.from_numpy(probs).to(device)

            logits, value_pred = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(pi * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred.squeeze(-1), z)
            loss = policy_loss + value_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += loss.item()
            policy_total += policy_loss.item()
            value_total += value_loss.item()
            batches += 1

    model.eval()
    b = max(batches, 1)
    return total / b, policy_total / b, value_total / b, batches


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AlphaZero-style self-play training')
    parser.add_argument('--epochs', type=int, default=5, help='Self-play + train iterations')
    parser.add_argument('--games-per-epoch', type=int, default=50, help='Self-play games per epoch')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--simulations', type=int, default=64, help='MCTS simulations per move')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--value-weight', type=float, default=1.0, help='Weight of the value loss')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient norm clip; <=0 disables')
    parser.add_argument('--buffer-size', type=int, default=100_000, help='Replay buffer positions')
    parser.add_argument('--passes-per-epoch', type=int, default=1, help='Passes over the replay buffer per epoch')
    parser.add_argument('--temp-moves', type=int, default=20, help='Plies sampled at temperature 1 before going greedy')
    parser.add_argument('--max-plies', type=int, default=200, help='Ply cap per game')
    parser.add_argument('--model-path', default='danibot.pth', help='Champion model (only replaced after gating)')
    parser.add_argument('--gating-games', type=int, default=20, help='Games in the promotion match; 0 disables gating')
    parser.add_argument('--gating-threshold', type=float, default=0.55, help='Score the candidate must beat to be promoted')
    parser.add_argument('--gating-simulations', type=int, default=None, help='MCTS sims during gating (defaults to --simulations)')
    parser.add_argument('--threads', type=int, default=None,
                        help='torch CPU threads for search (default: min(4, cores); more hurts batch-1 inference)')
    parser.add_argument('--reuse-tree', action='store_true',
                        help='Retain the MCTS tree between moves: better search quality, ~20%% slower per move')
    parser.add_argument('--allow-dead-value-head', action='store_true',
                        help='Run even if the value head is constant (self-play will not improve anything)')

    # Accepted for backwards compatibility with app.py; unused by MCTS self-play.
    parser.add_argument('--policy-top-n', type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument('--policy-sample-k', type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument('--exploration-epsilon', type=float, default=0.20, help=argparse.SUPPRESS)
    parser.add_argument('--random-opening-plies', type=int, default=6, help='Random plies at the start of each game, for diversity')
    parser.add_argument('--draw-sample-ratio', type=float, default=0.15, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.gating_simulations is None:
        args.gating_simulations = args.simulations

    # Batch-1 convolutions do not scale with threads; 8 threads measured slower
    # than 4 on this network.
    threads = args.threads if args.threads else min(4, os.cpu_count() or 1)
    torch.set_num_threads(max(1, threads))

    start_perf = time.perf_counter()
    emit(status="training_start", epochs=args.epochs, strategy="mcts_selfplay",
         fen_type="from_scratch", simulations=args.simulations, threads=threads,
         reuse_tree=args.reuse_tree,
         games_per_epoch=args.games_per_epoch, gating_games=args.gating_games,
         gating_threshold=args.gating_threshold, buffer_size=args.buffer_size,
         random_opening_plies=args.random_opening_plies)

    device = torch.device('cpu')  # single-position MCTS inference is fastest on CPU

    champion = load_model(args.model_path, device)

    if value_head_is_dead(champion):
        msg = ("value head is constant (all-zero targets during supervised training). "
               "MCTS evaluates every leaf as a draw, so self-play cannot improve the "
               "model. Retrain with a Result column first: "
               "python scripts/pgn_to_fen_move_csv.py --skip-unfinished, then "
               "python train_csv_danibot.py --csv <that csv>.")
        if not args.allow_dead_value_head:
            emit(status="aborted", error=msg)
            sys.exit(2)
        emit(warning=msg + " Continuing anyway because --allow-dead-value-head was set.")

    # The candidate trains; the champion stays frozen as the gating opponent.
    candidate = ChessNet(N_MOVES).to(device)
    candidate.load_state_dict(champion.state_dict())

    optimizer = torch.optim.Adam(candidate.parameters(), lr=args.lr, weight_decay=1e-5)
    buffer = deque(maxlen=args.buffer_size)
    game = TrainingGame()
    all_stats = Counter()
    promotions = 0

    for epoch in range(args.epochs):
        # Self-play runs on a frozen snapshot of the candidate; the eager
        # `candidate` is what the optimizer updates below.
        mcts = make_mcts(freeze_for_inference(candidate), reuse_tree=args.reuse_tree)
        play_start = time.perf_counter()
        epoch_stats = Counter()
        new_samples = 0

        for _ in range(args.games_per_epoch):
            samples, result = play_selfplay_game(
                mcts, game,
                simulations=args.simulations,
                opening_plies=args.random_opening_plies,
                temp_moves=args.temp_moves,
                max_plies=args.max_plies,
            )
            buffer.extend(samples)
            new_samples += len(samples)
            epoch_stats[result] += 1

        all_stats += epoch_stats
        play_time = time.perf_counter() - play_start

        emit(epoch_selfplay=epoch + 1,
             games_played=sum(epoch_stats.values()),
             decisive_games=epoch_stats['1-0'] + epoch_stats['0-1'],
             draws=epoch_stats['1/2-1/2'],
             new_samples=new_samples,
             buffer=len(buffer),
             play_time_s=round(play_time, 2),
             games_per_sec=round(sum(epoch_stats.values()) / max(play_time, 1e-9), 3))

        if new_samples == 0:
            emit(warning=f"epoch {epoch + 1}: no samples collected")
            continue

        avg_loss, p_loss, v_loss, batches = train_on_buffer(
            candidate, optimizer, buffer, args.batch_size, device,
            args.value_weight, args.grad_clip, args.passes_per_epoch,
        )
        emit(epoch_train=epoch + 1, batches=batches, loss=round(avg_loss, 4),
             policy_loss=round(p_loss, 4), value_loss=round(v_loss, 4))

        top = epoch_stats.most_common(1)[0][0] if epoch_stats else '*'
        winner = "White" if top == '1-0' else "Black" if top == '0-1' else "Draw"
        print_status(epoch + 1, args.epochs, avg_loss, winner, all_stats)

        # ── Gating: only promote a candidate that actually plays better ──────
        if args.gating_games > 0:
            gate_start = time.perf_counter()
            score, w, d, l = gating_match(
                candidate, champion,
                games=args.gating_games,
                simulations=args.gating_simulations,
                opening_plies=args.random_opening_plies,
                max_plies=args.max_plies,
                reuse_tree=args.reuse_tree,
            )
            promoted = score >= args.gating_threshold
            emit(epoch_gating=epoch + 1, score=round(score, 4), wins=w, draws=d,
                 losses=l, threshold=args.gating_threshold, promoted=promoted,
                 gating_time_s=round(time.perf_counter() - gate_start, 2))

            if promoted:
                torch.save(candidate.state_dict(), args.model_path)
                champion.load_state_dict(candidate.state_dict())
                promotions += 1
                emit(status="model_saved", epoch=epoch + 1, reason="passed_gating",
                     score=round(score, 4))
            else:
                # Roll the candidate back so a bad iteration cannot compound.
                candidate.load_state_dict(champion.state_dict())
                emit(status="model_rejected", epoch=epoch + 1, score=round(score, 4))
        else:
            torch.save(candidate.state_dict(), args.model_path)
            champion.load_state_dict(candidate.state_dict())
            promotions += 1
            emit(status="model_saved", epoch=epoch + 1, reason="gating_disabled")

    duration = time.perf_counter() - start_perf
    total_games = sum(all_stats.values())
    emit(status="training_complete",
         total_epochs=args.epochs,
         duration_seconds=round(duration, 2),
         duration_minutes=round(duration / 60.0, 2),
         games_played_total=total_games,
         games_per_second=round(total_games / max(duration, 1e-9), 3),
         promotions=promotions,
         model_path=args.model_path)


if __name__ == '__main__':
    main()
