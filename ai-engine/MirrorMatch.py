import torch
import torch.nn as nn
import torch.nn.utils
import sys
import json
import argparse
from collections import Counter
import random
import chess
import time

from ChessAI import ChessAI
from TrainingGame import TrainingGame
from ArchiveAlpha import encode_board

# Try to import Cython batch encoding
try:
    from fastgame.board_encode import encode_board_batch as cy_encode_board_batch
    HAS_BATCH_ENCODE = True
except ImportError:
    cy_encode_board_batch = None
    HAS_BATCH_ENCODE = False

# Try to import Cython fast training loop
try:
    from fastgame.fast_training_loop import play_games_fast as cy_play_games_fast
    HAS_FAST_LOOP = True
except ImportError:
    cy_play_games_fast = None
    HAS_FAST_LOOP = False

# Load move mapping
with open('move_mapping.json', 'r', encoding='utf-8') as f:
    move_list = json.load(f)
move_to_idx = {m: i for i, m in enumerate(move_list)}


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


def play_games(
    ai_white,
    ai_black,
    game,
    fen_positions,
    num_games,
    policy_top_n=3,
    policy_sample_k=2,
    exploration_epsilon=0.20,
    random_opening_plies=6,
    draw_sample_ratio=0.15,
    cpu_model=None,
):
    """Play N games and collect (board_state, move) samples from self-play.

    Uses Cython fast loop + batch encoding when available for ~3-5x speedup.
    Falls back to pure Python otherwise.
    """
    # ── Fast path: Cython batch play + batch encode ──────────────────────
    if HAS_FAST_LOOP and HAS_BATCH_ENCODE:
        states_np, moves_list, stats = cy_play_games_fast(
            ai_white, ai_black, game, fen_positions,
            num_games=num_games,
            policy_top_n=policy_top_n,
            policy_sample_k=policy_sample_k,
            exploration_epsilon=exploration_epsilon,
            random_opening_plies=random_opening_plies,
            draw_sample_ratio=draw_sample_ratio,
            move_to_idx=move_to_idx,
            encode_batch_fn=cy_encode_board_batch,
            cpu_model=cpu_model,
        )
        if states_np is not None and len(moves_list) > 0:
            # Return full tensor directly — training loop detects this and skips torch.stack
            samples_states = torch.from_numpy(states_np)
            samples_moves = moves_list
        else:
            samples_states = []
            samples_moves = []
        return samples_states, samples_moves, stats

    # ── Slow fallback: per-move encoding ─────────────────────────────────
    return _play_games_python(
        ai_white, ai_black, game, fen_positions, num_games,
        policy_top_n, policy_sample_k, exploration_epsilon,
        random_opening_plies, draw_sample_ratio,
        cpu_model=cpu_model,
    )


def _play_games_python(
    ai_white, ai_black, game, fen_positions, num_games,
    policy_top_n, policy_sample_k, exploration_epsilon,
    random_opening_plies, draw_sample_ratio,
    cpu_model=None,
):
    """Pure Python fallback for play_games."""
    samples_states = []
    samples_moves = []
    stats = Counter()

    for g in range(num_games):
        if fen_positions:
            fen = random.choice(fen_positions)
            game.reset_from_fen(fen)
        else:
            game.reset()

        white_history = []
        black_history = []
        ply_count = 0

        ai_white.model.eval()
        pred_cache_white = {}
        pred_cache_black = {}
        move_cache_white = {}
        move_cache_black = {}

        while not game.is_game_over():
            ply_count += 1
            is_white_turn = game.board.turn == chess.WHITE
            current_ai = ai_white if is_white_turn else ai_black

            legal_moves = list(game.board.legal_moves)
            if not legal_moves:
                break

            if ply_count <= random_opening_plies or random.random() < exploration_epsilon:
                move = random.choice(legal_moves)
            else:
                pred_cache = pred_cache_white if is_white_turn else pred_cache_black
                move_cache = move_cache_white if is_white_turn else move_cache_black
                move = current_ai.get_fast_move_from_model(
                    game.board,
                    top_n=policy_top_n,
                    sample_top_k=policy_sample_k,
                    prediction_cache=pred_cache,
                    move_cache=move_cache,
                    cpu_model=cpu_model,
                )
                if move is None:
                    move = random.choice(legal_moves)

            move_idx = move_to_idx.get(move.uci())
            if move_idx is not None:
                board_tensor = encode_board(game.board)
                if is_white_turn:
                    white_history.append((board_tensor, move_idx))
                else:
                    black_history.append((board_tensor, move_idx))

            success, _ = game.make_move_fast(move)
            if not success:
                break

        result = game.get_result()
        stats[result] += 1

        if result == '1-0':
            samples_states.extend([s for s, _ in white_history])
            samples_moves.extend([m for _, m in white_history])
            if black_history:
                half = max(1, len(black_history) // 2)
                sampled = random.sample(black_history, half)
                samples_states.extend([s for s, _ in sampled])
                samples_moves.extend([m for _, m in sampled])
        elif result == '0-1':
            samples_states.extend([s for s, _ in black_history])
            samples_moves.extend([m for _, m in black_history])
            if white_history:
                half = max(1, len(white_history) // 2)
                sampled = random.sample(white_history, half)
                samples_states.extend([s for s, _ in sampled])
                samples_moves.extend([m for _, m in sampled])
        else:
            # Draws dominate in symmetric self-play; train on a smaller random subset.
            if white_history:
                keep_w = max(1, int(len(white_history) * draw_sample_ratio))
                sampled_w = random.sample(white_history, min(keep_w, len(white_history)))
                samples_states.extend([s for s, _ in sampled_w])
                samples_moves.extend([m for _, m in sampled_w])
            if black_history:
                keep_b = max(1, int(len(black_history) * draw_sample_ratio))
                sampled_b = random.sample(black_history, min(keep_b, len(black_history)))
                samples_states.extend([s for s, _ in sampled_b])
                samples_moves.extend([m for _, m in sampled_b])

    return samples_states, samples_moves, stats


def main():
    parser = argparse.ArgumentParser(description='Mirror Match Training for Chess AI')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--games-per-epoch', type=int, default=200, help='Games to play per epoch')
    parser.add_argument('--batch-size', type=int, default=512, help='Training batch size')
    parser.add_argument('--policy-top-n', type=int, default=3, help='Top-N legal policy moves considered')
    parser.add_argument('--policy-sample-k', type=int, default=2, help='Sample uniformly from top-K policy moves')
    parser.add_argument('--exploration-epsilon', type=float, default=0.20, help='Probability of random move during self-play')
    parser.add_argument('--random-opening-plies', type=int, default=6, help='Number of opening plies forced random for diversity')
    parser.add_argument('--draw-sample-ratio', type=float, default=0.15, help='Fraction of draw moves kept for training')
    args = parser.parse_args()

    training_start_perf = time.perf_counter()
    print(json.dumps({
        "status": "training_start",
        "epochs": args.epochs,
        "strategy": "model",
        "fen_type": "from_scratch",
        "cython_fast_loop": HAS_FAST_LOOP,
        "cython_batch_encode": HAS_BATCH_ENCODE,
        "policy_top_n": args.policy_top_n,
        "policy_sample_k": args.policy_sample_k,
        "exploration_epsilon": args.exploration_epsilon,
        "random_opening_plies": args.random_opening_plies,
        "draw_sample_ratio": args.draw_sample_ratio
    }))
    sys.stdout.flush()

    # --- Two AI instances with identical model weights ---
    ai_white = ChessAI(is_white=True, default_strategy='model', load_model_from_disk=True)
    ai_black = ChessAI(is_white=False, default_strategy='model', load_model_from_disk=False)
    ai_black.model.load_state_dict(ai_white.model.state_dict())
    ai_black.model.to(ai_black.device)
    ai_black.model.eval()
    device = ai_white.device

    # Create a CPU copy of the model for fast self-play inference.
    # MPS/CUDA device transfers for batch=1 are slower than CPU-only inference.
    # JIT tracing eliminates Python nn.Module overhead (~225K __getattr__ calls).
    import copy
    cpu_model = copy.deepcopy(ai_white.model).cpu().eval()
    for p in cpu_model.parameters():
        p.requires_grad_(False)
    try:
        dummy_input = torch.zeros(1, 18, 8, 8, dtype=torch.float32)
        cpu_model = torch.jit.trace(cpu_model, dummy_input)
        cpu_model = torch.jit.freeze(cpu_model)
        # Warm up the JIT
        for _ in range(3):
            cpu_model(dummy_input)
        print(json.dumps({"info": "CPU model JIT traced + frozen for fast self-play"}))
    except Exception as e:
        print(json.dumps({"warning": f"JIT trace failed, using eager mode: {e}"}))
    sys.stdout.flush()

    game = TrainingGame()
    all_stats = Counter()

    optimizer = torch.optim.Adam(ai_white.model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Always from scratch in mirror-match training.
    fen_positions = []

    for epoch in range(args.epochs):
        # Keep black instance synchronized with the current white model snapshot.
        ai_black.model.load_state_dict(ai_white.model.state_dict())
        ai_black.model.eval()

        # Sync CPU model for fast self-play inference (re-trace JIT)
        cpu_model_eager = copy.deepcopy(ai_white.model).cpu().eval()
        for p in cpu_model_eager.parameters():
            p.requires_grad_(False)
        try:
            dummy_input = torch.zeros(1, 18, 8, 8, dtype=torch.float32)
            cpu_model = torch.jit.trace(cpu_model_eager, dummy_input)
            cpu_model = torch.jit.freeze(cpu_model)
            cpu_model(dummy_input)  # warm up
        except Exception:
            cpu_model = cpu_model_eager

        # --- Phase 1: Play games, collect winning moves ---
        epoch_play_start = time.perf_counter()
        states, moves, stats = play_games(
            ai_white, ai_black, game, fen_positions,
            num_games=args.games_per_epoch,
            policy_top_n=args.policy_top_n,
            policy_sample_k=args.policy_sample_k,
            exploration_epsilon=args.exploration_epsilon,
            random_opening_plies=args.random_opening_plies,
            draw_sample_ratio=args.draw_sample_ratio,
            cpu_model=cpu_model,
        )
        epoch_play_time = time.perf_counter() - epoch_play_start
        all_stats += stats

        if len(states) == 0:
            print(json.dumps({"warning": f"Epoch {epoch+1}: no training samples collected"}))
            sys.stdout.flush()
            continue

        # Track decisive-game rate to catch collapse back into all-draw self-play.
        decisive = stats.get('1-0', 0) + stats.get('0-1', 0)
        games_played = sum(stats.values())
        print(json.dumps({
            "epoch": epoch + 1,
            "games_played": games_played,
            "decisive_games": decisive,
            "draws": stats.get('1/2-1/2', 0),
            "play_time_s": round(epoch_play_time, 2),
            "games_per_sec": round(games_played / max(epoch_play_time, 1e-9), 2),
            "samples_collected": len(states),
        }))
        sys.stdout.flush()

        # --- Phase 2: Batch train on collected samples ---
        if isinstance(states, torch.Tensor):
            states_tensor = states.contiguous()
        else:
            states_tensor = torch.stack(states).contiguous()
        moves_tensor = torch.tensor(moves, dtype=torch.long) if not isinstance(moves, torch.Tensor) else moves

        ai_white.model.train()
        total_loss = 0.0
        num_samples = states_tensor.size(0)
        n_batches = (num_samples + args.batch_size - 1) // args.batch_size
        indices = torch.randperm(num_samples)

        for batch_idx, start in enumerate(range(0, num_samples, args.batch_size), start=1):
            batch_indices = indices[start:start + args.batch_size]
            X = states_tensor.index_select(0, batch_indices).to(device, non_blocking=(device.type == "cuda"))
            y = moves_tensor.index_select(0, batch_indices).to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            # ChessNet returns (policy_logits, value); only the policy head is
            # supervised here, since self-play samples carry no value target.
            logits, _ = ai_white.model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ai_white.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0 or batch_idx == n_batches:
                print(
                    f"Batch {batch_idx}/{n_batches} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {total_loss / batch_idx:.4f}"
                )
                sys.stdout.flush()

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Report the epoch's dominant outcome. stats is a Counter, so its key
        # order is insertion order, not "the last game played".
        top_result = stats.most_common(1)[0][0] if stats else '*'
        winner = "White" if top_result == "1-0" else "Black" if top_result == "0-1" else "Draw"

        print_status(epoch + 1, args.epochs, avg_loss, winner, all_stats)

        if (epoch + 1) % 10 == 0:
            torch.save(ai_white.model.state_dict(), "danibot.pth")
            print(json.dumps({"status": "model_saved", "epoch": epoch + 1}))
            sys.stdout.flush()

    torch.save(ai_white.model.state_dict(), "danibot.pth")
    duration_seconds = time.perf_counter() - training_start_perf
    total_games_played = sum(all_stats.values())
    games_per_second = total_games_played / max(duration_seconds, 1e-9)
    print(json.dumps({
        "status": "training_complete",
        "total_epochs": args.epochs,
        "duration_seconds": round(duration_seconds, 2),
        "duration_minutes": round(duration_seconds / 60.0, 2),
        "games_played_total": int(total_games_played),
        "games_per_second": round(games_per_second, 2),
        "cython_fast_loop": HAS_FAST_LOOP,
        "cython_batch_encode": HAS_BATCH_ENCODE,
    }))
    sys.stdout.flush()

    # Cleanup
    if hasattr(ai_white, 'engine') and ai_white.engine:
        ai_white.engine.quit()
    if hasattr(ai_black, 'engine') and ai_black.engine:
        ai_black.engine.quit()


if __name__ == "__main__":
    main()
