"""
Danibot ELO Estimator

Calculates an approximate ELO rating for the Danibot model by playing
matches against Stockfish at various known skill levels.

Stockfish Skill Level → approximate ELO mapping:
  Level  0 → ~800    Level  5 → ~1200   Level 10 → ~1700
  Level  1 → ~900    Level  6 → ~1300   Level 15 → ~2200
  Level  2 → ~1000   Level  7 → ~1400   Level 20 → ~3200
  Level  3 → ~1050   Level  8 → ~1500
  Level  4 → ~1100   Level  9 → ~1600

Usage:
    python estimate_elo.py [--games 20] [--max-moves 150] [--verbose]
"""

import sys
import os
import json
import argparse
import chess
import chess.engine
from ChessAI import ChessAI

# Stockfish Skill Level → approximate ELO
SKILL_TO_ELO = {
    0: 800,
    1: 900,
    2: 1000,
    3: 1050,
    4: 1100,
    5: 1200,
    6: 1300,
    7: 1400,
    8: 1500,
    9: 1600,
    10: 1700,
    11: 1800,
    12: 1900,
    13: 2000,
    14: 2100,
    15: 2200,
    16: 2400,
    17: 2600,
    18: 2800,
    19: 3000,
    20: 3200,
}

ENGINE_PATH = "/opt/homebrew/bin/stockfish"


def play_match(danibot, engine, skill_level, num_games, max_moves, verbose=False):
    """
    Play num_games between Danibot and Stockfish at a given skill level.
    Danibot alternates white/black each game.
    Returns (wins, draws, losses) from Danibot's perspective.
    """
    try:
        engine.configure({"Skill Level": skill_level})
    except Exception:
        pass  # Some builds don't support Skill Level

    wins = 0
    draws = 0
    losses = 0

    for g in range(num_games):
        board = chess.Board()
        danibot_is_white = (g % 2 == 0)
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            move_count += 1

            if (board.turn == chess.WHITE) == danibot_is_white:
                # Danibot's turn
                move = danibot.get_best_move_from_model(board)
                if move is None or move not in board.legal_moves:
                    move = list(board.legal_moves)[0] if board.legal_moves else None
            else:
                # Stockfish's turn
                result = engine.play(board, chess.engine.Limit(time=0.01))
                move = result.move

            if move is None:
                break
            board.push(move)

        # Determine result
        result = board.result()
        if result == "1-0":
            if danibot_is_white:
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            if danibot_is_white:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1

        if verbose:
            outcome = "WIN" if ((result == "1-0" and danibot_is_white) or
                                (result == "0-1" and not danibot_is_white)) else \
                      "LOSS" if ((result == "0-1" and danibot_is_white) or
                                 (result == "1-0" and not danibot_is_white)) else "DRAW"
            color = "White" if danibot_is_white else "Black"
            print(f"  Game {g+1}: Danibot ({color}) vs Stockfish L{skill_level} → {outcome} ({move_count} moves)")
            sys.stdout.flush()

    return wins, draws, losses


def compute_win_rate(wins, draws, losses):
    """Win rate using: wins count 1.0, draws count 0.5"""
    total = wins + draws + losses
    if total == 0:
        return 0.5
    return (wins + 0.5 * draws) / total


def estimate_elo_from_results(results):
    """
    Estimate Danibot ELO based on win rates against various Stockfish levels.
    Uses the level where Danibot wins ~50% as the anchor point.
    """
    # Find the skill level where win rate is closest to 50%
    best_level = None
    best_diff = float('inf')

    for level, (wins, draws, losses) in sorted(results.items()):
        wr = compute_win_rate(wins, draws, losses)
        diff = abs(wr - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_level = level

    if best_level is None:
        return 800  # fallback

    # Interpolate: if win rate > 50% at this level, ELO is between this and next level
    wr = compute_win_rate(*results[best_level])
    base_elo = SKILL_TO_ELO.get(best_level, 800)

    if wr > 0.5 and best_level + 1 in SKILL_TO_ELO:
        next_elo = SKILL_TO_ELO[best_level + 1]
        # Linear interpolation
        estimated = base_elo + (next_elo - base_elo) * (wr - 0.5) * 2
    elif wr < 0.5 and best_level - 1 in SKILL_TO_ELO:
        prev_elo = SKILL_TO_ELO[best_level - 1]
        estimated = base_elo - (base_elo - prev_elo) * (0.5 - wr) * 2
    else:
        estimated = base_elo

    return round(estimated)


def main():
    parser = argparse.ArgumentParser(description='Estimate Danibot ELO rating')
    parser.add_argument('--games', type=int, default=10, help='Games per skill level')
    parser.add_argument('--max-moves', type=int, default=200, help='Max moves per game')
    parser.add_argument('--min-level', type=int, default=0, help='Minimum Stockfish skill level to test')
    parser.add_argument('--max-level', type=int, default=10, help='Maximum Stockfish skill level to test')
    parser.add_argument('--verbose', action='store_true', help='Print individual game results')
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 DANIBOT ELO ESTIMATOR")
    print("=" * 60)
    print(f"Games per level: {args.games}")
    print(f"Max moves/game: {args.max_moves}")
    print(f"Skill levels: {args.min_level} → {args.max_level}")
    print()
    sys.stdout.flush()

    # Initialize Danibot (model strategy, no random)
    danibot = ChessAI(is_white=True, default_strategy='model')
    danibot.epsilon = 0.0  # No random moves during evaluation

    if not os.path.exists(ENGINE_PATH):
        print(f"❌ Stockfish not found at {ENGINE_PATH}")
        sys.exit(1)

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    results = {}
    stop_testing = False

    for level in range(args.min_level, args.max_level + 1):
        elo = SKILL_TO_ELO.get(level, "?")
        print(f"── Stockfish Level {level} (ELO ~{elo}) ──")
        sys.stdout.flush()

        wins, draws, losses = play_match(
            danibot, engine, level,
            num_games=args.games,
            max_moves=args.max_moves,
            verbose=args.verbose
        )

        wr = compute_win_rate(wins, draws, losses)
        results[level] = (wins, draws, losses)

        print(f"   Results: {wins}W / {draws}D / {losses}L  (Win rate: {wr*100:.1f}%)")
        print()
        sys.stdout.flush()

        # Stop early if getting crushed (win rate < 15% at this level)
        if wr < 0.15 and level >= args.min_level + 2:
            print(f"   ⏹ Stopping early — Danibot is outmatched at level {level}")
            stop_testing = True
            break

    engine.quit()

    # Cleanup Danibot's Stockfish engine if it has one
    if hasattr(danibot, 'engine') and danibot.engine:
        danibot.engine.quit()

    # Calculate ELO
    estimated_elo = estimate_elo_from_results(results)

    print("=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Level':>6} {'ELO':>6} {'W':>4} {'D':>4} {'L':>4} {'WinRate':>8}")
    print("-" * 40)

    for level, (w, d, l) in sorted(results.items()):
        elo = SKILL_TO_ELO.get(level, "?")
        wr = compute_win_rate(w, d, l)
        bar = "█" * int(wr * 20) + "░" * (20 - int(wr * 20))
        print(f"  L{level:>3}  {elo:>5}  {w:>3}  {d:>3}  {l:>3}  {wr*100:>6.1f}%  {bar}")

    print()
    print(f"🧠 Estimated Danibot ELO: {estimated_elo}")
    print("=" * 60)

    # Save results to file
    output = {
        "estimated_elo": estimated_elo,
        "games_per_level": args.games,
        "max_moves": args.max_moves,
        "results": {
            str(level): {"wins": w, "draws": d, "losses": l, "win_rate": round(compute_win_rate(w, d, l), 3)}
            for level, (w, d, l) in results.items()
        }
    }

    with open("elo_estimate.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"💾 Results saved to elo_estimate.json")
    sys.stdout.flush()


if __name__ == '__main__':
    main()

