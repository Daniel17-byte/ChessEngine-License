#!/usr/bin/env python3
"""
============================================================================
  Leela Chess Zero (lc0) Training Data Generator
============================================================================

Generează fișiere de training în formatul V6 folosit de lc0.

Formatul fișierelor de training lc0:
  - Fiecare fișier este un .gz (gzip) care conține înregistrări binare
  - Fiecare înregistrare (record) = un struct V6TrainingData (8356 bytes)
  - Datele conțin: input planes, policy, value, best moves, etc.

Surse de date:
  1. Self-play cu Stockfish (cel mai comun)
  2. Conversie din PGN-uri existente
  3. Jocuri random (pentru testare pipeline)

Usage:
  python generate_lc0_data.py --mode random --games 50
  python generate_lc0_data.py --mode pgn --pgn-file lichess_2500plus.pgn
  python generate_lc0_data.py --mode pgn --pgn-file lichess_2500plus.pgn --stockfish /opt/homebrew/bin/stockfish
  python generate_lc0_data.py --mode stockfish --games 100 --stockfish /opt/homebrew/bin/stockfish
  python generate_lc0_data.py --mode inspect --inspect-file lc0_training_data/training.000000.gz

Requires: python-chess, numpy  (already in requirements.txt)
Optional: stockfish binary for evaluation
============================================================================
"""

import struct
import gzip
import os
import sys
import argparse
import time
import random
from pathlib import Path

import chess
import chess.pgn
import numpy as np

# ── Reuse existing project encoding & move mapping ──────────────────────────
from ArchiveAlpha import encode_board_array

# ============================================================================
#   CONSTANTE FORMAT LC0 V6
# ============================================================================

# lc0 standard: 112 input planes (13 per position × 8 history + 12 constant)
LC0_INPUT_PLANES = 112

# lc0 policy: 1858 move indices
LC0_POLICY_SIZE = 1858

# V6 record: version(4) + planes(896) + policy(7432) + result(4) + extras
V6_RECORD_SIZE = 8356
V6_VERSION = 6

# Poziții per chunk .gz
DEFAULT_CHUNK_SIZE = 200


# ============================================================================
#   LC0 MOVE ENCODING  (Policy Index)
# ============================================================================
#
# lc0 codifică mutările astfel:
#   - Queen-like (include pion drepți, diagonale, tură, nebun, damă):
#       8 direcții × 7 distanțe = 56 tipuri, enumerate per from_square
#   - Knight moves: 8 tipuri per from_square
#   - Under-promotions (knight/bishop/rook × 3 direcții capture)
#   Total valid: exact 1858 indecși.
# ============================================================================

def _init_lc0_move_tables():
    """Construiește tabelul chess.Move -> policy index (1858 entries)."""
    queen_dirs = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
    ]
    knight_deltas = [
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2),
    ]
    underpromo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    underpromo_dirs = [-1, 0, 1]  # left capture, forward, right capture

    move_to_idx = {}
    idx = 0

    # Queen-like (include queen promotions, mapped to same index as non-promo)
    for from_sq in range(64):
        from_row, from_col = from_sq // 8, from_sq % 8
        for dx, dy in queen_dirs:
            for dist in range(1, 8):
                to_row = from_row + dy * dist
                to_col = from_col + dx * dist
                if 0 <= to_row < 8 and 0 <= to_col < 8:
                    to_sq = to_row * 8 + to_col
                    move = chess.Move(from_sq, to_sq)
                    if move not in move_to_idx:
                        move_to_idx[move] = idx
                    # Queen promotion shares the same index
                    if (from_row == 6 and to_row == 7) or (from_row == 1 and to_row == 0):
                        promo = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                        if promo not in move_to_idx:
                            move_to_idx[promo] = idx
                    idx += 1

    # Knight moves
    for from_sq in range(64):
        from_row, from_col = from_sq // 8, from_sq % 8
        for dx, dy in knight_deltas:
            to_row, to_col = from_row + dy, from_col + dx
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                to_sq = to_row * 8 + to_col
                move = chess.Move(from_sq, to_sq)
                if move not in move_to_idx:
                    move_to_idx[move] = idx
                idx += 1

    # Under-promotions (knight, bishop, rook × 3 column offsets)
    for from_sq in range(64):
        from_row, from_col = from_sq // 8, from_sq % 8
        for piece in underpromo_pieces:
            for d_col in underpromo_dirs:
                if from_row == 6:           # white under-promotion
                    to_row, to_col = 7, from_col + d_col
                    if 0 <= to_col < 8:
                        move_to_idx.setdefault(
                            chess.Move(from_sq, to_row * 8 + to_col, promotion=piece), idx)
                        idx += 1
                if from_row == 1:           # black under-promotion
                    to_row, to_col = 0, from_col + d_col
                    if 0 <= to_col < 8:
                        move_to_idx.setdefault(
                            chess.Move(from_sq, to_row * 8 + to_col, promotion=piece), idx)
                        idx += 1

    return move_to_idx, idx


_LC0_MOVE2IDX, _LC0_TOTAL_INDICES = _init_lc0_move_tables()


def move_to_lc0_policy_index(move: chess.Move) -> int:
    """chess.Move -> lc0 policy index (0..1857).  Returns 0 for unknown."""
    return _LC0_MOVE2IDX.get(move, 0)


# ============================================================================
#   LC0 INPUT PLANES  (112 planes de 8×8)
# ============================================================================

def encode_position_lc0(board: chess.Board, history: list = None) -> np.ndarray:
    """
    Encode position in lc0 V6 format: 112 planes of 8×8.

    Layout:
      planes 0..103  = 13 planes × 8 time-steps (current + 7 history)
                        6 our pieces, 6 their pieces, 1 repetition
      planes 104..111 = constant planes (castling, side, rule50, move#, pad)
    """
    if history is None:
        history = []
    planes = np.zeros((LC0_INPUT_PLANES, 8, 8), dtype=np.float32)

    boards = [board] + history[:7]
    for t, b in enumerate(boards):
        off = t * 13
        our, their = b.turn, not b.turn
        for pt in range(1, 7):                       # PAWN..KING
            for sq in b.pieces(pt, our):
                r, c = sq // 8, sq % 8
                if b.turn == chess.BLACK:
                    r = 7 - r
                planes[off + pt - 1, r, c] = 1.0
            for sq in b.pieces(pt, their):
                r, c = sq // 8, sq % 8
                if b.turn == chess.BLACK:
                    r = 7 - r
                planes[off + 6 + pt - 1, r, c] = 1.0
        if b.is_repetition(1):
            planes[off + 12, :, :] = 1.0

    # Constant planes (offset 104)
    co = 104
    if board.turn == chess.WHITE:
        cas = [(chess.WHITE, True), (chess.WHITE, False),
               (chess.BLACK, True), (chess.BLACK, False)]
    else:
        cas = [(chess.BLACK, True), (chess.BLACK, False),
               (chess.WHITE, True), (chess.WHITE, False)]
    for i, (color, kingside) in enumerate(cas):
        fn = board.has_kingside_castling_rights if kingside else board.has_queenside_castling_rights
        if fn(color):
            planes[co + i, :, :] = 1.0
    planes[co + 4, :, :] = 1.0                       # side-to-move (always 1, board is flipped)
    planes[co + 5, :, :] = board.halfmove_clock / 100.0
    return planes


def _planes_to_bits(planes: np.ndarray) -> bytes:
    """Float planes [112,8,8] -> 896 bytes of packed bits (little-endian u64)."""
    buf = bytearray(LC0_INPUT_PLANES * 8)
    for p in range(LC0_INPUT_PLANES):
        bits = 0
        plane = planes[p]
        for r in range(8):
            for c in range(8):
                if plane[r, c] > 0.5:
                    bits |= 1 << (r * 8 + c)
        struct.pack_into('<Q', buf, p * 8, bits)
    return bytes(buf)


# ============================================================================
#   V6 RECORD BUILDER
# ============================================================================

def create_v6_record(
    board: chess.Board,
    policy: dict,          # {chess.Move: float probability}
    result: float,         # +1 win, 0 draw, -1 loss (side-to-move perspective)
    history: list = None,
    root_q: float = 0.0,
    best_q: float = 0.0,
    root_d: float = 0.0,
    best_d: float = 0.0,
    root_m: float = 0.0,
    best_m: float = 0.0,
    plies_left: float = 0.0,
) -> bytes:
    """Build one V6 training record (8356 bytes)."""
    buf = bytearray()

    # 1. version  (u32)
    buf += struct.pack('<I', V6_VERSION)

    # 2. input planes  (112 × 8 = 896 bytes)
    buf += _planes_to_bits(encode_position_lc0(board, history))

    # 3. policy  (1858 × f32 = 7432 bytes)
    pol = np.zeros(LC0_POLICY_SIZE, dtype=np.float32)
    total = sum(policy.values())
    if total > 0:
        for mv, prob in policy.items():
            idx = move_to_lc0_policy_index(mv)
            if idx < LC0_POLICY_SIZE:
                pol[idx] = prob / total
    buf += pol.tobytes()

    # 4. result  (i32)
    buf += struct.pack('<i', int(result))

    # 5-11. float32 extras
    for v in (root_q, best_q, root_d, best_d, root_m, best_m, plies_left):
        buf += struct.pack('<f', v)

    # pad to exact size
    if len(buf) < V6_RECORD_SIZE:
        buf += b'\x00' * (V6_RECORD_SIZE - len(buf))
    return bytes(buf[:V6_RECORD_SIZE])


# ============================================================================
#   CHUNK WRITER
# ============================================================================

def write_chunk(records: list, output_dir: str, chunk_id: int) -> str:
    """Write list of V6 records into training.XXXXXX.gz and return path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"training.{chunk_id:06d}.gz")
    with gzip.open(path, 'wb') as f:
        for rec in records:
            f.write(rec)
    return path


# ============================================================================
#   GENERATORS
# ============================================================================

def _flush(records, output_dir, chunk_id):
    """Helper: flush records to chunk, return ([], next_chunk_id)."""
    if records:
        path = write_chunk(records, output_dir, chunk_id)
        print(f"  💾 Chunk {chunk_id}: {path}  ({len(records)} positions)")
        return [], chunk_id + 1
    return records, chunk_id


def _result_from_str(s: str):
    return {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(s)


# ── Stockfish helper ─────────────────────────────────────────────────────────

def _stockfish_policy(engine, board, time_limit=0.05, multipv=5):
    """Get soft policy from Stockfish multi-PV analysis."""
    import chess.engine as ce
    try:
        infos = engine.analyse(
            board, ce.Limit(time=time_limit),
            multipv=min(multipv, len(list(board.legal_moves))),
        )
    except Exception:
        legal = list(board.legal_moves)
        return {m: 1.0 / len(legal) for m in legal} if legal else {}

    scores = []
    for info in (infos if isinstance(infos, list) else [infos]):
        mv = info.get("pv", [None])[0]
        sc = info.get("score")
        if mv and sc:
            cp = sc.white().score(mate_score=10000)
            if cp is not None:
                scores.append((mv, cp))
    if not scores:
        legal = list(board.legal_moves)
        return {m: 1.0 / len(legal) for m in legal} if legal else {}
    mx = max(s for _, s in scores)
    exp = [(m, np.exp((cp - mx) / 100.0)) for m, cp in scores]
    tot = sum(e for _, e in exp)
    return {m: e / tot for m, e in exp}


# ── From PGN ─────────────────────────────────────────────────────────────────

def generate_from_pgn(pgn_path: str, output_dir: str, max_games: int = None,
                      stockfish_path: str = None, chunk_size: int = DEFAULT_CHUNK_SIZE):
    import chess.engine as ce
    print(f"📖 Reading PGN: {pgn_path}")
    engine = None
    if stockfish_path and stockfish_path != 'none' and os.path.isfile(stockfish_path):
        try:
            engine = ce.SimpleEngine.popen_uci(stockfish_path)
            print(f"♟️  Stockfish enabled: {stockfish_path}")
        except Exception as e:
            print(f"⚠️  Stockfish failed ({e}), using one-hot policy")
            engine = None
    else:
        print("📝 No Stockfish — using one-hot policy from played moves")

    records, chunk_id, games, positions = [], 0, 0, 0
    try:
        with open(pgn_path) as pf:
            while True:
                game = chess.pgn.read_game(pf)
                if game is None or (max_games and games >= max_games):
                    break
                wr = _result_from_str(game.headers.get("Result", "*"))
                if wr is None:
                    continue

                board = game.board()
                history = []
                moves = list(game.mainline_moves())
                for mi, mv in enumerate(moves):
                    res = wr if board.turn == chess.WHITE else -wr
                    pol = _stockfish_policy(engine, board) if engine else {mv: 1.0}
                    records.append(create_v6_record(
                        board=board, policy=pol, result=res,
                        history=history.copy(),
                        root_q=res * 0.5, best_q=res * 0.5,
                        plies_left=float(len(moves) - mi),
                    ))
                    positions += 1
                    history.insert(0, board.copy())
                    history = history[:7]
                    board.push(mv)
                    if len(records) >= chunk_size:
                        records, chunk_id = _flush(records, output_dir, chunk_id)

                games += 1
                if games % 100 == 0:
                    print(f"  🎮 Games: {games}, Positions: {positions}")
        records, chunk_id = _flush(records, output_dir, chunk_id)
    finally:
        if engine:
            engine.quit()

    print(f"\n✅ PGN done — {games} games, {positions} positions, {chunk_id} chunks → {output_dir}/")


# ── Stockfish self-play ──────────────────────────────────────────────────────

def generate_selfplay(output_dir: str, num_games: int = 100,
                      stockfish_path: str = None, time_limit: float = 0.1,
                      chunk_size: int = DEFAULT_CHUNK_SIZE):
    import chess.engine as ce
    if not stockfish_path or not os.path.exists(stockfish_path):
        print("❌ Stockfish binary required for self-play!")
        print(f"   Path given: {stockfish_path}")
        print("   Install:  brew install stockfish")
        sys.exit(1)

    engine = ce.SimpleEngine.popen_uci(stockfish_path)
    print(f"♟️  Stockfish self-play: {stockfish_path}")
    print(f"🎮 Generating {num_games} games …")

    records, chunk_id, positions = [], 0, 0
    try:
        for gi in range(num_games):
            board, history, game_buf = chess.Board(), [], []
            mc = 0
            while not board.is_game_over() and mc < 300:
                pol = _stockfish_policy(engine, board, time_limit)
                if not pol:
                    break
                # Exploration: 10 % random in first 30 plies
                if mc < 30 and random.random() < 0.1:
                    chosen = random.choice(list(board.legal_moves))
                else:
                    chosen = max(pol, key=pol.get)
                game_buf.append(dict(board=board.copy(), policy=pol, history=history.copy()))
                history.insert(0, board.copy())
                history = history[:7]
                board.push(chosen)
                mc += 1

            wr = _result_from_str(board.result()) or 0.0
            for i, gr in enumerate(game_buf):
                res = wr if gr['board'].turn == chess.WHITE else -wr
                records.append(create_v6_record(
                    board=gr['board'], policy=gr['policy'], result=res,
                    history=gr['history'],
                    root_q=res * 0.5, best_q=res * 0.5,
                    plies_left=float(len(game_buf) - i),
                ))
                positions += 1
                if len(records) >= chunk_size:
                    records, chunk_id = _flush(records, output_dir, chunk_id)

            if (gi + 1) % 10 == 0:
                print(f"  🎮 Games: {gi + 1}/{num_games}, Positions: {positions}")
        records, chunk_id = _flush(records, output_dir, chunk_id)
    finally:
        engine.quit()

    print(f"\n✅ Self-play done — {num_games} games, {positions} positions, {chunk_id} chunks")


# ── Random games (pipeline test) ─────────────────────────────────────────────

def generate_random(output_dir: str, num_games: int = 100,
                    chunk_size: int = DEFAULT_CHUNK_SIZE):
    print(f"🎲 Generating {num_games} random games (pipeline test) …")
    records, chunk_id, positions = [], 0, 0

    for gi in range(num_games):
        board, history, game_buf = chess.Board(), [], []
        while not board.is_game_over() and board.fullmove_number < 150:
            legal = list(board.legal_moves)
            if not legal:
                break
            pol = {m: 1.0 / len(legal) for m in legal}
            chosen = random.choice(legal)
            game_buf.append(dict(board=board.copy(), policy=pol, history=history.copy()))
            history.insert(0, board.copy())
            history = history[:7]
            board.push(chosen)

        wr = _result_from_str(board.result()) or 0.0
        for i, gr in enumerate(game_buf):
            res = wr if gr['board'].turn == chess.WHITE else -wr
            records.append(create_v6_record(
                board=gr['board'], policy=gr['policy'], result=res,
                history=gr['history'], plies_left=float(len(game_buf) - i),
            ))
            positions += 1
            if len(records) >= chunk_size:
                records, chunk_id = _flush(records, output_dir, chunk_id)

        if (gi + 1) % 50 == 0:
            print(f"  🎮 Games: {gi + 1}/{num_games}, Positions: {positions}")

    records, chunk_id = _flush(records, output_dir, chunk_id)
    print(f"\n✅ Random done — {num_games} games, {positions} positions, {chunk_id} chunks")


# ============================================================================
#   INSPECT / VERIFY
# ============================================================================

def inspect_file(filepath: str, max_records: int = 5):
    print(f"\n🔍 Inspecting: {filepath}")
    print(f"   File size: {os.path.getsize(filepath):,} bytes")

    with gzip.open(filepath, 'rb') as f:
        data = f.read()
    n = len(data) // V6_RECORD_SIZE
    rem = len(data) % V6_RECORD_SIZE
    print(f"   Decompressed: {len(data):,} bytes")
    print(f"   Records: {n}")
    print(f"   Remainder: {rem} bytes {'✅' if rem == 0 else '❌ (corrupt!)'}")

    for i in range(min(max_records, n)):
        off = i * V6_RECORD_SIZE
        rec = data[off:off + V6_RECORD_SIZE]
        ver = struct.unpack('<I', rec[:4])[0]
        pol_off = 4 + 112 * 8
        pol = np.frombuffer(rec[pol_off:pol_off + 1858 * 4], dtype=np.float32)
        res_off = pol_off + 1858 * 4
        res = struct.unpack('<i', rec[res_off:res_off + 4])[0]
        top = np.argsort(pol)[-5:][::-1]
        print(f"\n   Record {i}:")
        print(f"     Version : {ver}")
        print(f"     Result  : {res:+d} ({'win' if res == 1 else 'draw' if res == 0 else 'loss'})")
        print(f"     Top 5 policy indices : {top.tolist()}")
        print(f"     Top 5 policy probs   : {[f'{pol[j]:.4f}' for j in top]}")
        print(f"     Policy sum           : {pol.sum():.4f}")


# ============================================================================
#   MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Leela Chess Zero (lc0) Training Data Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick pipeline test with random games:
  python generate_lc0_data.py --mode random --games 50

  # Convert PGN (one-hot policy from played move):
  python generate_lc0_data.py --mode pgn --pgn-file lichess_2500plus.pgn --games 500

  # Convert PGN with Stockfish soft-policy:
  python generate_lc0_data.py --mode pgn --pgn-file lichess_2500plus.pgn \\
      --stockfish /opt/homebrew/bin/stockfish

  # Stockfish self-play:
  python generate_lc0_data.py --mode stockfish --games 100 \\
      --stockfish /opt/homebrew/bin/stockfish --time-limit 0.2

  # Inspect generated file:
  python generate_lc0_data.py --mode inspect \\
      --inspect-file lc0_training_data/training.000000.gz
        """,
    )
    parser.add_argument('--mode', choices=['pgn', 'stockfish', 'random', 'inspect'],
                        default='random')
    parser.add_argument('--output', default='lc0_training_data',
                        help='Output directory (default: lc0_training_data/)')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--pgn-file', type=str, help='PGN file to convert')
    parser.add_argument('--stockfish', type=str, default='/opt/homebrew/bin/stockfish')
    parser.add_argument('--time-limit', type=float, default=0.1,
                        help='Stockfish time per move (seconds)')
    parser.add_argument('--inspect-file', type=str, help='Training file to inspect')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Positions per .gz chunk (default: {DEFAULT_CHUNK_SIZE})')
    args = parser.parse_args()

    t0 = time.time()

    if args.mode == 'random':
        generate_random(args.output, args.games, args.chunk_size)
    elif args.mode == 'pgn':
        if not args.pgn_file:
            print("❌ --pgn-file is required for pgn mode"); sys.exit(1)
        generate_from_pgn(args.pgn_file, args.output, args.games,
                          args.stockfish, args.chunk_size)
    elif args.mode == 'stockfish':
        generate_selfplay(args.output, args.games, args.stockfish,
                          args.time_limit, args.chunk_size)
    elif args.mode == 'inspect':
        if not args.inspect_file:
            print("❌ --inspect-file is required for inspect mode"); sys.exit(1)
        inspect_file(args.inspect_file)

    print(f"\n⏱️  Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()

