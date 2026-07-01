#!/usr/bin/env python3
"""Export mainline PGN positions to a CSV with columns: FEN, Move.

Each CSV row contains:
  - FEN: board state before the move
  - Move: associated move from the PGN mainline

By default the move is written in SAN, because that matches the PGN move text.
Use --move-format uci if you want machine-friendly moves instead.

Examples:
  python scripts/pgn_to_fen_move_csv.py
  python scripts/pgn_to_fen_move_csv.py --max-games 10
  python scripts/pgn_to_fen_move_csv.py --move-format uci \
      --output-csv lichess_2500plus_fen_uci.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import chess
import chess.pgn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PGN = PROJECT_ROOT / "lichess_2500plus.pgn"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "lichess_2500plus_fen_move.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export FEN + move pairs from a PGN file into CSV."
    )
    parser.add_argument(
        "--input-pgn",
        type=Path,
        default=DEFAULT_INPUT_PGN,
        help=f"Input PGN path (default: {DEFAULT_INPUT_PGN})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--move-format",
        choices=["san", "uci"],
        default="san",
        help="Move format written to CSV: san = PGN-style move, uci = engine-friendly move",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional limit on processed games",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N valid games (0 disables progress output)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding used for input/output (default: utf-8)",
    )
    return parser


def move_to_text(board: chess.Board, move: chess.Move, move_format: str) -> str:
    if move_format == "uci":
        return move.uci()
    return board.san(move)


def export_fen_moves(
    input_pgn: Path,
    output_csv: Path,
    move_format: str,
    max_games: Optional[int],
    progress_every: int,
    encoding: str,
) -> tuple[int, int, int]:
    scanned_games = 0
    written_games = 0
    written_rows = 0
    skipped_games = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_pgn.open("r", encoding=encoding, errors="replace") as pgn_file, output_csv.open(
        "w", newline="", encoding=encoding
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["FEN", "Move"])

        while True:
            if max_games is not None and scanned_games >= max_games:
                break

            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            scanned_games += 1
            board = game.board()
            game_rows: list[tuple[str, str]] = []

            try:
                for move in game.mainline_moves():
                    fen = board.fen()
                    move_text = move_to_text(board, move, move_format)
                    game_rows.append((fen, move_text))
                    board.push(move)
            except Exception as exc:
                skipped_games += 1
                print(
                    f"Skipping malformed game #{scanned_games}: {exc}",
                    file=sys.stderr,
                )
                continue

            writer.writerows(game_rows)
            written_games += 1
            written_rows += len(game_rows)

            if progress_every > 0 and written_games % progress_every == 0:
                csv_file.flush()
                print(
                    f"Processed {scanned_games} games | written {written_games} games | rows {written_rows}",
                    file=sys.stderr,
                )

    return written_games, written_rows, skipped_games


def main() -> None:
    args = build_parser().parse_args()

    if not args.input_pgn.exists():
        print(f"Input PGN not found: {args.input_pgn}", file=sys.stderr)
        raise SystemExit(1)

    written_games, written_rows, skipped_games = export_fen_moves(
        input_pgn=args.input_pgn,
        output_csv=args.output_csv,
        move_format=args.move_format,
        max_games=args.max_games,
        progress_every=args.progress_every,
        encoding=args.encoding,
    )

    print("Done.")
    print(f"Input PGN: {args.input_pgn}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Move format: {args.move_format}")
    print(f"Written games: {written_games}")
    print(f"Written rows: {written_rows}")
    print(f"Skipped malformed games: {skipped_games}")


if __name__ == "__main__":
    main()
