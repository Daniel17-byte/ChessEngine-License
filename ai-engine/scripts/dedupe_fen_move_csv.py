#!/usr/bin/env python3
"""Deduplicate a FEN/Move CSV while preserving first-seen order.

Default behavior keeps unique (FEN, Move) pairs.
If you really want literal unique move strings only, use --dedupe-by move.

Examples:
  python scripts/dedupe_fen_move_csv.py
  python scripts/dedupe_fen_move_csv.py --in-place --backup
  python scripts/dedupe_fen_move_csv.py --dedupe-by move --in-place --backup
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def detect_default_input_csv() -> Path:
    candidates = [
        PROJECT_ROOT / "lichess_2200plus_fen_move.csv",
        PROJECT_ROOT / "lichess_2500plus_fen_move.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_INPUT_CSV = detect_default_input_csv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep only unique rows from a FEN/Move CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV path (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, writes <input>_unique.csv unless --in-place is used.",
    )
    parser.add_argument(
        "--dedupe-by",
        choices=["fen_move", "fen", "move"],
        default="fen_move",
        help="How uniqueness is determined: fen_move (default), fen, or move.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input file safely via a temporary file.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a .bak copy before replacing the input file when using --in-place.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print progress every N scanned rows (0 disables progress output).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding used for input/output (default: utf-8)",
    )
    return parser


def default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_unique{input_csv.suffix}")


def dedupe_key(row: dict[str, str], mode: str) -> Tuple[str, ...]:
    fen = row.get("FEN", "")
    move = row.get("Move", "")
    if mode == "fen":
        return (fen,)
    if mode == "move":
        return (move,)
    return fen, move


def validate_columns(fieldnames: Optional[Iterable[str]]) -> None:
    required = {"FEN", "Move"}
    actual = set(fieldnames or [])
    if not required.issubset(actual):
        missing = ", ".join(sorted(required - actual))
        print(f"Input CSV is missing required columns: {missing}", file=sys.stderr)
        raise SystemExit(1)


def dedupe_csv(
    input_csv: Path,
    output_csv: Path,
    mode: str,
    progress_every: int,
    encoding: str,
) -> tuple[int, int]:
    scanned_rows = 0
    kept_rows = 0
    seen: set[Tuple[str, ...]] = set()

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", newline="", encoding=encoding, errors="replace") as src, output_csv.open(
        "w", newline="", encoding=encoding
    ) as dst:
        reader = csv.DictReader(src)
        validate_columns(reader.fieldnames)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            scanned_rows += 1
            key = dedupe_key(row, mode)
            if key in seen:
                if progress_every > 0 and scanned_rows % progress_every == 0:
                    dst.flush()
                    print(
                        f"Scanned {scanned_rows} rows | kept {kept_rows} | removed {scanned_rows - kept_rows}",
                        file=sys.stderr,
                    )
                continue

            seen.add(key)
            writer.writerow(row)
            kept_rows += 1

            if progress_every > 0 and scanned_rows % progress_every == 0:
                dst.flush()
                print(
                    f"Scanned {scanned_rows} rows | kept {kept_rows} | removed {scanned_rows - kept_rows}",
                    file=sys.stderr,
                )

    return scanned_rows, kept_rows


def main() -> None:
    args = build_parser().parse_args()

    if not args.input_csv.exists():
        print(f"Input CSV not found: {args.input_csv}", file=sys.stderr)
        raise SystemExit(1)

    if args.in_place:
        output_csv = args.input_csv.with_suffix(args.input_csv.suffix + ".tmp")
    else:
        output_csv = args.output_csv or default_output_path(args.input_csv)

    scanned_rows, kept_rows = dedupe_csv(
        input_csv=args.input_csv,
        output_csv=output_csv,
        mode=args.dedupe_by,
        progress_every=args.progress_every,
        encoding=args.encoding,
    )

    final_output = output_csv
    if args.in_place:
        if args.backup:
            backup_path = args.input_csv.with_suffix(args.input_csv.suffix + ".bak")
            shutil.copy2(args.input_csv, backup_path)
            print(f"Backup created: {backup_path}")
        output_csv.replace(args.input_csv)
        final_output = args.input_csv

    print("Done.")
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {final_output}")
    print(f"Dedupe by: {args.dedupe_by}")
    print(f"Scanned rows: {scanned_rows}")
    print(f"Kept rows: {kept_rows}")
    print(f"Removed duplicates: {scanned_rows - kept_rows}")


if __name__ == "__main__":
    main()
