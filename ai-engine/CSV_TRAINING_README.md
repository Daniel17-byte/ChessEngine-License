# CSV DaniBot Training

Train `danibot.pth` directly from a `FEN,Move` CSV.

## Input format

Expected columns:

```csv
FEN,Move,Result
```

`Result` is optional but strongly recommended: it is the value-head target. Without
it the value head cannot be trained and is disabled automatically (see Notes).
Regenerate a CSV that has it with:

```bash
python scripts/pgn_to_fen_move_csv.py \
  --input-pgn lichess_2200plus.pgn \
  --output-csv lichess_2200plus_fen_move_result.csv \
  --skip-unfinished
```

The trainer supports:

- SAN moves, like `e4`, `Nf3`, `Qxd5`
- UCI moves, like `e2e4`, `g1f3`

Default CSV auto-detection prefers:

1. `lichess_2200plus_fen_move.csv`
2. `lichess_2500plus_fen_move.csv`

## Main training command

```bash
cd /Users/daniellungu/Desktop/ChessEngine/ChessEngine/ai-engine
source venv/bin/activate
python train_csv_danibot.py \
  --csv lichess_2200plus_fen_move.csv \
  --epochs 5 \
  --chunk-size 8192 \
  --batch-size 384 \
  --checkpoint-every-chunks 50 \
  --progress-every-chunks 10
```

## Resume after interruption

```bash
cd /Users/daniellungu/Desktop/ChessEngine/ChessEngine/ai-engine
source venv/bin/activate
python train_csv_danibot.py --resume
```

Resume state is stored in:

- `checkpoints/danibot_csv_resume.pt`

## Quick smoke test

```bash
cd /Users/daniellungu/Desktop/ChessEngine/ChessEngine/ai-engine
source venv/bin/activate
python csv_training_smoke.py
```

This runs on a tiny subset of rows and saves smoke artifacts in `checkpoints/`.

## Validation

A fraction of whole *games* (not random rows) is held out via `--val-fraction`
(default `0.01`). Holding out whole games matters: consecutive CSV rows are
positions from the same game, so a random row split would leak near-identical
positions into validation and report inflated accuracy.

After each epoch the trainer prints held-out loss, top-1 and top-5 accuracy, and
warns when the train/val top-1 gap exceeds 5pp. The best-validation-loss weights
are saved alongside the final model as `danibot_best.pth` — prefer that file for
play, since the last epoch is not necessarily the strongest.

Game boundaries are detected by the FEN half-move counter failing to advance, not
by looking for the start position: the shipped CSVs are deduplicated, so the start
position appears exactly once in the entire file.

## Useful flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--val-fraction` | `0.01` | Fraction of games held out; `0` disables validation |
| `--legal-mask` | off | Restrict the policy softmax to legal moves, matching inference. Sharper policy, extra CPU per position |
| `--label-smoothing` | `0.05` | Softens the single-move target — several moves are usually reasonable |
| `--shuffle-buffer` | `16` | Chunks mixed before batching, so a batch spans many games |
| `--prefetch` | `3` | Chunks encoded ahead on a loader thread, overlapping encode with training |
| `--warmup-steps` / `--min-lr` | `500` / `1e-5` | Per-step warmup + cosine decay (clamped to ≤10% of the run) |
| `--amp` | CUDA only | fp16 on CUDA. On MPS, bf16 autocast measured ~35% *slower*, so it is opt-in |

## Notes

- The policy head is trained from the CSV move.
- The value head is trained on the game result from the side-to-move's perspective
  when the CSV has a `Result` column. If it does not, the value loss is switched
  off entirely rather than regressed toward `0` — an all-zero target actively
  teaches the network that every position is a draw, which makes the value output
  useless to MCTS.
- Final model is saved directly as `danibot.pth`, compatible with `ChessAI`;
  the best-validation checkpoint is `danibot_best.pth`.
