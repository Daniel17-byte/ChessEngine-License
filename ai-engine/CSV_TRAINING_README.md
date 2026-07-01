# CSV DaniBot Training

Train `danibot.pth` directly from a `FEN,Move` CSV.

## Input format

Expected columns:

```csv
FEN,Move
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

## Notes

- The policy head is trained from the CSV move.
- The value head is regularized toward `0` because the CSV has no game-result label.
- Final model is saved directly as `danibot.pth`, compatible with `ChessAI`.
