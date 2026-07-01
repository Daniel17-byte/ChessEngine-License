import os
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from ChessNet import ChessNet
from ArchiveAlpha import encode_board
import chess


# ===== Dataset using pre-computed Stockfish labels =====
class PrecomputedDataset(Dataset):
    """Fast dataset that loads pre-computed Stockfish labels (no Stockfish at runtime)."""
    def __init__(self, labels_path="stockfish_labels.json", mapping_path="move_mapping.json"):
        with open(mapping_path) as f:
            self.idx_to_move = json.load(f)
        self.move_to_idx = {uci: i for i, uci in enumerate(self.idx_to_move)}
        self.n_moves = len(self.idx_to_move)

        with open(labels_path) as f:
            raw = json.load(f)

        # Filter: keep only valid (fen → move) pairs
        self.samples = [(fen, move) for fen, move in raw.items() if move is not None and move in self.move_to_idx]
        print(f"📦 Loaded {len(self.samples)} pre-computed training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, move_uci = self.samples[idx]
        board = chess.Board(fen)
        board_tensor = encode_board(board)  # [18, 8, 8]
        move_idx = self.move_to_idx[move_uci]
        return board_tensor, move_idx


# ===== Fallback: live Stockfish dataset (slow but works without pre-computation) =====
class LiveStockfishDataset(Dataset):
    """Slow dataset that queries Stockfish at runtime for each FEN."""
    def __init__(self, fens, engine_path, time_limit=0.05, mapping_path="move_mapping.json"):
        self.fens = fens
        self.time_limit = time_limit

        with open(mapping_path) as f:
            self.idx_to_move = json.load(f)
        self.move_to_idx = {uci: i for i, uci in enumerate(self.idx_to_move)}
        self.n_moves = len(self.idx_to_move)

        if os.path.exists(engine_path):
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        else:
            self.engine = None
            print(f"⚠️ Stockfish not found at {engine_path}")

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        board = chess.Board(fen)
        board_tensor = encode_board(board)

        best_move = self._get_stockfish_move(fen)
        move_idx = self.move_to_idx.get(best_move, 0)
        return board_tensor, move_idx

    def _get_stockfish_move(self, fen):
        if self.engine is None:
            return "0000"
        board = chess.Board(fen)
        if board.is_game_over():
            return "0000"
        result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
        return result.move.uci() if result.move else "0000"

    def __del__(self):
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.quit()


# ===== Utils =====
def load_fens_from_files(filepath="generated_games.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return json.load(file)
    return []


# ===== Main entry point =====
if __name__ == '__main__':
    import chess.engine
    import argparse

    parser = argparse.ArgumentParser(description='Stockfish Trainer')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    # ===== Parameters =====
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    engine_path = "/opt/homebrew/bin/stockfish"
    model_path = "danibot.pth"
    labels_path = "stockfish_labels.json"

    # ===== Choose dataset: pre-computed (fast) or live (slow) =====
    if os.path.exists(labels_path):
        print("🚀 Using pre-computed Stockfish labels (FAST mode)")
        dataset = PrecomputedDataset(labels_path=labels_path)
    else:
        print("🐌 Pre-computed labels not found. Using live Stockfish (SLOW mode)")
        print("   Run 'python precompute_stockfish.py' first for 10-50x faster training!")
        fens = load_fens_from_files("generated_games.json")
        if len(fens) == 0:
            raise ValueError("generated_games.json is empty or does not exist!")
        dataset = LiveStockfishDataset(fens, engine_path=engine_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # ===== Model, loss, optimizer =====
    n_moves = dataset.n_moves
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}, possible moves: {n_moves}, samples: {len(dataset)}, batches/epoch: {len(dataloader)}")

    model = ChessNet(n_moves).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # ===== Load existing model if available =====
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("✅ Existing model loaded, continuing training.")
        except (RuntimeError, KeyError) as e:
            print(f"⚠️ Could not load old model: {e}")
            print("Architecture changed — training from scratch.")

    # ===== Training =====
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        print(f"===== EPOCH {epoch+1}/{epochs} =====")
        sys.stdout.flush()

        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_samples += X.size(0)

            # Accuracy tracking
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()

            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                acc = 100.0 * correct / total_samples
                avg_loss = total_loss / (i + 1)
                print(
                    f"Batch {i+1}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Acc: {acc:.1f}% | "
                    f"FEN processed: {total_samples}"
                )
                sys.stdout.flush()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        acc = 100.0 * correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f} - Accuracy: {acc:.1f}%\n")
        sys.stdout.flush()

        # Save after each epoch
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved after epoch {epoch+1}")
        sys.stdout.flush()

    print(f"✅ Training complete! Model saved to '{model_path}'")
    sys.stdout.flush()
