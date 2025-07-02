#!/usr/bin/env python3
import argparse
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json

# ── dataset utilities ─────────────────────────────────────────────────────────

def load_samples(pgn_path, color):
    """
    Scan the PGN and collect (fen, move_uci) pairs for the given side to move.
    """
    samples = []
    with open(pgn_path, 'r', encoding='utf-8') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                if board.turn == color:
                    samples.append((board.fen(), move.uci()))
                board.push(move)
    return samples

class ChessDataset(Dataset):
    def __init__(self, samples, move2idx):
        """
        samples: list of (fen, uci)
        move2idx: dict mapping uci→int label
        """
        self.samples = samples
        self.move2idx = move2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, uci = self.samples[idx]
        board = chess.Board(fen)
        x = encode_board(board)              # [12×8×8] float tensor
        y = self.move2idx[uci]               # integer label
        return x, y

def encode_board(board):
    """
    12‐plane binary feature:
      planes 0–5  = white pawn, knight, bishop, rook, queen, king
      planes 6–11 = black pawn, knight, bishop, rook, queen, king
    Returns a torch.FloatTensor of shape [12,8,8].
    """
    arr = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        pt = piece.piece_type  # 1..6
        color = piece.color    # True=white, False=black
        plane = (pt - 1) + (0 if color else 6)
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        arr[plane, rank, file] = 1.0
    return torch.from_numpy(arr)

# ── model ────────────────────────────────────────────────────────────────────

class ChessNet(nn.Module):
    def __init__(self, n_moves):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 8 * 8, n_moves)

    def forward(self, x):
        # x: [B,12,8,8]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ── training loop ───────────────────────────────────────────────────────────

def train(model, loader, device, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    for ep in range(1, epochs+1):
        print(f"\nEpoch {ep}/{epochs} - training on {len(loader.dataset)} samples in {len(loader)} batches")
        model.train()
        total_loss = 0.0
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(loader)} - loss: {loss.item():.4f}")
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"Epoch {ep}/{epochs}  –  loss: {avg:.4f}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pgn',        default='lichess_db.pgn',
                   help='PGN file of games')
    p.add_argument('--epochs',     type=int, default=5)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--chunk_size', type=int, default=100,
                   help='number of games to process per chunk')
    p.add_argument('--resume_chunk', type=int, default=0,
                   help='chunk index to resume training from (0 to start fresh)')
    args = p.parse_args()
    print(f"Arguments: pgn={args.pgn}, epochs={args.epochs}, batch_size={args.batch_size}")

    # Load static move mappings from JSON file
    print("Loading move mappings from move_mapping.json...")
    with open('move_mapping.json', 'r', encoding='utf-8') as fmap:
        move_list = json.load(fmap)
    w2i = {m: i for i, m in enumerate(move_list)}
    b2i = w2i
    print(f"Loaded {len(move_list)} moves from mapping file.")

    # Initialize device and networks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    net_w = ChessNet(len(move_list))
    net_b = ChessNet(len(move_list))
    net_w.to(device)
    net_b.to(device)

    # Determine starting chunk and load checkpoint if resuming
    chunk_idx = args.resume_chunk
    if args.resume_chunk > 0:
        # Load checkpoint for both models
        ckpt_w = f'trained_model_white_chunk{chunk_idx}.pth'
        ckpt_b = f'trained_model_black_chunk{chunk_idx}.pth'
        if os.path.exists(ckpt_w) and os.path.exists(ckpt_b):
            print(f"Resuming from chunk {chunk_idx}: loading {ckpt_w} and {ckpt_b}")
            state_w = torch.load(ckpt_w, map_location=device)
            state_b = torch.load(ckpt_b, map_location=device)
            net_w.load_state_dict(state_w, strict=False)
            net_b.load_state_dict(state_b, strict=False)
            print("✅ Checkpoint loaded.")
        else:
            print(f"⚠️ Checkpoints for chunk {chunk_idx} not found, starting fresh.")

    # Chunked training over PGN
    with open(args.pgn, 'r', encoding='utf-8') as f:
        # Skip already-processed games
        games_to_skip = args.resume_chunk * args.chunk_size
        for _ in range(games_to_skip):
            if chess.pgn.read_game(f) is None:
                break
        while True:
            chunk_idx += 1
            games = []
            for _ in range(args.chunk_size):
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                games.append(g)
            if not games:
                break
            # collect samples for this chunk
            white_samples = []
            black_samples = []
            for game in games:
                board = game.board()
                for move in game.mainline_moves():
                    if board.turn == chess.WHITE:
                        white_samples.append((board.fen(), move.uci()))
                    else:
                        black_samples.append((board.fen(), move.uci()))
                    board.push(move)
            print(f"Chunk: {len(games)} games, {len(white_samples)} white samples, {len(black_samples)} black samples")
            # create loaders and train one epoch per chunk
            w_loader = DataLoader(
                ChessDataset(white_samples, w2i),
                batch_size=args.batch_size, shuffle=True,
                num_workers=2, pin_memory=True
            )
            b_loader = DataLoader(
                ChessDataset(black_samples, b2i),
                batch_size=args.batch_size, shuffle=True,
                num_workers=2, pin_memory=True
            )
            train(net_w, w_loader, device, epochs=1)
            train(net_b, b_loader, device, epochs=1)

            if chunk_idx % 100 == 0:
                torch.save(net_w.state_dict(), f'trained_model_white_chunk{chunk_idx}.pth')
                torch.save(net_b.state_dict(), f'trained_model_black_chunk{chunk_idx}.pth')
                print(f"Saved models at chunk {chunk_idx}")

    # Save final models
    torch.save(net_w.state_dict(), 'trained_model_white.pth')
    torch.save(net_b.state_dict(), 'trained_model_black.pth')
    print("Saved final models.")

if __name__ == '__main__':
    main()