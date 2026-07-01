#!/usr/bin/env python3
"""
============================================================================
  DaniBot Leela-style Trainer
============================================================================

Antrenează DaniBot folosind datele V6 generate de generate_lc0_data.py.
La final, modelul este automat convertit în formatul ChessNet (18 planes,
move_mapping.json) și salvat ca danibot.pth — singurul fișier necesar.

Pipeline complet:
  1. Generează date:   python generate_lc0_data.py --mode pgn --pgn-file lichess_2500plus.pgn --games 1000
                   sau: python generate_lc0_data.py --mode stockfish --games 200 --stockfish /opt/homebrew/bin/stockfish
  2. Antrenează:        python train_leela.py --data-dir lc0_training_data --epochs 20
  3. Joacă:             Modelul salvat automat în danibot.pth, gata de joc cu ChessAI

Usage:
  python train_leela.py --data-dir lc0_training_data --epochs 10
  python train_leela.py --data-dir lc0_training_data --epochs 30 --batch-size 256 --lr 0.001
  python train_leela.py --data-dir lc0_training_data --epochs 50 --n-filters 256 --n-res-blocks 10
============================================================================
"""

import struct
import gzip
import os
import sys
import argparse
import glob
import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================================
#   CONSTANTE (identice cu generate_lc0_data.py)
# ============================================================================

LC0_INPUT_PLANES = 112
LC0_POLICY_SIZE  = 1858
V6_RECORD_SIZE   = 8356
V6_VERSION       = 6


# ============================================================================
#   MODEL INTERN:  _LeelaNet  (policy + value, 112 input planes)
#   Folosit doar pe durata training-ului. NU se salvează pe disk.
# ============================================================================

class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class _LeelaNet(nn.Module):
    """Rețea internă Leela-style. Input [B, 112, 8, 8] → policy [B, 1858] + value [B, 1]."""
    def __init__(self, n_filters=128, n_res_blocks=6):
        super().__init__()
        self.conv_in = nn.Conv2d(LC0_INPUT_PLANES, n_filters, 3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(n_filters)
        self.res_tower = nn.Sequential(*[_ResBlock(n_filters) for _ in range(n_res_blocks)])

        self.pol_conv = nn.Conv2d(n_filters, 32, 1, bias=False)
        self.pol_bn   = nn.BatchNorm2d(32)
        self.pol_fc   = nn.Linear(32 * 64, LC0_POLICY_SIZE)

        self.val_conv = nn.Conv2d(n_filters, 1, 1, bias=False)
        self.val_bn   = nn.BatchNorm2d(1)
        self.val_fc1  = nn.Linear(64, 128)
        self.val_fc2  = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_tower(x)
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = self.pol_fc(p.view(p.size(0), -1))
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = torch.tanh(self.val_fc2(F.relu(self.val_fc1(v.view(v.size(0), -1)))))
        return p, v


# ============================================================================
#   DATASET: citește .gz chunk-uri V6
# ============================================================================

class _Lc0Dataset(Dataset):
    def __init__(self, data_dir: str, max_positions: int = None):
        self.planes, self.policy, self.results = [], [], []

        gz_files = sorted(glob.glob(os.path.join(data_dir, "training.*.gz")))
        if not gz_files:
            raise FileNotFoundError(
                f"Nu am găsit fișiere training.*.gz în {data_dir}!\n"
                f"Rulează: python generate_lc0_data.py --mode pgn --pgn-file lichess_2500plus.pgn"
            )

        print(f"📂 Încarc {len(gz_files)} chunk-uri din {data_dir}/ …")
        total = 0
        for gz_path in gz_files:
            with gzip.open(gz_path, 'rb') as f:
                data = f.read()
            for i in range(len(data) // V6_RECORD_SIZE):
                off = i * V6_RECORD_SIZE
                rec = data[off:off + V6_RECORD_SIZE]
                if struct.unpack('<I', rec[:4])[0] != V6_VERSION:
                    continue
                self.planes.append(self._bits_to_planes(rec[4:4 + 112 * 8]))
                pol_off = 4 + 112 * 8
                self.policy.append(np.frombuffer(rec[pol_off:pol_off + 1858 * 4], dtype=np.float32).copy())
                res_off = pol_off + 1858 * 4
                self.results.append(float(struct.unpack('<i', rec[res_off:res_off + 4])[0]))
                total += 1
            if max_positions and total >= max_positions:
                break

        self.planes  = np.stack(self.planes)
        self.policy  = np.stack(self.policy)
        self.results = np.array(self.results, dtype=np.float32)
        print(f"✅ {len(self)} poziții încărcate")

    @staticmethod
    def _bits_to_planes(raw: bytes) -> np.ndarray:
        planes = np.zeros((112, 8, 8), dtype=np.float32)
        for p in range(112):
            bits = struct.unpack_from('<Q', raw, p * 8)[0]
            if not bits:
                continue
            for r in range(8):
                for c in range(8):
                    if bits & (1 << (r * 8 + c)):
                        planes[p, r, c] = 1.0
        return planes

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.planes[idx]),
                torch.from_numpy(self.policy[idx]),
                torch.tensor(self.results[idx], dtype=torch.float32))


# ============================================================================
#   LC0 MOVE INDEX → UCI  (pentru distillation)
# ============================================================================

def _build_lc0_idx_to_uci():
    """Construiește reverse-map:  lc0 policy index → list[UCI string]."""
    import chess
    queen_dirs = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    knight_d   = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]
    idx2uci, idx = {}, 0

    for sq in range(64):
        r, c = sq // 8, sq % 8
        for dx, dy in queen_dirs:
            for d in range(1, 8):
                tr, tc = r + dy*d, c + dx*d
                if 0 <= tr < 8 and 0 <= tc < 8:
                    uci = chess.square_name(sq) + chess.square_name(tr*8+tc)
                    idx2uci.setdefault(idx, []).append(uci)
                    if (r == 6 and tr == 7) or (r == 1 and tr == 0):
                        idx2uci[idx].append(uci + 'q')
                    idx += 1
    for sq in range(64):
        r, c = sq // 8, sq % 8
        for dx, dy in knight_d:
            tr, tc = r + dy, c + dx
            if 0 <= tr < 8 and 0 <= tc < 8:
                idx2uci.setdefault(idx, []).append(
                    chess.square_name(sq) + chess.square_name(tr*8+tc))
                idx += 1
    for sq in range(64):
        r, c = sq // 8, sq % 8
        for piece_ch in ['n','b','r']:
            for dc in [-1, 0, 1]:
                if r == 6:
                    tc = c + dc
                    if 0 <= tc < 8:
                        idx2uci.setdefault(idx, []).append(
                            chess.square_name(sq) + chess.square_name(7*8+tc) + piece_ch)
                        idx += 1
                if r == 1:
                    tc = c + dc
                    if 0 <= tc < 8:
                        idx2uci.setdefault(idx, []).append(
                            chess.square_name(sq) + chess.square_name(0*8+tc) + piece_ch)
                        idx += 1
    return idx2uci


# ============================================================================
#   DISTILLATION:  _LeelaNet → ChessNet → danibot.pth
# ============================================================================

def _distill_to_danibot(leela: _LeelaNet, device, output_path: str,
                        n_distill_epochs: int = 10, n_positions: int = 5000):
    """
    Knowledge distillation: _LeelaNet (intern, 112 planes) → ChessNet (18 planes).
    Generează poziții random, obține predicțiile Leela, și antrenează ChessNet.
    Salvează rezultatul ca danibot.pth.
    """
    import chess
    from ChessNet import ChessNet
    from ArchiveAlpha import encode_board_array
    from generate_lc0_data import encode_position_lc0

    print(f"\n{'='*70}")
    print(f"  🔄 Distillation: Leela → DaniBot ({output_path})")
    print(f"{'='*70}")

    # Move mappings
    with open('move_mapping.json', 'r') as f:
        move_list = json.load(f)
    move2idx = {m: i for i, m in enumerate(move_list)}
    n_moves = len(move_list)

    lc0_idx2uci = _build_lc0_idx_to_uci()

    # Generează poziții diverse
    print(f"  🎲 Generez {n_positions} poziții de distillation …")
    positions = []
    for _ in range(n_positions):
        board = chess.Board()
        for _ in range(random.randint(1, 80)):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(random.choice(legal))
        if not board.is_game_over():
            positions.append(board.copy())
    print(f"  ✅ {len(positions)} poziții valide")

    # Modele
    leela.eval()
    danibot = ChessNet(n_moves).to(device)

    # Dacă există deja un danibot.pth, îl încărcăm ca punct de start
    if os.path.exists(output_path):
        try:
            danibot.load_state_dict(torch.load(output_path, map_location=device))
            print(f"  📦 DaniBot existent încărcat ca start → {output_path}")
        except (RuntimeError, KeyError):
            print(f"  🆕 Arhitectură diferită, distill de la zero")

    danibot_opt = optim.Adam(danibot.parameters(), lr=1e-3)
    batch_size = 128

    for ep in range(n_distill_epochs):
        random.shuffle(positions)
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        danibot.train()

        for start in range(0, len(positions), batch_size):
            batch = positions[start:start + batch_size]

            # Encode Leela (112 planes) și obține predicții
            leela_in = torch.from_numpy(
                np.stack([encode_position_lc0(b) for b in batch])
            ).to(device)
            with torch.no_grad():
                leela_pol, _ = leela(leela_in)
                leela_probs = F.softmax(leela_pol, dim=1).cpu()  # [B, 1858]

            # Encode ChessNet (18 planes) și creează targets
            cn_inputs, targets = [], []
            for i, b in enumerate(batch):
                cn_inputs.append(encode_board_array(b))
                target = torch.zeros(n_moves)
                probs_i = leela_probs[i]
                for lc0_idx in range(LC0_POLICY_SIZE):
                    p = probs_i[lc0_idx].item()
                    if p < 1e-6:
                        continue
                    for uci in lc0_idx2uci.get(lc0_idx, []):
                        cn_idx = move2idx.get(uci)
                        if cn_idx is not None:
                            target[cn_idx] += p
                s = target.sum().item()
                if s > 0.01:
                    target /= s
                targets.append(target)

            cn_tensor = torch.from_numpy(np.stack(cn_inputs)).to(device)
            tgt_tensor = torch.stack(targets).to(device)

            danibot_opt.zero_grad(set_to_none=True)
            logits = danibot(cn_tensor)
            loss = -(tgt_tensor * F.log_softmax(logits, dim=1)).sum(1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(danibot.parameters(), 1.0)
            danibot_opt.step()

            ep_loss += loss.item() * len(batch)
            ep_correct += (logits.argmax(1) == tgt_tensor.argmax(1)).sum().item()
            ep_n += len(batch)

        avg = ep_loss / max(ep_n, 1)
        acc = 100.0 * ep_correct / max(ep_n, 1)
        print(f"  Distill {ep+1}/{n_distill_epochs} | Loss: {avg:.4f} | Acc: {acc:.1f}%")
        sys.stdout.flush()

        # Save periodic la fiecare 3 epoci de distillation
        if (ep + 1) % 3 == 0:
            torch.save(danibot.state_dict(), output_path)
            print(f"  💾 DaniBot salvat (distill epoch {ep+1}) → {output_path}")
            sys.stdout.flush()

    torch.save(danibot.state_dict(), output_path)
    print(f"\n  ✅ DaniBot salvat → {output_path}")


# ============================================================================
#   TRAINING LOOP
# ============================================================================

def train(args):
    t_start = time.time()

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = _Lc0Dataset(args.data_dir, max_positions=args.max_positions)
    n = len(dataset)
    n_val = max(1, int(n * 0.05))
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0,
                              pin_memory=(device.type == 'cuda'))
    print(f"📊 Train: {n_train}, Val: {n_val}, Batches/epoch: {len(train_loader)}")

    # ── Model intern (Leela, checkpoint temporar pe disk) ───────────────
    leela = _LeelaNet(n_filters=args.n_filters, n_res_blocks=args.n_res_blocks).to(device)
    n_params = sum(p.numel() for p in leela.parameters())
    print(f"🧠 Leela intern: {args.n_filters} filters, {args.n_res_blocks} blocks, {n_params:,} params")

    # Resume din checkpoint dacă există (crash recovery)
    ckpt_path = '.leela_checkpoint.pth'
    if os.path.exists(ckpt_path):
        try:
            leela.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"♻️  Checkpoint Leela restaurat din {ckpt_path}")
        except (RuntimeError, KeyError) as e:
            print(f"⚠️  Checkpoint invalid ({e}), pornesc de la zero")

    optimizer = optim.Adam(leela.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    vw = args.value_weight
    best_val = float('inf')

    # ── Leela Training ────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        leela.train()
        ep_pol, ep_val, ep_tot, ep_ok, ep_n = 0., 0., 0., 0, 0
        ep_t = time.time()

        for bi, (planes, tgt_pol, tgt_val) in enumerate(train_loader):
            planes  = planes.to(device)
            tgt_pol = tgt_pol.to(device)
            tgt_val = tgt_val.to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            pred_p, pred_v = leela(planes)

            p_loss = -(tgt_pol * F.log_softmax(pred_p, dim=1)).sum(1).mean()
            v_loss = F.mse_loss(pred_v, tgt_val)
            loss = p_loss + vw * v_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(leela.parameters(), 1.0)
            optimizer.step()

            bs = planes.size(0)
            ep_pol += p_loss.item() * bs
            ep_val += v_loss.item() * bs
            ep_tot += loss.item() * bs
            ep_ok  += (pred_p.argmax(1) == tgt_pol.argmax(1)).sum().item()
            ep_n   += bs

            if (bi + 1) % 20 == 0 or (bi + 1) == len(train_loader):
                print(f"  Batch {bi+1}/{len(train_loader)} | "
                      f"Policy: {ep_pol/ep_n:.4f} | Value: {ep_val/ep_n:.4f} | "
                      f"Acc: {100.*ep_ok/ep_n:.1f}% | Pos: {ep_n}")
                sys.stdout.flush()

        scheduler.step()
        dt = time.time() - ep_t

        # Validation
        leela.eval()
        vp, vv, vn, vc = 0., 0., 0, 0
        with torch.no_grad():
            for planes, tgt_pol, tgt_val in val_loader:
                planes  = planes.to(device)
                tgt_pol = tgt_pol.to(device)
                tgt_val = tgt_val.to(device).unsqueeze(1)
                pp, pv = leela(planes)
                vp += -(tgt_pol * F.log_softmax(pp, dim=1)).sum(1).sum().item()
                vv += F.mse_loss(pv, tgt_val, reduction='sum').item()
                vn += planes.size(0)
                vc += (pp.argmax(1) == tgt_pol.argmax(1)).sum().item()

        vpa, vva = vp / max(vn, 1), vv / max(vn, 1)
        vtot = vpa + vw * vva

        print(f"\n{'='*70}")
        print(f"  EPOCH {epoch+1}/{args.epochs} | "
              f"Train: {ep_tot/ep_n:.4f} (pol={ep_pol/ep_n:.4f} val={ep_val/ep_n:.4f}) | "
              f"Acc: {100.*ep_ok/ep_n:.1f}%")
        print(f"  Val: {vtot:.4f} (pol={vpa:.4f} val={vva:.4f}) | "
              f"Val Acc: {100.*vc/max(vn,1):.1f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {dt:.1f}s ({ep_n/max(dt,1e-9):.0f} pos/s)")
        print(f"{'='*70}\n")
        sys.stdout.flush()

        if vtot < best_val:
            best_val = vtot
            print(f"  ⭐ Best val loss: {vtot:.4f}")

        # ── Periodic save: checkpoint Leela + distill → danibot.pth ───────
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            # Salvează checkpoint intern Leela (temporar, se șterge la final)
            ckpt_path = '.leela_checkpoint.pth'
            torch.save(leela.state_dict(), ckpt_path)
            print(f"  💾 Leela checkpoint → {ckpt_path}")

            # Distill intermediar → danibot.pth (3 epoci rapide)
            print(f"  🔄 Distill intermediar → {args.model_path} ...")
            _distill_to_danibot(
                leela, device,
                output_path=args.model_path,
                n_distill_epochs=min(5, args.distill_epochs),
                n_positions=min(3000, args.distill_positions),
            )
            print(f"  ✅ DaniBot actualizat la epoch {epoch+1}")
            sys.stdout.flush()

    # Cleanup checkpoint temporar
    if os.path.exists('.leela_checkpoint.pth'):
        os.remove('.leela_checkpoint.pth')

    # ── Distillation finală → danibot.pth ─────────────────────────────────
    total_leela_time = time.time() - t_start
    print(f"\n🏁 Leela training complet în {total_leela_time:.0f}s ({total_leela_time/60:.1f} min)")
    print(f"   Acum convertesc → danibot.pth …\n")

    _distill_to_danibot(
        leela, device,
        output_path=args.model_path,
        n_distill_epochs=args.distill_epochs,
        n_positions=args.distill_positions,
    )

    total_time = time.time() - t_start
    print(f"\n✅ Complet! {args.epochs} epochs Leela + distillation în {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"   Model final: {args.model_path}")
    print(f"   Gata de joc cu ChessAI! 🎮")


# ============================================================================
#   MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DaniBot Leela-style Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline complet:

  # Pas 1: Generează date de training
  python generate_lc0_data.py --mode pgn --pgn-file lichess_2500plus.pgn --games 1000

  # Pas 2: Antrenează DaniBot  (un singur fișier → danibot.pth)
  python train_leela.py --data-dir lc0_training_data --epochs 20

  # Alternativ cu Stockfish self-play:
  python generate_lc0_data.py --mode stockfish --games 200 \\
      --stockfish /opt/homebrew/bin/stockfish
  python train_leela.py --data-dir lc0_training_data --epochs 10
        """,
    )

    # Data
    parser.add_argument('--data-dir', default='lc0_training_data',
                        help='Director cu fișierele training.*.gz')
    parser.add_argument('--max-positions', type=int, default=None,
                        help='Limită maximă de poziții')

    # Leela architecture
    parser.add_argument('--n-filters', type=int, default=128,
                        help='Filtre în residual tower (default: 128)')
    parser.add_argument('--n-res-blocks', type=int, default=6,
                        help='Blocuri residuale (default: 6)')

    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epoci Leela training (default: 20)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate (default: 0.002)')
    parser.add_argument('--value-weight', type=float, default=1.0,
                        help='Ponderea value loss (default: 1.0)')

    # Distillation
    parser.add_argument('--distill-epochs', type=int, default=10,
                        help='Epoci distillation Leela → DaniBot (default: 10)')
    parser.add_argument('--distill-positions', type=int, default=5000,
                        help='Poziții generate pentru distillation (default: 5000)')

    # Output (SINGURUL model pe disk)
    parser.add_argument('--model-path', default='danibot.pth',
                        help='Calea modelului final (default: danibot.pth)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

