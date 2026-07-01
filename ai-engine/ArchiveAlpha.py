import argparse
import sys
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
from ChessNet import ChessNet
from contextlib import nullcontext
import os
import json
import time

USE_CYTHON_ENCODE = os.getenv("CHESS_ENCODE_FORCE_PYTHON", "0") != "1"
if USE_CYTHON_ENCODE:
    try:
        from fastgame.board_encode import encode_board_array as cy_encode_board_array
        HAS_CYTHON_ENCODE = True
    except ImportError:
        cy_encode_board_array = None
        HAS_CYTHON_ENCODE = False
    try:
        from fastgame.board_encode import encode_board_batch as cy_encode_board_batch
        HAS_CYTHON_BATCH = True
    except ImportError:
        cy_encode_board_batch = None
        HAS_CYTHON_BATCH = False
else:
    cy_encode_board_array = None
    HAS_CYTHON_ENCODE = False
    cy_encode_board_batch = None
    HAS_CYTHON_BATCH = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    njit = None
    HAS_NUMBA = False

# ── dataset utilities ─────────────────────────────────────────────────────────

# Removed per-sample FEN parsing dataset; chunk tensors are built directly
# from live board states to avoid expensive board reconstruction in __getitem__.

def _encode_board_numpy(board):
    """Fast NumPy encoding using bitboards."""
    arr = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes from bitboards (python-chess stores them as uint64 masks)
    piece_specs = (
        (chess.PAWN, chess.WHITE, 0),
        (chess.KNIGHT, chess.WHITE, 1),
        (chess.BISHOP, chess.WHITE, 2),
        (chess.ROOK, chess.WHITE, 3),
        (chess.QUEEN, chess.WHITE, 4),
        (chess.KING, chess.WHITE, 5),
        (chess.PAWN, chess.BLACK, 6),
        (chess.KNIGHT, chess.BLACK, 7),
        (chess.BISHOP, chess.BLACK, 8),
        (chess.ROOK, chess.BLACK, 9),
        (chess.QUEEN, chess.BLACK, 10),
        (chess.KING, chess.BLACK, 11),
    )

    for piece_type, color, plane in piece_specs:
        bb = board.pieces_mask(piece_type, color)
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            arr[plane, sq // 8, sq % 8] = 1.0
            bb ^= lsb

    # Turn indicator
    if board.turn == chess.WHITE:
        arr[12, :, :] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        arr[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        arr[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        arr[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        arr[16, :, :] = 1.0

    # En passant
    if board.ep_square is not None:
        arr[17, board.ep_square // 8, board.ep_square % 8] = 1.0

    return arr


if HAS_NUMBA:
    @njit(cache=True)
    def _fill_piece_planes_numba(arr, masks):
        for plane in range(12):
            bb = masks[plane]
            for sq in range(64):
                if (bb >> sq) & np.uint64(1):
                    arr[plane, sq // 8, sq % 8] = 1.0


    def _encode_board_numba(board):
        arr = np.zeros((18, 8, 8), dtype=np.float32)
        masks = np.array([
            board.pieces_mask(chess.PAWN, chess.WHITE),
            board.pieces_mask(chess.KNIGHT, chess.WHITE),
            board.pieces_mask(chess.BISHOP, chess.WHITE),
            board.pieces_mask(chess.ROOK, chess.WHITE),
            board.pieces_mask(chess.QUEEN, chess.WHITE),
            board.pieces_mask(chess.KING, chess.WHITE),
            board.pieces_mask(chess.PAWN, chess.BLACK),
            board.pieces_mask(chess.KNIGHT, chess.BLACK),
            board.pieces_mask(chess.BISHOP, chess.BLACK),
            board.pieces_mask(chess.ROOK, chess.BLACK),
            board.pieces_mask(chess.QUEEN, chess.BLACK),
            board.pieces_mask(chess.KING, chess.BLACK),
        ], dtype=np.uint64)

        _fill_piece_planes_numba(arr, masks)

        if board.turn == chess.WHITE:
            arr[12, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            arr[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            arr[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            arr[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            arr[16, :, :] = 1.0
        if board.ep_square is not None:
            arr[17, board.ep_square // 8, board.ep_square % 8] = 1.0

        return arr


def encode_board_array(board):
    """18-plane feature encoding with Cython/Numba/NumPy fallback (NumPy array)."""
    if HAS_CYTHON_ENCODE:
        arr = cy_encode_board_array(board)
    elif HAS_NUMBA:
        arr = _encode_board_numba(board)
    else:
        arr = _encode_board_numpy(board)
    return arr.astype(np.float32, copy=False)


def encode_board(board):
    """Backward-compatible tensor helper used by gameplay/training scripts."""
    return torch.from_numpy(encode_board_array(board))


def _result_to_value(result_str: str, turn_is_white: bool) -> float:
    """Convert game result to value from perspective of side to move."""
    if result_str == '1-0':
        return 1.0 if turn_is_white else -1.0
    elif result_str == '0-1':
        return -1.0 if turn_is_white else 1.0
    else:
        return 0.0


def build_chunk_tensors(games, move2idx):
    """Encode one chunk of games into contiguous tensors once, then train from them.

    Uses Cython batch encoding when available for ~3-5x speedup on the encoding step.
    Returns (x_tensor, y_policy_tensor, y_value_tensor) or (None, None, None).
    """
    # Collect boards and labels first, then batch-encode
    boards = []
    y_list = []
    v_list = []

    for game in games:
        result_str = game.headers.get("Result", "*")
        if result_str not in ("1-0", "0-1", "1/2-1/2"):
            continue
        board = game.board()
        for move in game.mainline_moves():
            idx = move2idx.get(move.uci())
            if idx is not None:
                boards.append(board.copy())
                y_list.append(idx)
                v_list.append(_result_to_value(result_str, board.turn == chess.WHITE))
            board.push(move)

    if not boards:
        return None, None, None

    # Batch encode all boards at once (Cython fast path)
    if HAS_CYTHON_BATCH:
        x_np = cy_encode_board_batch(boards)
    else:
        x_list = [encode_board_array(b) for b in boards]
        x_np = np.stack(x_list, axis=0)

    y_np = np.asarray(y_list, dtype=np.int64)
    v_np = np.asarray(v_list, dtype=np.float32)
    return torch.from_numpy(x_np), torch.from_numpy(y_np), torch.from_numpy(v_np)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Train on Lichess PGN archive')
    p.add_argument('--pgn',          default='lichess_2350plus.pgn', help='PGN file of games')
    p.add_argument('--epochs',       type=int, default=10, help='Number of training epochs')
    p.add_argument('--batch-size',   type=int, default=None, help='Positions per batch (auto-tuned per device when omitted)')
    p.add_argument('--chunk-size',   type=int, default=300, help='Games per chunk')
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--model-path',   default='danibot.pth', help='Path to save/load model')
    p.add_argument('--amp', dest='amp', action='store_true', help='Enable mixed precision on CUDA/MPS')
    p.add_argument('--no-amp', dest='amp', action='store_false', help='Disable mixed precision')
    p.add_argument('--compile', dest='compile_model', action='store_true', help='Enable torch.compile for model')
    p.add_argument('--no-compile', dest='compile_model', action='store_false', help='Disable torch.compile')
    p.add_argument('--chunk-on-device', dest='chunk_on_device', action='store_true', help='Move whole chunk to device before batching')
    p.add_argument('--no-chunk-on-device', dest='chunk_on_device', action='store_false', help='Keep chunk on CPU and copy per batch')
    p.add_argument('--grad-clip', type=float, default=1.0, help='Gradient norm clip value; <=0 disables clipping')
    p.add_argument('--training-only', dest='training_only', action='store_true', help='Use throughput-focused training mode (skip accuracy metrics)')
    p.add_argument('--full-metrics', dest='training_only', action='store_false', help='Compute full accuracy metrics during training')
    p.set_defaults(amp=None, compile_model=False, chunk_on_device=None, training_only=True)
    args = p.parse_args()

    # ── Load move mapping ─────────────────────────────────────────────────
    print("Loading move mappings from move_mapping.json...")
    sys.stdout.flush()
    with open('move_mapping.json', 'r', encoding='utf-8') as fmap:
        move_list = json.load(fmap)
    move2idx = {m: i for i, m in enumerate(move_list)}
    n_moves = len(move_list)
    print(f"Loaded {n_moves} moves from mapping file.")
    sys.stdout.flush()

    # ── Check PGN file ────────────────────────────────────────────────────
    if not os.path.exists(args.pgn):
        print(f"❌ PGN file not found: {args.pgn}")
        print("Download a Lichess database from https://database.lichess.org/")
        sys.stdout.flush()
        sys.exit(1)

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Device-aware defaults for no-argument runs; explicit CLI values still win.
    if device.type == 'mps':
        default_batch_size = 384
        default_amp = False
        default_chunk_on_device = False
    elif device.type == 'cuda':
        default_batch_size = 256
        default_amp = True
        default_chunk_on_device = True
    else:
        default_batch_size = 256
        default_amp = False
        default_chunk_on_device = False

    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    amp_enabled = args.amp if args.amp is not None else default_amp
    chunk_on_device = args.chunk_on_device if args.chunk_on_device is not None else default_chunk_on_device
    print(
        f"Device: {device}, moves: {n_moves}, batch_size: {batch_size}, amp: {amp_enabled}, "
        f"compile: {args.compile_model}, chunk_on_device: {chunk_on_device}, training_only: {args.training_only}"
    )
    sys.stdout.flush()

    # ── Single shared model (consistent with rest of app) ─────────────────
    model = ChessNet(n_moves).to(device)

    adam_kwargs = dict(lr=args.lr, weight_decay=1e-5)
    if device.type == 'cuda':
        try:
            optimizer = torch.optim.Adam(model.parameters(), fused=True, **adam_kwargs)
        except TypeError:
            optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)

    criterion = nn.CrossEntropyLoss()

    # Load existing model if available
    if os.path.exists(args.model_path):
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"✅ Existing model loaded from {args.model_path}")
        except (RuntimeError, KeyError) as e:
            print(f"⚠️ Could not load old model: {e}")
            print("Architecture changed — training from scratch.")
    sys.stdout.flush()

    if args.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("✅ torch.compile enabled")
        except Exception as e:
            print(f"⚠️ torch.compile failed, continuing without it: {e}")
        sys.stdout.flush()

    if amp_enabled and device.type in ('cuda', 'mps'):
        autocast_ctx = lambda: torch.autocast(device_type=device.type, dtype=torch.float16)
    else:
        autocast_ctx = nullcontext

    # Use the newer GradScaler API when available to avoid deprecation warnings.
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and device.type == 'cuda')
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == 'cuda')

    # ── Chunked training over PGN ─────────────────────────────────────────
    for epoch in range(args.epochs):
        print(f"===== EPOCH {epoch+1}/{args.epochs} =====")
        sys.stdout.flush()

        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_batches = 0
        chunk_idx = 0

        with open(args.pgn, 'r', encoding='utf-8') as f:
            while True:
                # Read a chunk of games
                games = []
                for _ in range(args.chunk_size):
                    g = chess.pgn.read_game(f)
                    if g is None:
                        break
                    games.append(g)

                if not games:
                    break

                chunk_idx += 1
                prep_start = time.perf_counter()
                result = build_chunk_tensors(games, move2idx)
                prep_time = time.perf_counter() - prep_start

                if result[0] is None:
                    continue
                x_chunk, y_chunk, v_chunk = result

                # Pin host memory for faster H2D copies when batching from CPU tensors.
                if device.type == 'cuda' and not chunk_on_device:
                    x_chunk = x_chunk.pin_memory()
                    y_chunk = y_chunk.pin_memory()
                    v_chunk = v_chunk.pin_memory()

                x_chunk_dev = None
                y_chunk_dev = None
                v_chunk_dev = None
                used_chunk_on_device = False
                move_start = time.perf_counter()
                if chunk_on_device and device.type in ('cuda', 'mps'):
                    try:
                        x_chunk_dev = x_chunk.to(device, non_blocking=(device.type == 'cuda'))
                        y_chunk_dev = y_chunk.to(device, non_blocking=(device.type == 'cuda'))
                        v_chunk_dev = v_chunk.to(device, non_blocking=(device.type == 'cuda'))
                        used_chunk_on_device = True
                    except RuntimeError as e:
                        print(f"⚠️ chunk-on-device disabled for this chunk (OOM/fallback): {e}")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                move_time = time.perf_counter() - move_start

                model.train()
                chunk_total = 0
                chunk_loss_tensor = torch.zeros((), device=device)
                chunk_correct_tensor = torch.zeros((), device=device, dtype=torch.int64) if not args.training_only else None

                num_samples = x_chunk.size(0)
                n_batches = (num_samples + batch_size - 1) // batch_size
                perm_device = device if used_chunk_on_device else torch.device('cpu')
                indices = torch.randperm(num_samples, device=perm_device)
                train_start = time.perf_counter()

                for start in range(0, num_samples, batch_size):
                    batch_indices = indices[start:start + batch_size]

                    if used_chunk_on_device:
                        xb = x_chunk_dev.index_select(0, batch_indices)
                        yb = y_chunk_dev.index_select(0, batch_indices)
                        vb = v_chunk_dev.index_select(0, batch_indices)
                    else:
                        xb = x_chunk.index_select(0, batch_indices)
                        yb = y_chunk.index_select(0, batch_indices)
                        vb = v_chunk.index_select(0, batch_indices)
                        xb = xb.to(device, non_blocking=(device.type == 'cuda'))
                        yb = yb.to(device, non_blocking=(device.type == 'cuda'))
                        vb = vb.to(device, non_blocking=(device.type == 'cuda'))

                    optimizer.zero_grad(set_to_none=True)

                    with autocast_ctx():
                        output = model(xb)
                        # Support both (policy, value) and policy-only models
                        if isinstance(output, tuple):
                            logits, value_pred = output
                            policy_loss = criterion(logits, yb)
                            value_loss = nn.functional.mse_loss(value_pred.squeeze(-1), vb)
                            loss = policy_loss + value_loss
                        else:
                            logits = output
                            loss = criterion(logits, yb)

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        if args.grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()

                    chunk_total += xb.size(0)
                    chunk_loss_tensor += loss.detach()
                    if not args.training_only:
                        with torch.no_grad():
                            _, predicted = logits.max(1)
                            chunk_correct_tensor += predicted.eq(yb).sum()

                train_time = time.perf_counter() - train_start
                chunk_loss = chunk_loss_tensor.item()
                chunk_correct = int(chunk_correct_tensor.item()) if not args.training_only else 0

                epoch_loss += chunk_loss
                if not args.training_only:
                    epoch_correct += chunk_correct
                epoch_total += chunk_total
                epoch_batches += n_batches

                # Print progress every chunk
                if chunk_idx % 10 == 0 or len(games) < args.chunk_size:
                    total_avg_loss = epoch_loss / max(epoch_batches, 1)
                    chunk_avg_loss = chunk_loss / max(n_batches, 1)
                    pos_per_sec = chunk_total / max(train_time, 1e-9)
                    if args.training_only:
                        print(
                            f"Batch {chunk_idx} | "
                            f"Chunk: {len(games)} games, {num_samples} positions | "
                            f"Prep: {prep_time:.2f}s | "
                            f"Move: {move_time:.2f}s | "
                            f"Train: {train_time:.2f}s ({pos_per_sec:.0f} pos/s) | "
                            f"Loss: {chunk_avg_loss:.4f} | "
                            f"Avg Loss: {total_avg_loss:.4f} | "
                            f"Positions: {epoch_total}"
                        )
                    else:
                        total_acc = 100.0 * epoch_correct / max(epoch_total, 1)
                        print(
                            f"Batch {chunk_idx} | "
                            f"Chunk: {len(games)} games, {num_samples} positions | "
                            f"Prep: {prep_time:.2f}s | "
                            f"Move: {move_time:.2f}s | "
                            f"Train: {train_time:.2f}s ({pos_per_sec:.0f} pos/s) | "
                            f"Loss: {chunk_avg_loss:.4f} | "
                            f"Avg Loss: {total_avg_loss:.4f} | "
                            f"Acc: {total_acc:.1f}% | "
                            f"Positions: {epoch_total}"
                        )
                    sys.stdout.flush()

                # Save checkpoint every 100 chunks
                if chunk_idx % 100 == 0:
                    torch.save(model.state_dict(), args.model_path)
                    print(f"💾 Checkpoint saved at chunk {chunk_idx}")
                    sys.stdout.flush()

        # End of epoch
        epoch_time = time.perf_counter() - epoch_start
        if epoch_total > 0:
            avg_loss = epoch_loss / max(epoch_batches, 1)
            pos_per_sec = epoch_total / max(epoch_time, 1e-9)
            if args.training_only:
                print(
                    f"Epoch {epoch+1}/{args.epochs} - "
                    f"Avg Loss: {avg_loss:.4f} - "
                    f"Positions: {epoch_total} - "
                    f"Time: {epoch_time:.1f}s ({pos_per_sec:.0f} pos/s)"
                )
            else:
                acc = 100.0 * epoch_correct / epoch_total
                print(
                    f"Epoch {epoch+1}/{args.epochs} - "
                    f"Avg Loss: {avg_loss:.4f} - "
                    f"Accuracy: {acc:.1f}% - "
                    f"Positions: {epoch_total} - "
                    f"Time: {epoch_time:.1f}s ({pos_per_sec:.0f} pos/s)"
                )
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - No positions processed")
        sys.stdout.flush()

        # Save after each epoch
        torch.save(model.state_dict(), args.model_path)
        print(f"💾 Model saved to {args.model_path}")
        sys.stdout.flush()

    print(f"✅ Training complete! Model saved to '{args.model_path}'")
    sys.stdout.flush()


if __name__ == '__main__':
    main()

