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
import math
import queue
import threading
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

def lr_at_step(step, peak_lr, min_lr, warmup_steps, total_steps):
    """Linear warmup then cosine decay, evaluated per optimizer step.

    The old schedule was a constant LR for the whole run, which leaves accuracy
    on the table late in training.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    total = max(total_steps, warmup_steps + 1)
    progress = (step - warmup_steps) / max(total - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def prefetch_iter(iterable, depth):
    """Run `iterable` on a background thread so PGN parsing overlaps with training."""
    if depth <= 0:
        yield from iterable
        return

    q = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for item in iterable:
                q.put(item)
        except BaseException as exc:
            q.put(exc)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        while thread.is_alive():
            try:
                if q.get(timeout=0.1) is sentinel:
                    break
            except queue.Empty:
                pass


def read_game_chunks(pgn_path, chunk_size, move2idx, encoding='utf-8', skip_games=0):
    """Yield (x, y, v, n_games) tensors, one chunk of games at a time.

    `skip_games` steps over the validation games that sit at the head of the file.
    """
    with open(pgn_path, 'r', encoding=encoding, errors='replace') as f:
        for _ in range(skip_games):
            if chess.pgn.read_game(f) is None:
                return
        while True:
            games = []
            for _ in range(chunk_size):
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                games.append(g)
            if not games:
                return
            x, y, v = build_chunk_tensors(games, move2idx)
            if x is not None:
                yield x, y, v, len(games)


@torch.no_grad()
def evaluate_holdout(model, holdout, device, criterion, batch_size, value_weight):
    """Policy/value metrics on games the trainer never sees."""
    if not holdout:
        return None
    model.eval()
    rows = loss_sum = policy_sum = value_sum = 0.0
    top1 = top5 = 0
    for x_chunk, y_chunk, v_chunk in holdout:
        for start in range(0, x_chunk.size(0), batch_size):
            xb = x_chunk[start:start + batch_size].to(device)
            yb = y_chunk[start:start + batch_size].to(device)
            vb = v_chunk[start:start + batch_size].to(device)
            logits, value_pred = model(xb)
            policy_loss = criterion(logits, yb)
            value_loss = nn.functional.mse_loss(value_pred.squeeze(-1), vb)
            n = xb.size(0)
            rows += n
            loss_sum += (policy_loss + value_weight * value_loss).item() * n
            policy_sum += policy_loss.item() * n
            value_sum += value_loss.item() * n
            top1 += logits.argmax(1).eq(yb).sum().item()
            top5 += logits.topk(min(5, logits.size(1)), dim=1).indices.eq(yb.unsqueeze(1)).any(1).sum().item()
    model.train()
    rows = max(rows, 1)
    return {
        'rows': int(rows),
        'loss': loss_sum / rows,
        'policy': policy_sum / rows,
        'value': value_sum / rows,
        'top1': 100.0 * top1 / rows,
        'top5': 100.0 * top5 / rows,
    }


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
    p.add_argument('--val-games', type=int, default=2000, help='Games held out for validation; 0 disables validation')
    p.add_argument('--val-fraction', type=float, default=0.02, help='Fraction of chunks routed to the held-out set')
    p.add_argument('--min-lr', type=float, default=1e-5, help='Final LR of the cosine decay')
    p.add_argument('--warmup-steps', type=int, default=500, help='Linear LR warmup steps; 0 disables warmup')
    p.add_argument('--total-steps', type=int, default=None, help='Optimizer steps the cosine decay spans (estimated from PGN size when omitted)')
    p.add_argument('--label-smoothing', type=float, default=0.05, help='Policy cross-entropy label smoothing')
    p.add_argument('--value-weight', type=float, default=1.0, help='Weight of the value loss')
    p.add_argument('--prefetch', type=int, default=3, help='Chunks parsed/encoded ahead on a loader thread; 0 disables it')
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

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.total_steps is None:
        # ~340 bytes of PGN per training position; only shapes the LR curve.
        est_rows = max(int(os.path.getsize(args.pgn) / 340), batch_size)
        args.total_steps = max(int(est_rows * args.epochs / batch_size), 1)
    args.warmup_steps = min(args.warmup_steps, max(args.total_steps // 10, 1))
    print(f"LR schedule: {args.lr} -> {args.min_lr} over ~{args.total_steps} steps (warmup {args.warmup_steps})")
    sys.stdout.flush()

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

    if amp_enabled and device.type == 'cuda':
        autocast_ctx = lambda: torch.autocast(device_type='cuda', dtype=torch.float16)
    elif amp_enabled and device.type == 'mps':
        # fp16 on MPS runs without a GradScaler, so gradients can underflow to zero.
        # bf16 has the range to be safe; note it measures slower than fp32 here.
        autocast_ctx = lambda: torch.autocast(device_type='mps', dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext

    # Use the newer GradScaler API when available to avoid deprecation warnings.
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and device.type == 'cuda')
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == 'cuda')

    # ── Hold out whole games for validation ───────────────────────────────
    # Whole games, not random positions: consecutive positions come from the same
    # game, so a positional split would leak near-duplicates and inflate accuracy.
    holdout = []
    holdout_games = 0
    if args.val_games > 0:
        print(f"Reserving up to {args.val_games} games for validation...")
        sys.stdout.flush()
        for x_c, y_c, v_c, n_g in read_game_chunks(args.pgn, args.chunk_size, move2idx):
            holdout.append((x_c, y_c, v_c))
            holdout_games += n_g
            if holdout_games >= args.val_games:
                break
        print(f"Held out {holdout_games} games / {sum(t[0].size(0) for t in holdout)} positions")
        sys.stdout.flush()

    global_step = 0
    best_val_loss = float('inf')
    best_model_path = os.path.splitext(args.model_path)[0] + '_best' + os.path.splitext(args.model_path)[1]

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

        chunk_source = read_game_chunks(
            args.pgn, args.chunk_size, move2idx, skip_games=holdout_games,
        )
        wait_start = time.perf_counter()
        for x_chunk, y_chunk, v_chunk, n_games in prefetch_iter(chunk_source, args.prefetch):
                # Parsing/encoding happens on the loader thread, so this is only
                # the time we actually sat waiting for it.
                prep_time = time.perf_counter() - wait_start
                chunk_idx += 1

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

                    lr_now = lr_at_step(global_step, args.lr, args.min_lr,
                                        args.warmup_steps, args.total_steps)
                    for group in optimizer.param_groups:
                        group['lr'] = lr_now
                    global_step += 1

                    optimizer.zero_grad(set_to_none=True)

                    with autocast_ctx():
                        output = model(xb)
                        # Support both (policy, value) and policy-only models
                        if isinstance(output, tuple):
                            logits, value_pred = output
                            policy_loss = criterion(logits, yb)
                            value_loss = nn.functional.mse_loss(value_pred.squeeze(-1), vb)
                            loss = policy_loss + args.value_weight * value_loss
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
                if chunk_idx % 10 == 0 or n_games < args.chunk_size:
                    total_avg_loss = epoch_loss / max(epoch_batches, 1)
                    chunk_avg_loss = chunk_loss / max(n_batches, 1)
                    pos_per_sec = chunk_total / max(train_time, 1e-9)
                    if args.training_only:
                        print(
                            f"Batch {chunk_idx} | "
                            f"Chunk: {n_games} games, {num_samples} positions | "
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
                            f"Chunk: {n_games} games, {num_samples} positions | "
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

                wait_start = time.perf_counter()

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

        metrics = evaluate_holdout(model, holdout, device, criterion, batch_size, args.value_weight)
        if metrics is not None:
            print(
                f"Validation | Loss: {metrics['loss']:.4f} | Policy: {metrics['policy']:.4f} | "
                f"Value: {metrics['value']:.4f} | Top-1: {metrics['top1']:.1f}% | "
                f"Top-5: {metrics['top5']:.1f}% | Rows: {metrics['rows']}"
            )
            if metrics['loss'] < best_val_loss:
                best_val_loss = metrics['loss']
                torch.save(model.state_dict(), best_model_path)
                print(f"🏆 New best validation loss — saved {best_model_path}")
            sys.stdout.flush()

        # Save after each epoch
        torch.save(model.state_dict(), args.model_path)
        print(f"💾 Model saved to {args.model_path}")
        sys.stdout.flush()

    print(f"✅ Training complete! Model saved to '{args.model_path}'")
    if best_val_loss < float('inf'):
        print(f"   Best validation loss {best_val_loss:.4f} -> {best_model_path}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()

