import torch
import os

from fsspec.registry import default

from ChessAI import ChessAI
from Game import Game
import json
from collections import Counter
import random
import chess

from ArchiveAlpha import encode_board

# Load move mapping for training
with open('move_mapping.json', 'r', encoding='utf-8') as f:
    move_list = json.load(f)
move_to_idx = {m: i for i, m in enumerate(move_list)}

# def load_fens_from_files(filepath="generated_games.json"):
#     fens = []
#     if os.path.exists(filepath):
#         with open(filepath, "r") as file:
#             fens = json.load(file)
#     return fens

ai_white = ChessAI(is_white=True, default_strategy="model")
ai_black = ChessAI(is_white=False, default_strategy="model")
game = Game(ai_white, ai_black)
stats = Counter()

optimizer_white = torch.optim.Adam(ai_white.model.parameters(), lr=0.001)
optimizer_black = torch.optim.Adam(ai_black.model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 1000
max_moves_per_game = 40

def compute_base(move_count_):
    base_ = 50.0
    if move_count_ < 10:
        base_ *= 2.6
    elif move_count_ < 20:
        base_ *= 2.4
    elif move_count_ < 30:
        base_ *= 2.2
    elif move_count_ < 40:
        base_ *= 2.0
    elif move_count_ < 50:
        base_ *= 1.8
    elif move_count_ < 60:
        base_ *= 1.6
    elif move_count_ < 70:
        base_ *= 1.4
    elif move_count_ < 80:
        base_ *= 1.2
    elif move_count_ < 90:
        base_ *= 1.0
    else:
        base_ *= 0.8
    return base_

# fen_positions = load_fens_from_files()

def get_weight_sum(model_):
    return sum(p.sum().item() for p in model_.parameters())

prev_white = get_weight_sum(ai_white.model)
prev_black = get_weight_sum(ai_black.model)

for epoch in range(num_epochs):
    # if fen_positions:
    #     fen = random.choice(fen_positions)
    #     game.reset_from_fen(fen)
    # else:
    #     game.reset()
    game.reset()
    history = []
    move_count = 0

    while not game.is_game_over() and move_count < max_moves_per_game:
        move_count += 1
        current_ai = ai_white if game.board.turn else ai_black

        move = current_ai.select_move(game.board)

        if move:
            board_state = encode_board(chess.Board(game.get_fen())).unsqueeze(0)
            move_index = move_to_idx.get(move.uci())

            if board_state is not None and move_index is not None:
                _, move_info = game.make_move(move.uci())
                step_reward = move_info.get("reward", 0.0)
                moved_by_white = game.board.turn == chess.BLACK
                history.append((board_state, move_index, moved_by_white, step_reward))

    result = game.get_result()
    if result == '1-0':
        base = compute_base(move_count)
        reward = {True: base, False: -base}
    elif result == '0-1':
        base = compute_base(move_count)
        reward = {True: -base, False: base}
    else:
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9
        }
        white_score = 0
        black_score = 0
        for square, piece in game.board.piece_map().items():
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_score += value
            else:
                black_score += value
        total = white_score + black_score
        if total == 0:
            reward = {True: -10.0, False: -10.0}
        else:
            ratio = 15.0 / total
            reward = {
                True: (white_score - black_score) * ratio,
                False: (black_score - white_score) * ratio
            }

    total_loss = 0.0
    total_scaled_reward = 0.0

    # Învățare: aplicăm loss pe fiecare mutare cu reward ca "greutate"
    for state, move_index, moved_by_white, step_reward in history:
        model = ai_white.model if moved_by_white else ai_black.model

        optimizer = optimizer_white if moved_by_white else optimizer_black

        prediction = model(state)
        target = torch.tensor([move_index])

        total_reward = (reward[moved_by_white] + step_reward) / move_count
        ce_loss = loss_fn(prediction, target)
        scaled_reward = max(-1.0, min(1.0, total_reward))
        raw_loss = ce_loss.item()
        scaled_loss = ce_loss * scaled_reward

        optimizer.zero_grad()
        scaled_loss.backward()
        optimizer.step()

        total_loss += raw_loss
        total_scaled_reward += total_reward

    stats[result] += 1
    print(f"🎯 Rezultat: {result} | Mutări: {move_count} | 🏆 Reward: Alb = {reward[True]:.2f}, Negru = {reward[False]:.2f}")

    if (epoch + 1) % 50 == 0:
        print(f"🏁 WHITE {stats['1-0']} | BLACK {stats['0-1']} | DRAW {stats['1/2-1/2']} | Total: {stats['*']} ")
        torch.save(ai_white.model.state_dict(), "trained_model_white.pth")
        torch.save(ai_black.model.state_dict(), "trained_model_black.pth")
        # curr_white = get_weight_sum(ai_white.model)
        # curr_black = get_weight_sum(ai_black.model)
        # print(f"Δ alb: {curr_white - prev_white:.6f}, Δ negru: {curr_black - prev_black:.6f}")
        # prev_white = curr_white
        # prev_black = curr_black

curr_white = get_weight_sum(ai_white.model)
curr_black = get_weight_sum(ai_black.model)
torch.save(ai_white.model.state_dict(), "trained_model_white.pth")
torch.save(ai_black.model.state_dict(), "trained_model_black.pth")
print("💾 Modele salvate în 'trained_model_white.pth' și 'trained_model_black.pth'")