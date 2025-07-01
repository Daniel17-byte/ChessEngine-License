import torch
import os
from ChessAI import ChessAI
from Game import Game
from ChessNet import encode_fen
import json
from collections import Counter
import random
import chess

def load_fens_from_files(filepath="generated_endgames.json"):
    fens = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            fens = json.load(f)
    return fens

with open("move_mapping.json") as f:
    idx_to_move = json.load(f)

move_to_idx = {uci: i for i, uci in enumerate(idx_to_move)}

ai_white = ChessAI(is_white=True)
ai_black = ChessAI(is_white=False)
game = Game(ai_white, ai_black)
stats = Counter()

optimizer_white = torch.optim.Adam(ai_white.model.parameters(), lr=0.001)
optimizer_black = torch.optim.Adam(ai_black.model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 500
max_moves_per_game = 16

fen_positions = load_fens_from_files()

for epoch in range(num_epochs):
    if fen_positions:
        fen = random.choice(fen_positions)
        game.reset_from_fen(fen)
    else:
        game.reset()
    # game.reset()
    history = []
    move_count = 0

    while not game.is_game_over() and move_count < max_moves_per_game:
        move_count += 1
        current_ai = ai_white if game.board.turn else ai_black

        move = current_ai.select_move(game.board)

        if move:
            board_state = encode_fen(game.get_fen()).unsqueeze(0)
            move_index = move_to_idx.get(move.uci())

            if board_state is not None and move_index is not None:
                _, move_info = game.make_move(move.uci())
                step_reward = move_info.get("reward", 0.0)
                was_white = game.board.turn == chess.BLACK
                history.append((board_state, move_index, was_white, step_reward))

    result = game.get_result()
    if result == '1-0':
        base = 15.0
        if move_count < 10:
            base *= 1.5
        elif move_count < 20:
            base *= 1.2
        elif move_count < 30:
            base *= 1.0
        else:
            base *= 0.8
        reward = {True: base, False: -base}
    elif result == '0-1':
        base = 15.0
        if move_count < 10:
            base *= 1.5
        elif move_count < 20:
            base *= 1.2
        elif move_count < 30:
            base *= 1.0
        else:
            base *= 0.8
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
            reward = {True: 0.0, False: 0.0}
        else:
            ratio = 15.0 / total
            reward = {
                True: (white_score - black_score) * ratio,
                False: (black_score - white_score) * ratio
            }

    total_loss = 0.0
    total_scaled_reward = 0.0

    # Învățare: aplicăm loss pe fiecare mutare cu reward ca "greutate"
    for state, move_index, was_white, step_reward in history:
        model = ai_white.model if was_white else ai_black.model
        optimizer = optimizer_white if was_white else optimizer_black

        prediction = model(state)
        target = torch.tensor([move_index])

        total_reward = reward[was_white] + step_reward
        ce_loss = loss_fn(prediction, target)
        scaled_reward = torch.clamp(torch.tensor(total_reward), -1.0, 1.0)
        raw_loss = ce_loss.item()
        scaled_loss = ce_loss * scaled_reward
        # print(f"🔻 Loss: {loss.item():.2f} | Reward: {total_reward:.2f} | {'Alb' if was_white else 'Negru'}")

        optimizer.zero_grad()
        scaled_loss.backward()
        optimizer.step()

        total_loss += raw_loss
        total_scaled_reward += total_reward

    # print(f"📉 Loss total (pe joc): {total_loss:.4f} | 🎁 Reward total: {total_scaled_reward:.2f}")

    stats[result] += 1
    print(f"🎯 Rezultat: {result} | Mutări: {move_count} | 🏆 Reward: Alb = {reward[True]:.2f}, Negru = {reward[False]:.2f}")
    total_games = stats['1-0'] + stats['0-1'] + stats['1/2-1/2'] + stats['*']

    if (epoch + 1) % 50 == 0:
        print(f"🏁 WHITE {stats['1-0']} | BLACK {stats['0-1']} | DRAW {stats['1/2-1/2']} | Total: {total_games} ")
        torch.save(ai_white.model.state_dict(), "trained_model_white.pth")
        torch.save(ai_black.model.state_dict(), "trained_model_black.pth")

torch.save(ai_white.model.state_dict(), "trained_model_white.pth")
torch.save(ai_black.model.state_dict(), "trained_model_black.pth")
print("💾 Modele salvate în 'trained_model_white.pth' și 'trained_model_black.pth'")