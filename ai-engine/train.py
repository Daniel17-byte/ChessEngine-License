import torch
import os
from ChessAI import ChessAI
from Game import Game
from ChessNet import encode_fen
import json
from collections import Counter
import random

def load_fens_from_files(filepath="generated_games.json"):
    fens = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            fens = json.load(f)
    return fens

with open("move_mapping.json") as f:
    idx_to_move = json.load(f)

move_to_idx = {uci: i for i, uci in enumerate(idx_to_move)}

ai_white = ChessAI()
ai_black = ChessAI()
game = Game()
stats = Counter()

if os.path.exists("trained_model.pth"):
    state_dict = torch.load("trained_model.pth")
    ai_white.model.load_state_dict(state_dict)
    ai_black.model.load_state_dict(state_dict)
    ai_white.model.eval()
    ai_black.model.eval()
    print("✅ Model încărcat.")
else:
    print("⚠️ Fără model anterior — se pornește de la zero.")

optimizer = torch.optim.Adam(ai_white.model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 500
max_moves_per_game = 30

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
                history.append((board_state, move_index, game.board.turn, step_reward))

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
        reward = {True: 0.0, False: 0.0}

    total_loss = 0.0
    total_scaled_reward = 0.0

    # Învățare: aplicăm loss pe fiecare mutare cu reward ca "greutate"
    for state, move_index, was_white, step_reward in history:
        prediction = ai_white.model(state)
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
    print(f"🎯 Rezultat: {result} | Mutări: {move_count}")
    total_games = stats['1-0'] + stats['0-1'] + stats['1/2-1/2'] + stats['*']

    if (epoch + 1) % 50 == 0:
        print(f"🏁 WHITE {stats['1-0']} | BLACK {stats['0-1']} | DRAW {stats['1/2-1/2']} | Total: {total_games} ")
        torch.save(ai_white.model.state_dict(), "trained_model.pth")

torch.save(ai_white.model.state_dict(), "trained_model.pth")
print("💾 Model salvat în 'trained_model.pth'")