import torch
import os
from ChessAI import ChessAI
from Game import Game
from ChessNet import encode_fen
import json
from collections import Counter

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

num_epochs = 1000
max_moves_per_game = 20

for epoch in range(num_epochs):
    print(f"\n🌀 === Epoch {epoch + 1}/{num_epochs} ===")
    game.reset()

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

    # Recompensă finală: +1 pentru câștigător, -1 pentru pierzător, 0 pentru remiză
    result = game.get_result()
    if result == '1-0':
        reward = {True: 1.0, False: -1.0}
    elif result == '0-1':
        reward = {True: -1.0, False: 1.0}
    else:
        reward = {True: 0.0, False: 0.0}

    # Învățare: aplicăm loss pe fiecare mutare cu reward ca "greutate"
    for state, move_index, was_white, step_reward in history:
        prediction = ai_white.model(state)
        target = torch.tensor([move_index])

        total_reward = reward[was_white] + step_reward
        loss = loss_fn(prediction, target) * total_reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    stats[result] += 1
    # print(f"🎯 Rezultat: {result} | Mutări: {move_count}")
    print(f"📊 Statistici: {dict(stats)}")

torch.save(ai_white.model.state_dict(), "trained_model.pth")
print("💾 Model salvat în 'trained_model.pth'")