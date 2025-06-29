import torch
import os
from ChessAI import ChessAI
from Game import Game
from ChessNet import encode_fen

ai_white = ChessAI()
ai_black = ChessAI()
game = Game()

if os.path.exists("trained_model.pth"):
    state_dict = torch.load("trained_model.pth")
    ai_white.model.load_state_dict(state_dict)
    ai_black.model.load_state_dict(state_dict)
    ai_white.model.eval()
    ai_black.model.eval()
    print("✅ Model încărcat pentru ambele părți.")
else:
    print("⚠️ Nu există model anterior — începem de la zero.")

optimizer = torch.optim.Adam(ai_white.model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 1000

for epoch in range(num_epochs):
    print(f"\n🌀 === Epoch {epoch + 1}/{num_epochs} ===")
    game.reset()
    print("🔁 Joc resetat.")

    max_moves_per_game = 5
    move_count = 0

    while not game.is_game_over() and move_count < max_moves_per_game:
        move_count += 1
        print(f"[E{epoch + 1} - M{move_count}]")
        if game.board.turn:
            current_ai = ai_white
            player = "Alb"
        else:
            current_ai = ai_black
            player = "Negru"

        move = current_ai.get_best_move_from_model(game.board)

        if move:
            print(f"♟️ {player} joacă: {move}")
            game.make_move(move.uci())

            board_state = encode_fen(game.get_fen()).unsqueeze(0)
            if board_state is None:
                print("⚠️ FEN invalid — se trece peste mutare.")
                continue

            move_index = current_ai.move_to_index(move)
            target = torch.tensor([move_index])

            prediction = current_ai.model(board_state)
            loss = loss_fn(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("🏁 Joc terminat.")

torch.save(ai_white.model.state_dict(), "trained_model.pth")
print("💾 Model salvat în 'trained_model.pth'")