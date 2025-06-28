import torch
from ChessAI import ChessAI
from Game import Game

# Inițializăm AI-ul și jocul
ai = ChessAI()
game = Game()

from ChessNet import encode_fen

# Setăm optimizatorul și funcția de pierdere
optimizer = torch.optim.Adam(ai.model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Număr de epoci de antrenare
num_epochs = 1000

for epoch in range(num_epochs):
    # Resetăm jocul pentru fiecare episod
    game.reset()

    while True:
        board_state = encode_fen(game.get_fen()).unsqueeze(0)
        if board_state is None:
            break  # Ieșim dacă nu avem stare validă

        prediction = ai.model(board_state)  # [1, num_moves]

        move = game.get_last_move_uci()
        if move is None:
            break  # Nu există mutare anterioară, ieșim din buclă

        move_index = ai.move_to_index(move)
        target = torch.tensor([move_index])

        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

        if game.is_game_over():
            break

        # Așteptăm mutarea utilizatorului pentru a continua antrenamentul în timp real
        game.wait_for_user_move()

# Salvăm modelul antrenat
torch.save(ai.model.state_dict(), "trained_model.pth")
print("Modelul a fost salvat în 'trained_model.pth'")