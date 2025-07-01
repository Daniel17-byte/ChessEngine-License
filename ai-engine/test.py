
import torch
from ChessNet import ChessNet, encode_fen
import os

# Verificare încărcare model
model_white_path = "trained_model_white.pth"
model_black_path = "trained_model_black.pth"

model_white = ChessNet()
model_black = ChessNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_white.to(device)
model_black.to(device)

if os.path.exists(model_white_path):
    model_white.load_state_dict(torch.load(model_white_path, map_location=device))
    model_white.eval()
    print(f"✅ Model alb încărcat cu succes.")
else:
    print("❌ Model alb NU există.")

if os.path.exists(model_black_path):
    model_black.load_state_dict(torch.load(model_black_path, map_location=device))
    model_black.eval()
    print(f"✅ Model negru încărcat cu succes.")
else:
    print("❌ Model negru NU există.")

# Afișare sumă ponderi inițiale pentru model alb și negru
white_params_sum = sum(p.sum().item() for p in model_white.parameters())
black_params_sum = sum(p.sum().item() for p in model_black.parameters())

print("📊 Sumă ponderi model alb:", white_params_sum)
print("📊 Sumă ponderi model negru:", black_params_sum)

# Verificare encoding fen pentru poziția inițială
initial_fen = "rn1qkbnr/pp3ppp/4p3/2pp4/3P1B2/2P5/PP2PPPP/RN1QKBNR w KQkq - 0 5"
encoded_tensor = encode_fen(initial_fen)
print("♟️ Valori unice în encoding-ul poziției:", torch.unique(encoded_tensor))

# Normalizare tensor (verificare numerică stabilă)
normalized_tensor = (encoded_tensor - encoded_tensor.mean()) / (encoded_tensor.std() + 1e-5)
print("📐 Valori unice după normalizare:", torch.unique(normalized_tensor))
