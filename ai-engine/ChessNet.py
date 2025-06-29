import json
with open("move_mapping.json") as f:
    MOVE_MAPPING = json.load(f)
OUTPUT_SIZE = len(MOVE_MAPPING)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: FEN encoded as 773-bit vector (typical when one-hot encoding board state)
        self.fc1 = nn.Linear(773, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        # Output: logits over all possible legal moves (up to 4672 options)
        self.out = nn.Linear(64, OUTPUT_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)


def encode_fen(fen: str) -> torch.Tensor:
    # Dummy encoder: returns zero tensor with correct shape
    return torch.zeros(773)


def evaluate_position(model, fen: str) -> float:
    model.eval()
    with torch.no_grad():
        input_tensor = encode_fen(fen).unsqueeze(0)  # Ensure correct input encoding
        score = model(input_tensor)
        return score.item()


if __name__ == "__main__":
    model = ChessNet()
    sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print("Position score:", evaluate_position(model, sample_fen))