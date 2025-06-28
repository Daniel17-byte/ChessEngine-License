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
        self.out = nn.Linear(128, 1)  # Output: evaluation score of the position

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


def encode_fen(fen: str) -> torch.Tensor:
    # Placeholder encoder: convert a FEN to a 773-bit tensor (example format)
    # You should replace this with proper one-hot encoding of FEN board state
    # For now, returns a random tensor for testing
    return torch.randn(773)


def evaluate_position(model, fen: str) -> float:
    model.eval()
    with torch.no_grad():
        input_tensor = encode_fen(fen)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        score = model(input_tensor)
        return score.item()


if __name__ == "__main__":
    model = ChessNet()
    sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print("Position score:", evaluate_position(model, sample_fen))