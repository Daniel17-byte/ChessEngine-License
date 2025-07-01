import json
with open("move_mapping.json") as f:
    MOVE_MAPPING = json.load(f)
OUTPUT_SIZE = len(MOVE_MAPPING)

import torch
import torch.nn as nn
import torch.nn.functional as fun

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
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = fun.relu(self.fc3(x))
        x = fun.relu(self.fc4(x))
        return self.out(x)


def encode_fen(fen: str) -> torch.Tensor:
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    tensor = torch.zeros(773)
    parts = fen.split()
    board_part = parts[0]
    turn_part = parts[1]
    castling_part = parts[2]

    # Encode board (64 squares x 12 piece types = 768)
    rows = board_part.split('/')
    for i, row in enumerate(rows):
        col = 0
        for c in row:
            if c.isdigit():
                col += int(c)
            else:
                index = piece_to_index.get(c)
                if index is not None:
                    square_index = i * 8 + col
                    tensor[square_index * 12 + index] = 1
                    col += 1

    # Encode turn (1 bit)
    tensor[768] = 1 if turn_part == 'w' else 0

    # Encode castling rights (4 bits: KQkq)
    tensor[769] = 1 if 'K' in castling_part else 0
    tensor[770] = 1 if 'Q' in castling_part else 0
    tensor[771] = 1 if 'k' in castling_part else 0
    tensor[772] = 1 if 'q' in castling_part else 0

    return tensor


def evaluate_position(model, fen: str) -> float:
    model.eval()
    with torch.no_grad():
        input_tensor = encode_fen(fen).unsqueeze(0)  # Ensure correct input encoding
        score = model(input_tensor)
        return score.item()