import chess.pgn
import json

def extract_fens_from_pgn(pgn_path, output_json):
    fens = []
    with open(pgn_path, encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                fens.append(board.fen())
                if len(fens) >= 30000:
                    break
            if len(fens) >= 30000:
                break

    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(fens[20000:30000], json_file, indent=2)

# Apel funcție principală
extract_fens_from_pgn("lichess_db.pgn", "generated_games.json")