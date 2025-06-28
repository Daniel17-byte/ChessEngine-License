from flask import Flask, request, jsonify
from flask_cors import CORS
from Game import Game
import chess

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
game = Game(vs_ai=True)

@app.route('/api/game/get_board', methods=['GET'])
def get_board():
    return jsonify({'board': game.get_board_fen()})

@app.route('/api/game/make_move', methods=['POST'])
def make_move():
    print("\n==============================")
    print("🔵 [API] POST /make_move")

    data = request.get_json()
    move = data.get('move')
    print(f"➡️  Player move received: {move}")

    try:
        chess_move = chess.Move.from_uci(move)
    except:
        print("❌ Invalid move format.")
        return jsonify({'error': 'Invalid move format'}), 400

    if chess_move not in game.board.legal_moves:
        print("❌ Move is not legal.")
        return jsonify({'error': 'Illegal move'}), 400

    if not move:
        print("❌ No move provided.")
        print("==============================")
        return jsonify({'error': 'No move provided'}), 400
    try:
        success = game.make_move(move)
        print(f"✅ Player move valid: {success}")

        if not success:
            print("❌ Invalid move attempted.")
            print("==============================")
            return jsonify({'error': 'Invalid move', 'board': game.get_board_fen()}), 400

        print("♟️  Player move applied.")
        print("📥  Current board FEN after player move:", game.get_board_fen())

        if not game.is_game_over() and game.board.turn == chess.BLACK:
            print("🤖 AI is calculating move...")
            ai_move = game.ai_move()
            print(f"🤖 AI moved: {ai_move}")
            print("📥  Current board FEN after AI move:", game.get_board_fen())
            print("==============================")
            return jsonify({
                'board': game.get_board_fen(),
                'result': 'Move successful',
                'ai_move': ai_move
            })
        else:
            print("🏁 Game is over after this move.")
            print("==============================")
            return jsonify({'board': game.get_board_fen(), 'result': 'Game over'})
    except Exception as e:
        print(f"🔥 Exception occurred: {e}")
        print("==============================")
        return jsonify({'error': str(e)}), 500

@app.route('/api/game/reset', methods=['POST'])
def reset():
    game.reset()
    return jsonify({'message': 'Board reset', 'board': game.get_board_fen()})

if __name__ == '__main__':
    app.run(debug=True, port=5050, host='0.0.0.0')