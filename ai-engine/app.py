from flask import Flask, request, jsonify
from flask_cors import CORS
from Game import Game
import chess

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
game = Game()

@app.route('/api/game/get_board', methods=['GET'])
def get_board():
    return jsonify({
        'board': game.get_board_fen(),
        'turn': 'white' if game.board.turn == chess.WHITE else 'black',
        'is_check': game.board.is_check(),
        'is_checkmate': game.board.is_checkmate(),
        'is_stalemate': game.board.is_stalemate(),
        'is_insufficient_material': game.board.is_insufficient_material()
    })

@app.route('/api/game/make_move', methods=['POST'])
def make_move():
    print("\n==============================")
    print("🔵 [API] POST /make_move")

    data = request.get_json()
    move = data.get('move')
    if move and len(move) == 4:
        from_square = chess.parse_square(move[:2])
    to_square = chess.parse_square(move[2:])
    piece = game.board.piece_at(from_square)

    if piece and piece.piece_type == chess.PAWN:
        rank_from = chess.square_rank(from_square)
        rank_to = chess.square_rank(to_square)
        # White promotes on rank 7->8, black on 2->1
        if (piece.color == chess.WHITE and rank_from == 6 and rank_to == 7) or \
                (piece.color == chess.BLACK and rank_from == 1 and rank_to == 0):
            move += 'q'


    print(f"➡️  Player move received: {move}")

    try:
        chess_move = chess.Move.from_uci(move)
    except:
        print("❌ Invalid move format.")
        return jsonify({'error': 'Invalid move format', 'board': game.get_board_fen()}), 400

    if chess_move not in game.board.legal_moves:
        print("❌ Move is not legal.")
        return jsonify({'error': 'Illegal move', 'board': game.get_board_fen()}), 400

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

        if not game.is_game_over() and game.board.turn == chess.BLACK:
            ai_move = game.ai_move()
            print(f"🤖 AI moved: {ai_move}")
            print("==============================")
            return jsonify({
                'board': game.get_board_fen(),
                'result': 'Move successful',
                'ai_move': ai_move,
                'turn': 'white' if game.board.turn == chess.WHITE else 'black',
                'is_check': game.board.is_check(),
                'is_checkmate': game.board.is_checkmate(),
                'is_stalemate': game.board.is_stalemate(),
                'is_insufficient_material': game.board.is_insufficient_material()
            })
        else:
            print("==============================")
            return jsonify({
                'board': game.get_board_fen(),
                'result': 'Game over',
                'turn': 'white' if game.board.turn == chess.WHITE else 'black',
                'is_check': game.board.is_check(),
                'is_checkmate': game.board.is_checkmate(),
                'is_stalemate': game.board.is_stalemate(),
                'is_insufficient_material': game.board.is_insufficient_material()
            })
    except Exception as e:
        print(f"🔥 Exception occurred: {e}")
        print("==============================")
        return jsonify({'error': str(e)}), 500

@app.route('/api/game/reset', methods=['POST'])
def reset():
    game.reset()
    return jsonify({
        'message': 'Board reset',
        'board': game.get_board_fen(),
        'turn': 'white' if game.board.turn == chess.WHITE else 'black',
        'is_check': game.board.is_check(),
        'is_checkmate': game.board.is_checkmate(),
        'is_stalemate': game.board.is_stalemate(),
        'is_insufficient_material': game.board.is_insufficient_material()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5050, host='0.0.0.0')