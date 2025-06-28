package org.library.gamebe.service;

import lombok.Getter;
import org.library.gamebe.dto.MoveResult;
import org.library.gamebe.model.Position;
import org.library.gamebe.util.MoveStatus;
import org.library.gamebe.model.Board;
import org.library.gamebe.util.Color;
import org.library.gamebe.model.Move;
import org.library.gamebe.model.Piece;
import org.springframework.stereotype.Service;

@Getter
@Service
public class GameService {
    private final Board board;
    private Color currentTurn = Color.WHITE;

    public GameService() {
        this.board = new Board();
    }

    public GameService(Board cloneBoard) {
        this.board = cloneBoard;
    }

    public MoveResult processMove(Move move) {
        Piece piece = board.getPiece(move.getFrom());

        if (piece == null) {
            return new MoveResult(MoveStatus.INVALID, "No piece at source position.");
        }

        if (piece.getColor() != currentTurn) {
            return new MoveResult(MoveStatus.INVALID, "It's " + currentTurn + "'s turn.");
        }

        if (!isMoveLegal(piece, move)) {
            return new MoveResult(MoveStatus.INVALID, "Illegal move for this piece.");
        }

        board.movePiece(move);
        toggleTurn();

        if (isCheckmate(currentTurn)) {
            return new MoveResult(MoveStatus.CHECKMATE, "Checkmate! " + piece.getColor() + " wins.");
        }

        if (isDraw()) {
            return new MoveResult(MoveStatus.DRAW, "Draw detected.");
        }

        if (isCheck(currentTurn)) {
            return new MoveResult(MoveStatus.CHECK, piece.getColor() + " puts opponent in check.");
        }

        return new MoveResult(MoveStatus.VALID, "Move accepted.");
    }

    public void resetGame() {
        board.init();
        currentTurn = Color.WHITE;
    }

    private void toggleTurn() {
        currentTurn = (currentTurn == Color.WHITE) ? Color.BLACK : Color.WHITE;
    }

    private boolean isMoveLegal(Piece piece, Move move) {
        int fromRow = move.getFrom().getRow();
        int fromCol = move.getFrom().getCol();
        int toRow = move.getTo().getRow();
        int toCol = move.getTo().getCol();

        int dRow = toRow - fromRow;
        int dCol = toCol - fromCol;

        Piece target = board.getPiece(move.getTo());
        if (target != null && target.getColor() == piece.getColor()) {
            return false;
        }

        return switch (piece.getType()) {
            case PAWN -> validatePawnMove(piece, fromRow, fromCol, toRow, toCol, target);
            case ROOK -> dRow == 0 || dCol == 0;
            case KNIGHT -> Math.abs(dRow) == 2 && Math.abs(dCol) == 1 ||
                    Math.abs(dRow) == 1 && Math.abs(dCol) == 2;
            case BISHOP -> Math.abs(dRow) == Math.abs(dCol);
            case QUEEN -> dRow == 0 || dCol == 0 || Math.abs(dRow) == Math.abs(dCol);
            case KING -> Math.abs(dRow) <= 1 && Math.abs(dCol) <= 1;
        };
    }

    private boolean validatePawnMove(Piece piece, int fromRow, int fromCol, int toRow, int toCol, Piece target) {
        int direction = piece.getColor() == Color.WHITE ? -1 : 1;
        int startRow = piece.getColor() == Color.WHITE ? 6 : 1;

        // Mutare simplă înainte
        if (fromCol == toCol && target == null) {
            if (toRow == fromRow + direction) return true;
            if (fromRow == startRow && toRow == fromRow + 2 * direction) return true;
        }

        // Captură pe diagonală
        if (Math.abs(fromCol - toCol) == 1 && toRow == fromRow + direction && target != null) {
            return true;
        }

        return false;
    }

    private boolean isCheck(Color color) {
        Position kingPos = board.findKingPosition(color);
        if (kingPos == null) return false;

        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                Piece attacker = board.getPiece(new Position(row, col));
                if (attacker != null && attacker.getColor() != color) {
                    Move mockMove = new Move(new Position(row, col), kingPos);
                    if (isMoveLegal(attacker, mockMove)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private boolean isCheckmate(Color color) {
        if (!isCheck(color))
            return false;

        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                Piece piece = board.getPiece(new Position(row, col));
                if (piece != null && piece.getColor() == color) {
                    for (int r = 0; r < 8; r++) {
                        for (int c = 0; c < 8; c++) {
                            Move candidate = new Move(new Position(row, col), new Position(r, c));
                            if (isMoveLegal(piece, candidate)) {
                                // Simulăm mutarea
                                Board clone = board.cloneBoard();
                                clone.movePiece(candidate);
                                if (!new GameService(clone).isCheck(color)) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

    private boolean isDraw() {
        if (isCheck(currentTurn)) return false;

        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                Piece piece = board.getPiece(new Position(row, col));
                if (piece != null && piece.getColor() == currentTurn) {
                    for (int r = 0; r < 8; r++) {
                        for (int c = 0; c < 8; c++) {
                            Move candidate = new Move(new Position(row, col), new Position(r, c));
                            if (isMoveLegal(piece, candidate)) {
                                Board clone = board.cloneBoard();
                                clone.movePiece(candidate);
                                if (!new GameService(clone).isCheck(currentTurn)) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
}