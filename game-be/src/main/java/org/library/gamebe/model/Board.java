package org.library.gamebe.model;

import lombok.*;
import org.library.gamebe.util.Color;
import org.library.gamebe.util.PieceType;

import lombok.*;

@Data
public class Board {
    private Piece[][] grid = new Piece[8][8];

    public Board() {
        init();
    }

    public void init() {
        grid = new Piece[8][8];

        for (int i = 0; i < 8; i++) {
            grid[1][i] = new Piece(PieceType.PAWN, Color.BLACK);
            grid[6][i] = new Piece(PieceType.PAWN, Color.WHITE);
        }

        grid[0][0] = new Piece(PieceType.ROOK, Color.BLACK);
        grid[0][7] = new Piece(PieceType.ROOK, Color.BLACK);
        grid[7][0] = new Piece(PieceType.ROOK, Color.WHITE);
        grid[7][7] = new Piece(PieceType.ROOK, Color.WHITE);

        grid[0][1] = new Piece(PieceType.KNIGHT, Color.BLACK);
        grid[0][6] = new Piece(PieceType.KNIGHT, Color.BLACK);
        grid[7][1] = new Piece(PieceType.KNIGHT, Color.WHITE);
        grid[7][6] = new Piece(PieceType.KNIGHT, Color.WHITE);

        grid[0][2] = new Piece(PieceType.BISHOP, Color.BLACK);
        grid[0][5] = new Piece(PieceType.BISHOP, Color.BLACK);
        grid[7][2] = new Piece(PieceType.BISHOP, Color.WHITE);
        grid[7][5] = new Piece(PieceType.BISHOP, Color.WHITE);

        grid[0][3] = new Piece(PieceType.QUEEN, Color.BLACK);
        grid[0][4] = new Piece(PieceType.KING, Color.BLACK);
        grid[7][3] = new Piece(PieceType.QUEEN, Color.WHITE);
        grid[7][4] = new Piece(PieceType.KING, Color.WHITE);
    }

    public Piece getPiece(Position pos) {
        return grid[pos.getRow()][pos.getCol()];
    }

    public void movePiece(Move move) {
        Piece piece = getPiece(move.getFrom());
        grid[move.getTo().getRow()][move.getTo().getCol()] = piece;
        grid[move.getFrom().getRow()][move.getFrom().getCol()] = null;
    }

    /**
     * Returns the position of the king for the given color
     */
    public Position findKingPosition(Color color) {
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                Piece piece = grid[row][col];
                if (piece != null &&
                        piece.getType() == PieceType.KING &&
                        piece.getColor() == color) {
                    return new Position(row, col);
                }
            }
        }
        return null;
    }

    /**
     * Deep clone of the board (pieces are copied by value)
     */
    public Board cloneBoard() {
        Board clone = new Board();
        clone.grid = new Piece[8][8];
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                Piece original = this.grid[row][col];
                if (original != null) {
                    clone.grid[row][col] = new Piece(
                            original.getType(),
                            original.getColor()
                    );
                }
            }
        }
        return clone;
    }
}