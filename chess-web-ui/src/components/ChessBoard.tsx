"use client";

import React, { useState } from "react";
import styles from "./ChessBoard.module.css";
import { useChess } from "../context/ChessContext";

const pieceSymbols: Record<string, string> = {
    P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
    p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
};

function parseFEN(fen: string): string[][] {
    const board: string[][] = [];
    const rows = (fen ?? "").split(" ")[0].split("/");

    for (const row of rows) {
        const cells: string[] = [];
        for (const char of row) {
            if (isNaN(parseInt(char))) {
                cells.push(char);
            } else {
                cells.push(...Array(parseInt(char)).fill(""));
            }
        }
        board.push(cells);
    }

    return board;
}

const ChessBoard: React.FC = () => {
    const { fen, makePlayerMove, isLoading } = useChess();
    const board = fen ? parseFEN(fen) : Array(8).fill(null).map(() => Array(8).fill(""));
    const [selected, setSelected] = useState<{ row: number; col: number } | null>(null);

    const getSquareName = (row: number, col: number) => {
        const file = "abcdefgh"[col];
        const rank = 8 - row;
        return `${file}${rank}`;
    };

    const handleClick = async (row: number, col: number) => {
        if (isLoading) return;
        if (!selected) {
            if (board[row][col] !== "") setSelected({ row, col });
        } else {
            const from = getSquareName(selected.row, selected.col);
            const to = getSquareName(row, col);
            const success = await makePlayerMove(from, to);
            if (success) setSelected(null);
        }
    };

    return (
        <div className={styles.board}>
            {board.map((row, rowIndex) => (
                <div className={styles.row} key={rowIndex}>
                    {row.map((cell, colIndex) => {
                        const square = getSquareName(rowIndex, colIndex);
                        const selectedClass =
                            selected?.row === rowIndex && selected?.col === colIndex
                                ? styles.selected
                                : "";

                        return (
                            <div
                                key={square}
                                className={`${styles.cell} ${selectedClass}`}
                                onClick={() => handleClick(rowIndex, colIndex)}
                            >
                                {pieceSymbols[cell] || ""}
                            </div>
                        );
                    })}
                </div>
            ))}
        </div>
    );
};

export default ChessBoard;