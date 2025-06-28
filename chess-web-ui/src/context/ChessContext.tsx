"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { getBoard, makeMove, resetBoard } from "../api/chessApi";

interface ChessContextType {
    fen: string;
    isLoading: boolean;
    makePlayerMove: (from: string, to: string) => Promise<boolean>;
    resetGame: () => void;
}

export const ChessContext = createContext<ChessContextType | undefined>(undefined);

export const ChessProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [fen, setFen] = useState<string>("");
    const [isLoading, setIsLoading] = useState(false);

    const syncBoard = async () => {
        try {
            const fenFromServer = await getBoard();
            setFen(fenFromServer ?? "");
        } catch (error) {
            console.error("Failed to sync board:", error);
        }
    };

    const makePlayerMove = async (from: string, to: string): Promise<boolean> => {
        if (isLoading) return false;
        const move = from + to;

        setIsLoading(true);
        try {
            const data = await makeMove(move);
            setFen(data.board ?? "");
            return true;
        } catch (err) {
            console.error("Move error:", err);
            return false;
        } finally {
            setIsLoading(false);
        }
    };

    const resetGame = async () => {
        setIsLoading(true);
        try {
            const fen = await resetBoard();
            setFen(fen ?? "");
        } catch (err) {
            console.error("Reset error:", err);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        syncBoard();
    }, []);

    return (
        <ChessContext.Provider value={{ fen, isLoading, makePlayerMove, resetGame }}>
            {children}
        </ChessContext.Provider>
    );
};

export const useChess = (): ChessContextType => {
    const context = useContext(ChessContext);
    if (!context) throw new Error("useChess must be used within ChessProvider");
    return context;
};