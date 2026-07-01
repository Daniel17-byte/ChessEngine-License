"use client";

import React, { createContext, useContext, useState, useCallback, useEffect } from "react";
import { getBoard, makeMove, resetBoard, setPlayerColor as setBackendPlayerColor } from "../api/chessApi";
import { recordWin, recordLoss, recordDraw } from "../api/statsApi";

export enum GameType {
    AI = "ai",
    PVP = "pvp",
}

export enum PlayerColor {
    WHITE = "white",
    BLACK = "black",
}

export enum AiOpponent {
    DANIBOT = "model",
    STOCKFISH = "stockfish",
}

export type GameResult = "win" | "loss" | "draw" | null;

interface ChessContextType {
    fen: string;
    setFen: (fen: string) => void;
    isLoading: boolean;
    makePlayerMove: (from: string, to: string, promotion?: string) => Promise<boolean>;
    resetGame: () => void;
    lastAiMove: string | null;
    setLastAiMove: (move: string | null) => void;
    isGameOver: boolean;
    gameResult: GameResult;
    dismissGameResult: () => void;
    boardFlipped: boolean;
    toggleBoardFlip: () => void;
    gameType: GameType;
    playerColor: PlayerColor;
    aiOpponent: AiOpponent;
    pvpMatchId: number | null;
    opponentId: string | null;
    setGameSettings: (
        gameType: GameType,
        playerColor: PlayerColor,
        aiOpponent?: AiOpponent,
        multiplayerOptions?: { matchId?: number; opponentId?: string | null }
    ) => void;
    moveCount: number;
    gameStarted: boolean;
    setGameStarted: (started: boolean) => void;
    isCheckmate: boolean;
    isStalemate: boolean;
    isCheck: boolean;
}

export const ChessContext = createContext<ChessContextType | undefined>(undefined);

export const ChessProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [fen, setFen] = useState<string>("");
    const [isLoading, setIsLoading] = useState(false);
    const [lastAiMove, setLastAiMove] = useState<string | null>(null);
    const [isGameOver, setIsGameOver] = useState(false);
    const [gameResult, setGameResult] = useState<GameResult>(null);
    const [boardFlipped, setBoardFlipped] = useState(false);
    const [gameType, setGameType] = useState<GameType>(GameType.AI);
    const [playerColor, setPlayerColor] = useState<PlayerColor>(PlayerColor.WHITE);
    const [aiOpponent, setAiOpponent] = useState<AiOpponent>(AiOpponent.DANIBOT);
    const [pvpMatchId, setPvpMatchId] = useState<number | null>(null);
    const [opponentId, setOpponentId] = useState<string | null>(null);
    const [moveCount, setMoveCount] = useState(0);
    const [gameStarted, setGameStarted] = useState(false);
    const [isCheckmate, setIsCheckmate] = useState(false);
    const [isStalemate, setIsStalemate] = useState(false);
    const [isCheck, setIsCheck] = useState(false);

    const getUserId = (): string | null => {
        try {
            const stored = sessionStorage.getItem("user");
            if (stored) {
                const user = JSON.parse(stored);
                return user.username || user.uuid || null;
            }
            const guestId = localStorage.getItem("guest_matchmaking_player_id");
            if (guestId) {
                return guestId;
            }
        } catch { /* ignore */ }
        return null;
    };

    const recordGameResult = useCallback(async (result: GameResult) => {
        const userId = getUserId();
        if (!userId || !result) return;
        try {
            if (result === "win") await recordWin(userId);
            else if (result === "loss") await recordLoss(userId);
            else if (result === "draw") await recordDraw(userId);
        } catch (err) {
            console.error("Failed to record game result:", err);
        }
    }, []);

    const determineResult = useCallback((data: {
        is_checkmate?: boolean;
        is_stalemate?: boolean;
        is_insufficient_material?: boolean;
        turn?: string;
    }): GameResult => {
        if (data.is_checkmate) {
            const loserTurn = data.turn;
            if (loserTurn === playerColor) return "loss";
            return "win";
        }
        if (data.is_stalemate || data.is_insufficient_material) {
            return "draw";
        }
        return null;
    }, [playerColor]);

    const makePlayerMove = async (from: string, to: string, promotion?: string): Promise<boolean> => {
        if (isLoading || !gameStarted) return false;
        const move = from + to + (promotion || "");
        const playerId = getUserId();

        setIsLoading(true);
        try {
            if (gameType === GameType.PVP) {
                await setBackendPlayerColor(playerColor, {
                    matchId: pvpMatchId,
                    playerId,
                });
            }

            const data = await makeMove(move, {
                matchId: gameType === GameType.PVP ? pvpMatchId : null,
                playerId,
            });

            if (data.error) {
                console.warn("Move rejected:", data.error);
                if (data.board) setFen(data.board);
                return false;
            }

            setFen(data.board ?? "");
            setLastAiMove(data.ai_move ?? null);
            setIsCheck(data.is_check || false);
            setIsCheckmate(data.is_checkmate || false);
            setIsStalemate(data.is_stalemate || false);

            const over = data.is_checkmate || data.is_stalemate || data.is_insufficient_material || false;
            setIsGameOver(over);
            setMoveCount(prev => prev + 1);

            if (over) {
                const result = determineResult(data);
                setGameResult(result);
                await recordGameResult(result);
            }

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
            const fenResult = await resetBoard({
                matchId: gameType === GameType.PVP ? pvpMatchId : null,
            });
            setFen(fenResult ?? "");
            setLastAiMove(null);
            setIsGameOver(false);
            setGameResult(null);
            setMoveCount(0);
            setIsCheck(false);
            setIsCheckmate(false);
            setIsStalemate(false);
            setGameStarted(false);
            setPvpMatchId(null);
            setOpponentId(null);
        } catch (err) {
            console.error("Reset error:", err);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (gameType !== GameType.PVP || !gameStarted) {
            return;
        }

        let stopped = false;

        const syncBoard = async () => {
            const boardState = await getBoard({ matchId: pvpMatchId });
            if (!boardState || stopped) return;

            setFen(boardState.board ?? "");
            setIsCheck(boardState.is_check || false);
            setIsCheckmate(boardState.is_checkmate || false);
            setIsStalemate(boardState.is_stalemate || false);

            const over = boardState.is_checkmate || boardState.is_stalemate || boardState.is_insufficient_material || false;
            setIsGameOver(over);

            if (over) {
                const result = determineResult(boardState);
                setGameResult(result);
            }
        };

        void syncBoard();
        const interval = window.setInterval(() => {
            void syncBoard();
        }, 1200);

        return () => {
            stopped = true;
            window.clearInterval(interval);
        };
    }, [gameType, gameStarted, determineResult, pvpMatchId]);

    const dismissGameResult = () => {
        setGameResult(null);
    };

    const toggleBoardFlip = () => {
        setBoardFlipped(prev => !prev);
    };

    const setGameSettings = (
        newGameType: GameType,
        newPlayerColor: PlayerColor,
        newAiOpponent?: AiOpponent,
        multiplayerOptions?: { matchId?: number; opponentId?: string | null }
    ) => {
        setGameType(newGameType);
        setPlayerColor(newPlayerColor);
        setAiOpponent(newAiOpponent || AiOpponent.DANIBOT);
        setPvpMatchId(multiplayerOptions?.matchId ?? null);
        setOpponentId(multiplayerOptions?.opponentId ?? null);
        setBoardFlipped(newPlayerColor === PlayerColor.BLACK);
        setMoveCount(0);
        setLastAiMove(null);
        setIsGameOver(false);
        setGameResult(null);
        setIsCheck(false);
        setIsCheckmate(false);
        setIsStalemate(false);
        setGameStarted(true);
    };

    return (
        <ChessContext.Provider value={{
            fen,
            setFen,
            isLoading,
            makePlayerMove,
            resetGame,
            lastAiMove,
            setLastAiMove,
            isGameOver,
            gameResult,
            dismissGameResult,
            boardFlipped,
            toggleBoardFlip,
            gameType,
            playerColor,
            aiOpponent,
            pvpMatchId,
            opponentId,
            setGameSettings,
            moveCount,
            gameStarted,
            setGameStarted,
            isCheckmate,
            isStalemate,
            isCheck,
        }}>
            {children}
        </ChessContext.Provider>
    );
};

export const useChess = (): ChessContextType => {
    const context = useContext(ChessContext);
    if (!context) throw new Error("useChess must be used within ChessProvider");
    return context;
};