"use client";

import React, { useEffect, useRef, useState } from "react";
import { useChess, GameType, PlayerColor, AiOpponent } from "../context/ChessContext";
import { startNewGame } from "../api/chessApi";
import { getMatchById, getQueueStatus, joinQueue, leaveQueue, QueueStatusResponse } from "../api/matchmakingApi";
import styles from "./NewGameDialog.module.css";

interface NewGameDialogProps {
    isOpen: boolean;
    onClose: () => void;
}

const NewGameDialog: React.FC<NewGameDialogProps> = ({ isOpen, onClose }) => {
    const { setGameSettings, setFen, setLastAiMove } = useChess();
    const [selectedGameType, setSelectedGameType] = useState<GameType>(GameType.AI);
    const [selectedColor, setSelectedColor] = useState<PlayerColor>(PlayerColor.WHITE);
    const [selectedOpponent, setSelectedOpponent] = useState<AiOpponent>(AiOpponent.DANIBOT);
    const [isLoading, setIsLoading] = useState(false);
    const [queueStatus, setQueueStatus] = useState<QueueStatusResponse | null>(null);
    const [queueError, setQueueError] = useState<string | null>(null);
    const queuePollRef = useRef<number | null>(null);

    const backendPlayerColor = selectedColor === PlayerColor.WHITE ? "white" : "black";

    const stopQueuePolling = () => {
        if (queuePollRef.current !== null) {
            window.clearInterval(queuePollRef.current);
            queuePollRef.current = null;
        }
    };

    const getPlayerIdentity = (): { playerId: string; playerName: string } => {
        try {
            const storedUser = sessionStorage.getItem("user");
            if (storedUser) {
                const user = JSON.parse(storedUser);
                const playerId = String(user.uuid || user.username || "").trim();
                const playerName = String(user.username || playerId || "Player").trim();
                if (playerId) {
                    return { playerId, playerName };
                }
            }
        } catch {
            // no-op
        }

        const guestStorageKey = "guest_matchmaking_player_id";
        const existingGuestId = localStorage.getItem(guestStorageKey);
        const guestId = existingGuestId || `guest-${crypto.randomUUID()}`;
        if (!existingGuestId) {
            localStorage.setItem(guestStorageKey, guestId);
        }

        return { playerId: guestId, playerName: guestId };
    };

    const handleMatchedGameStart = async (matchId: number) => {
        const { playerId } = getPlayerIdentity();
        const match = await getMatchById(matchId);
        if (!match) {
            setQueueError("Am gasit meciul, dar nu am putut citi detaliile lui.");
            return;
        }

        const isPlayerOne = match.playerOneId === playerId;
        const assignedColor = isPlayerOne ? PlayerColor.WHITE : PlayerColor.BLACK;
        const opponent = isPlayerOne ? match.playerTwoId || null : match.playerOneId || null;

        const responseData = await startNewGame(GameType.PVP, assignedColor, selectedOpponent, {
            matchId,
            playerId,
        });
        if (!responseData.success) {
            setQueueError("Nu am putut porni jocul PvP pe backend.");
            return;
        }
        if (responseData.board) {
            setFen(responseData.board);
        }

        setGameSettings(GameType.PVP, assignedColor, selectedOpponent, { matchId, opponentId: opponent });
        setLastAiMove(null);
        setQueueStatus(null);
        setQueueError(null);
        onClose();
    };

    const handleLeaveQueue = async () => {
        const { playerId } = getPlayerIdentity();
        stopQueuePolling();
        await leaveQueue(playerId);
        setQueueStatus(null);
        setQueueError(null);
        setIsLoading(false);
    };

    const handleStartGame = async () => {
        setIsLoading(true);
        setQueueError(null);
        try {
            if (selectedGameType === GameType.PVP) {
                const identity = getPlayerIdentity();
                const queueResponse = await joinQueue({
                    playerId: identity.playerId,
                    playerName: identity.playerName,
                    rating: 1200,
                });

                if (!queueResponse) {
                    setQueueError("Nu m-am putut conecta la matchmaking.");
                    return;
                }

                setQueueStatus(queueResponse);

                if (queueResponse.status === "MATCHED" && queueResponse.matchId) {
                    await handleMatchedGameStart(queueResponse.matchId);
                    return;
                }

                if (queueResponse.status !== "QUEUED") {
                    setQueueError("Status necunoscut in matchmaking.");
                    return;
                }

                stopQueuePolling();
                queuePollRef.current = window.setInterval(async () => {
                    const status = await getQueueStatus(identity.playerId);
                    if (!status) return;

                    setQueueStatus(status);
                    if (status.status === "MATCHED" && status.matchId) {
                        stopQueuePolling();
                        setIsLoading(true);
                        await handleMatchedGameStart(status.matchId);
                        setIsLoading(false);
                    }
                }, 1500);

                return;
            }

            const responseData = await startNewGame(
                selectedGameType,
                backendPlayerColor,
                selectedOpponent
            );

            if (!responseData.success) {
                console.error("Failed to start game on backend");
                return;
            }

            setGameSettings(selectedGameType, selectedColor, selectedOpponent);

            if (responseData.board) {
                setFen(responseData.board);
            }
            if (responseData.ai_move) {
                setLastAiMove(responseData.ai_move);
            }

            setQueueStatus(null);
            setQueueError(null);

            onClose();
        } catch (error) {
            console.error("Error starting game:", error);
            setQueueError("A aparut o eroare la pornirea jocului.");
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (!isOpen && queueStatus?.status === "QUEUED") {
            void handleLeaveQueue();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen]);

    useEffect(() => {
        return () => {
            stopQueuePolling();
        };
    }, []);

    const waitingInQueue = queueStatus?.status === "QUEUED";

    if (!isOpen) return null;

    return (
        <div className={styles.overlay}>
            <div className={styles.dialog}>
                <h2>🎮 New Game</h2>

                <div className={styles.section}>
                    <h3>Choose Your Color</h3>
                    <div className={styles.colorOptions}>
                        <button
                            className={`${styles.colorBtn} ${selectedColor === PlayerColor.WHITE ? styles.selected : ""}`}
                            onClick={() => setSelectedColor(PlayerColor.WHITE)}
                            disabled={isLoading}
                        >
                            ♔ White
                        </button>
                        <button
                            className={`${styles.colorBtn} ${selectedColor === PlayerColor.BLACK ? styles.selected : ""}`}
                            onClick={() => setSelectedColor(PlayerColor.BLACK)}
                            disabled={isLoading}
                        >
                            ♚ Black
                        </button>
                    </div>
                </div>

                <div className={styles.section}>
                    <h3>Choose Opponent</h3>
                    <div className={styles.opponentOptions}>
                        <button
                            className={`${styles.opponentBtn} ${selectedGameType === GameType.AI ? styles.selected : ""}`}
                            onClick={() => setSelectedGameType(GameType.AI)}
                            disabled={isLoading}
                        >
                            🤖 Play vs AI
                        </button>
                        <button
                            className={`${styles.opponentBtn} ${selectedGameType === GameType.PVP ? styles.selected : ""}`}
                            onClick={() => setSelectedGameType(GameType.PVP)}
                            disabled={isLoading || waitingInQueue}
                        >
                            👥 Play vs Player (Matchmaking)
                        </button>
                    </div>
                </div>

                {/* AI Opponent Selection — only shown when AI is selected */}
                {selectedGameType === GameType.AI && (
                    <div className={styles.section}>
                        <h3>Choose AI Engine</h3>
                        <div className={styles.opponentOptions}>
                            <button
                                className={`${styles.aiBtn} ${selectedOpponent === AiOpponent.DANIBOT ? styles.selectedDanibot : ""}`}
                                onClick={() => setSelectedOpponent(AiOpponent.DANIBOT)}
                                disabled={isLoading}
                            >
                                <span className={styles.aiBtnIcon}>🧠</span>
                                <span className={styles.aiBtnName}>Danibot</span>
                                <span className={styles.aiBtnDesc}>Custom Neural Network</span>
                            </button>
                            <button
                                className={`${styles.aiBtn} ${selectedOpponent === AiOpponent.STOCKFISH ? styles.selectedStockfish : ""}`}
                                onClick={() => setSelectedOpponent(AiOpponent.STOCKFISH)}
                                disabled={isLoading}
                            >
                                <span className={styles.aiBtnIcon}>♞</span>
                                <span className={styles.aiBtnName}>Stockfish</span>
                                <span className={styles.aiBtnDesc}>World-class Engine</span>
                            </button>
                        </div>
                    </div>
                )}

                <div className={styles.actions}>
                    <button
                        className={styles.startBtn}
                        onClick={handleStartGame}
                        disabled={isLoading || waitingInQueue}
                    >
                        {isLoading ? "⏳ Starting..." : waitingInQueue ? "🔎 Searching opponent..." : "▶️ Start Game"}
                    </button>
                    <button
                        className={styles.cancelBtn}
                        onClick={() => {
                            if (waitingInQueue) {
                                void handleLeaveQueue();
                            }
                            onClose();
                        }}
                        disabled={isLoading && !waitingInQueue}
                    >
                        {waitingInQueue ? "✕ Leave Queue" : "✕ Cancel"}
                    </button>
                </div>

                {(queueStatus || queueError) && (
                    <div className={styles.queueInfo}>
                        {queueError && <p className={styles.queueError}>{queueError}</p>}
                        {!queueError && queueStatus?.status === "QUEUED" && (
                            <p>
                                In cautare... pozitia ta: <strong>{queueStatus.queuePosition}</strong> / {queueStatus.queueSize}
                            </p>
                        )}
                        {!queueError && queueStatus?.status === "MATCHED" && queueStatus.matchId && (
                            <p>Meci gasit! Match #{queueStatus.matchId}</p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default NewGameDialog;

