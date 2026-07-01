"use client";

import React, { useState } from "react";
import ChessBoard from "../../components/ChessBoard";
import NewGameDialog from "../../components/NewGameDialog";
import { GameOverModal } from "../../components/GameOverModal";
import { MatchStatus } from "../../components/MatchStatus";
import { ChessProvider, useChess } from "../../context/ChessContext";
import { Layout } from "../../components/Layout";
import { ProtectedRoute } from "../../components/ProtectedRoute";

export default function PlayPage() {
    return (
        <ProtectedRoute>
            <Layout>
                <ChessProvider>
                    <PlayContent />
                </ChessProvider>
            </Layout>
        </ProtectedRoute>
    );
}

function PlayContent() {
    const { toggleBoardFlip, gameType, playerColor, aiOpponent, isGameOver, gameStarted, isCheck, pvpMatchId, opponentId } = useChess();
    const [showNewGameDialog, setShowNewGameDialog] = useState(false);

    const handleBoardClick = () => {
        if (!gameStarted && !showNewGameDialog) {
            setShowNewGameDialog(true);
        }
    };

    return (
        <main style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            height: "calc(100vh - 64px)",
            padding: "12px 16px",
            overflow: "hidden",
            boxSizing: "border-box",
        }}>
            {/* Top bar: only visible when game is started */}
            {gameStarted && (
                <div style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    width: "100%",
                    maxWidth: 600,
                    marginBottom: 8,
                    flexWrap: "wrap",
                    justifyContent: "center",
                }}>
                    <button onClick={() => setShowNewGameDialog(true)} style={btnStyle("#4CAF50")}>
                        🎮 New Game
                    </button>
                    <button onClick={toggleBoardFlip} style={btnStyle("#2196F3")}>
                        🔄 Flip
                    </button>

                    <div style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        fontSize: 13,
                        color: "#555",
                        background: "#f5f5f5",
                        padding: "6px 14px",
                        borderRadius: 6,
                    }}>
                        <span>
                            {gameType === "ai"
                                ? (aiOpponent === "model" ? "🧠 vs Danibot" : "♞ vs Stockfish")
                                : `👥 ${opponentId ? `vs ${opponentId}` : "vs Player"}${pvpMatchId ? ` (#${pvpMatchId})` : ""}`}
                        </span>
                        <span>•</span>
                        <span>{playerColor === "white" ? "⚪ White" : "⚫ Black"}</span>
                        {isCheck && <span style={{ color: "#d32f2f", fontWeight: 700 }}>• ⚠️ Check!</span>}
                        {isGameOver && <span style={{ color: "#d32f2f", fontWeight: 700 }}>• 🏁 Game Over</span>}
                    </div>
                </div>
            )}

            {/* Match status - compact */}
            {gameStarted && <MatchStatus />}

            {/* Board area */}
            <div
                style={{
                    flex: 1,
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    width: "100%",
                    position: "relative",
                    minHeight: 0,
                }}
                onClick={handleBoardClick}
            >
                {!gameStarted ? (
                    <div style={{
                        textAlign: "center",
                        cursor: "pointer",
                    }}>
                        <div style={{ fontSize: 80, marginBottom: 20 }}>♟️</div>
                        <h2 style={{ margin: "0 0 10px", color: "#333", fontSize: 26 }}>Ready to Play?</h2>
                        <p style={{ color: "#999", margin: 0, fontSize: 15 }}>Click anywhere to start a new game</p>
                    </div>
                ) : (
                    <ChessBoard />
                )}
            </div>

            {/* Dialogs */}
            <NewGameDialog
                isOpen={showNewGameDialog}
                onClose={() => setShowNewGameDialog(false)}
            />
            <GameOverModal onNewGame={() => setShowNewGameDialog(true)} />
        </main>
    );
}

function btnStyle(bg: string): React.CSSProperties {
    return {
        padding: "8px 16px",
        backgroundColor: bg,
        color: "white",
        border: "none",
        borderRadius: 6,
        cursor: "pointer",
        fontSize: 14,
        fontWeight: 600,
    };
}
