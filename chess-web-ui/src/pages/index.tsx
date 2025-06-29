"use client";

import React, { useContext } from "react";
import ChessBoard from "../components/ChessBoard";
import {ChessProvider, ChessContext, useChess} from "../context/ChessContext";

export default function Home() {
    return (
        <ChessProvider>
            <HomeContent />
        </ChessProvider>
    );
}

function HomeContent() {
    const { resetGame } = useChess();

    return (
        <main style={{ padding: 20, display: "flex", flexDirection: "column", alignItems: "center" }}>
            <button onClick={resetGame} style={{ marginBottom: 20 }}>
                ♻️ Resetează jocul
            </button>
            <h1>♟️ Chess Engine </h1>
            <div style={{ display: "flex", justifyContent: "center" }}>
                <ChessBoard />
            </div>
        </main>
    );
}