import { io, Socket } from "socket.io-client";

const ADMIN_API_HOST = process.env.NEXT_PUBLIC_ADMIN_API_HOST || "http://localhost:8080";
const SOCKET_HOST = process.env.NEXT_PUBLIC_ADMIN_SOCKET_HOST || "http://localhost:5050";
const ADMIN_API_BASE = `${ADMIN_API_HOST}/api/admin`;

export interface TrainingStatus {
    is_training: boolean;
    strategy: string | null;
    epochs: number;
    max_epochs: number;
    games_played: number;
    wins_white: number;
    wins_black: number;
    draws: number;
    loss_value: number;
    accuracy: number;
    start_time: string | null;
    current_status: string;
}

export interface TrainingStrategy {
    id: string;
    name: string;
    description: string;
    recommended_epochs: number;
}

// --- WebSocket for real-time training status ---
let socket: Socket | null = null;

export function connectTrainingSocket(
    onStatus: (status: TrainingStatus) => void
): () => void {
    if (socket) {
        socket.disconnect();
    }

    socket = io(SOCKET_HOST, {
        transports: ["websocket", "polling"],
        reconnection: true,
        reconnectionDelay: 1000,
    });

    socket.on("connect", () => {
        console.log("🟢 Training WebSocket connected");
    });

    socket.on("training_status", (data: TrainingStatus) => {
        onStatus(data);
    });

    socket.on("disconnect", () => {
        console.log("🔴 Training WebSocket disconnected");
    });

    // Return cleanup function
    return () => {
        if (socket) {
            socket.disconnect();
            socket = null;
        }
    };
}

// --- REST APIs (start, stop, strategies) ---

export const getTrainingStatus = async (): Promise<TrainingStatus | null> => {
    try {
        const res = await fetch(`${ADMIN_API_BASE}/training/status`, {
            credentials: "include",
        });
        if (!res.ok) return null;
        return await res.json();
    } catch (error) {
        console.error("Get training status failed:", error);
        return null;
    }
};

export const startTraining = async (
    strategy: string,
    epochs: number,
    max_moves: number = 80,
    white_strategy: string = "model",
    black_strategy: string = "model",
    fen_type: string = "endgames"
): Promise<boolean> => {
    try {
        const res = await fetch(`${ADMIN_API_BASE}/training/start`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                strategy,
                epochs,
                max_moves,
                white_strategy,
                black_strategy,
                fen_type
            }),
            credentials: "include",
        });
        return res.ok;
    } catch (error) {
        console.error("Start training failed:", error);
        return false;
    }
};

export const stopTraining = async (): Promise<boolean> => {
    try {
        const res = await fetch(`${ADMIN_API_BASE}/training/stop`, {
            method: "POST",
            credentials: "include",
        });
        return res.ok;
    } catch (error) {
        console.error("Stop training failed:", error);
        return false;
    }
};

export const getStrategies = async (): Promise<TrainingStrategy[]> => {
    try {
        const res = await fetch(`${ADMIN_API_BASE}/strategies`, {
            credentials: "include",
        });
        if (!res.ok) return [];
        const data = await res.json();
        return data.strategies || [];
    } catch (error) {
        console.error("Get strategies failed:", error);
        return [];
    }
};

// --- ELO Estimation ---

export interface EloLevelResult {
    wins: number;
    draws: number;
    losses: number;
    win_rate: number;
    elo: number;
}

export interface EloStatus {
    is_running: boolean;
    current_level: number;
    max_level: number;
    games_per_level: number;
    current_game: number;
    results: Record<string, EloLevelResult>;
    estimated_elo: number | null;
    status: string;
}

export const startEloEstimation = async (gamesPerLevel: number = 20, maxLevel: number = 10): Promise<boolean> => {
    try {
        const res = await fetch(`${ADMIN_API_BASE}/elo/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ games_per_level: gamesPerLevel, max_level: maxLevel }),
            credentials: "include",
        });
        return res.ok;
    } catch (error) {
        console.error("Start ELO estimation failed:", error);
        return false;
    }
};

export const stopEloEstimation = async (): Promise<boolean> => {
    try {
        const res = await fetch(`${ADMIN_API_BASE}/elo/stop`, {
            method: "POST",
            credentials: "include",
        });
        return res.ok;
    } catch (error) {
        console.error("Stop ELO estimation failed:", error);
        return false;
    }
};

export const getEloStatus = async (): Promise<EloStatus | null> => {
    try {
        const res = await fetch(`${ADMIN_API_BASE}/elo/status`, {
            credentials: "include",
        });
        if (!res.ok) return null;
        return await res.json();
    } catch (error) {
        console.error("Get ELO status failed:", error);
        return null;
    }
};

export function connectEloSocket(onStatus: (status: EloStatus) => void): () => void {
    // Reuse the existing socket or create one
    if (!socket) {
        socket = io(SOCKET_HOST, {
            transports: ["websocket", "polling"],
            reconnection: true,
            reconnectionDelay: 1000,
        });
    }

    socket.on("elo_status", (data: EloStatus) => {
        onStatus(data);
    });

    return () => {
        if (socket) {
            socket.off("elo_status");
        }
    };
}

