const API_BASE = process.env.NEXT_PUBLIC_STATS_API_BASE || "http://localhost:8080/api/stats";

export interface GameStat {
    userId: string;
    wins: number;
    losses: number;
    draws: number;
    totalGames: number;
    winRate: number;
}

// Ensure totalGames and winRate are always present (backend may or may not compute them)
function normalizeStat(raw: any): GameStat {
    const wins = raw.wins || 0;
    const losses = raw.losses || 0;
    const draws = raw.draws || 0;
    const totalGames = raw.totalGames ?? (wins + losses + draws);
    const winRate = raw.winRate ?? (totalGames > 0 ? wins / totalGames : 0);
    return {
        userId: raw.userId,
        wins,
        losses,
        draws,
        totalGames,
        winRate,
    };
}

export const getPlayerStats = async (userId: string): Promise<GameStat | null> => {
    try {
        const res = await fetch(`${API_BASE}/${userId}`, {
            credentials: "include",
        });

        if (!res.ok) return null;
        const raw = await res.json();
        if (!raw) return null;
        return normalizeStat(raw);
    } catch (error) {
        console.error("Get stats failed:", error);
        return null;
    }
};

export const getAllStats = async (): Promise<GameStat[]> => {
    try {
        const res = await fetch(`${API_BASE}/all`, {
            credentials: "include",
        });

        if (!res.ok) return [];
        const rawList = await res.json();
        return (rawList || []).map(normalizeStat);
    } catch (error) {
        console.error("Get all stats failed:", error);
        return [];
    }
};

export const recordWin = async (userId: string): Promise<GameStat | null> => {
    try {
        const res = await fetch(`${API_BASE}/win/${userId}`, {
            method: "POST",
            credentials: "include",
        });

        if (!res.ok) return null;
        return normalizeStat(await res.json());
    } catch (error) {
        console.error("Record win failed:", error);
        return null;
    }
};

export const recordLoss = async (userId: string): Promise<GameStat | null> => {
    try {
        const res = await fetch(`${API_BASE}/loss/${userId}`, {
            method: "POST",
            credentials: "include",
        });

        if (!res.ok) return null;
        return normalizeStat(await res.json());
    } catch (error) {
        console.error("Record loss failed:", error);
        return null;
    }
};

export const recordDraw = async (userId: string): Promise<GameStat | null> => {
    try {
        const res = await fetch(`${API_BASE}/draw/${userId}`, {
            method: "POST",
            credentials: "include",
        });

        if (!res.ok) return null;
        return normalizeStat(await res.json());
    } catch (error) {
        console.error("Record draw failed:", error);
        return null;
    }
};
