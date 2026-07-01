const API_BASE = process.env.NEXT_PUBLIC_MATCHMAKING_API_BASE || "http://localhost:8080/api";

export interface Match {
    id: number;
    playerOneId: string;
    playerTwoId?: string;
    status: "WAITING" | "ONGOING" | "FINISHED";
    winnerId?: string;
    createdAt?: string;
}

export interface QueueJoinRequest {
    playerId: string;
    playerName: string;
    rating?: number;
}

export interface QueueStatusResponse {
    playerId: string;
    status: "QUEUED" | "MATCHED" | "NOT_IN_QUEUE";
    queuePosition: number;
    queueSize: number;
    matchId: number | null;
}

export const createMatch = async (playerOneId: string): Promise<Match | null> => {
    try {
        const res = await fetch(`${API_BASE}/matches/create?playerOneId=${playerOneId}`, {
            method: "POST",
            credentials: "include",
        });

        if (!res.ok) return null;
        return await res.json();
    } catch (error) {
        console.error("Create match failed:", error);
        return null;
    }
};

export const joinMatch = async (matchId: number, playerTwoId: string): Promise<Match | null> => {
    try {
        const res = await fetch(`${API_BASE}/matches/join/${matchId}?playerTwoId=${playerTwoId}`, {
            method: "POST",
            credentials: "include",
        });

        if (!res.ok) return null;
        return await res.json();
    } catch (error) {
        console.error("Join match failed:", error);
        return null;
    }
};

export const getAllMatches = async (): Promise<Match[]> => {
    try {
        const res = await fetch(`${API_BASE}/matches`, {
            credentials: "include",
        });

        if (!res.ok) return [];
        return await res.json();
    } catch (error) {
        console.error("Get all matches failed:", error);
        return [];
    }
};

export const joinQueue = async (request: QueueJoinRequest): Promise<QueueStatusResponse | null> => {
    try {
        const res = await fetch(`${API_BASE}/matchmaking/join`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            credentials: "include",
            body: JSON.stringify(request),
        });

        if (!res.ok) return null;
        return await res.json();
    } catch (error) {
        console.error("Join queue failed:", error);
        return null;
    }
};

export const leaveQueue = async (playerId: string): Promise<QueueStatusResponse | null> => {
    try {
        const res = await fetch(`${API_BASE}/matchmaking/leave`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            credentials: "include",
            body: JSON.stringify({ playerId }),
        });

        if (!res.ok) return null;
        return await res.json();
    } catch (error) {
        console.error("Leave queue failed:", error);
        return null;
    }
};

export const getQueueStatus = async (playerId: string): Promise<QueueStatusResponse | null> => {
    try {
        const res = await fetch(`${API_BASE}/matchmaking/status/${playerId}`, {
            credentials: "include",
        });

        if (!res.ok) return null;
        return await res.json();
    } catch (error) {
        console.error("Get queue status failed:", error);
        return null;
    }
};

export const getMatchById = async (matchId: number): Promise<Match | null> => {
    try {
        const res = await fetch(`${API_BASE}/matches/${matchId}`, {
            credentials: "include",
        });

        if (!res.ok) return null;
        return await res.json();
    } catch (error) {
        console.error("Get match failed:", error);
        return null;
    }
};

