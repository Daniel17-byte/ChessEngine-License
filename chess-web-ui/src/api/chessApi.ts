const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8080/api/game";

export interface GameRequestContext {
    matchId?: number | string | null;
    playerId?: string | null;
}

const withMatchIdQuery = (path: string, context?: GameRequestContext): string => {
    if (!context?.matchId) {
        return `${API_BASE}/${path}`;
    }

    const matchId = encodeURIComponent(String(context.matchId));
    return `${API_BASE}/${path}?matchId=${matchId}`;
};

export const setPlayerColor = async (color: "white" | "black", context?: GameRequestContext): Promise<boolean> => {
    try {
        const res = await fetch(`${API_BASE}/set_player_color`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                color,
                matchId: context?.matchId,
                playerId: context?.playerId,
            }),
        });
        return res.ok;
    } catch {
        return false;
    }
};

export const getBoard = async (context?: GameRequestContext): Promise<{
    board: string;
    turn: "white" | "black";
    is_check: boolean;
    is_checkmate: boolean;
    is_stalemate: boolean;
    is_insufficient_material: boolean;
} | null> => {
    try {
        const res = await fetch(withMatchIdQuery("get_board", context));
        if (!res.ok) return null;
        return await res.json();
    } catch {
        return null;
    }
};

export const makeMove = async (
    move: string,
    context?: GameRequestContext
): Promise<{
    board?: string;
    ai_move?: string;
    turn?: "white" | "black";
    is_check?: boolean;
    is_checkmate?: boolean;
    is_stalemate?: boolean;
    is_insufficient_material?: boolean;
    error?: string;
}> => {
    try {
        const res = await fetch(`${API_BASE}/make_move`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                move,
                matchId: context?.matchId,
                playerId: context?.playerId,
            }),
        });

        const data = await res.json();
        if (!res.ok) {
            return {
                error: data?.error || "Unknown error",
                board: data?.board,
            };
        }

        return data;
    } catch {
        return { error: "Network error" };
    }
};

export const resetBoard = async (context?: GameRequestContext): Promise<string | null> => {
    try {
        const res = await fetch(`${API_BASE}/reset`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ matchId: context?.matchId }),
        });
        if (!res.ok) return null;
        const data = await res.json();
        return data.board;
    } catch {
        return null;
    }
};

export const startNewGame = async (
    gameType: "ai" | "pvp",
    playerColor: "white" | "black",
    aiStrategy: string = "model",
    context?: GameRequestContext
): Promise<{
    success: boolean;
    board?: string;
    ai_move?: string;
    turn?: string;
}> => {
    try {
        const res = await fetch(`${API_BASE}/start_new_game`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                gameType,
                playerColor,
                aiStrategy,
                matchId: context?.matchId,
                playerId: context?.playerId,
            }),
        });
        if (!res.ok) return { success: false };
        const data = await res.json();
        return { success: true, ...data };
    } catch {
        return { success: false };
    }
};
