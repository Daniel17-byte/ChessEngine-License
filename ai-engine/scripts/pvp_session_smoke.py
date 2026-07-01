#!/usr/bin/env python3
"""Quick smoke runner for per-match PvP session isolation."""

from pathlib import Path
import sys

AI_ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ENGINE_ROOT))

from app import app


def main() -> int:
    client = app.test_client()

    client.post(
        "/api/game/start_new_game",
        json={"gameType": "pvp", "playerColor": "white", "matchId": "demo-1", "playerId": "alice"},
    )
    client.post(
        "/api/game/start_new_game",
        json={"gameType": "pvp", "playerColor": "black", "matchId": "demo-1", "playerId": "bob"},
    )
    client.post(
        "/api/game/start_new_game",
        json={"gameType": "pvp", "playerColor": "white", "matchId": "demo-2", "playerId": "carol"},
    )
    client.post(
        "/api/game/start_new_game",
        json={"gameType": "pvp", "playerColor": "black", "matchId": "demo-2", "playerId": "dave"},
    )

    move_1 = client.post(
        "/api/game/make_move",
        json={"matchId": "demo-1", "playerId": "alice", "move": "e2e4"},
    )
    move_2 = client.post(
        "/api/game/make_move",
        json={"matchId": "demo-2", "playerId": "carol", "move": "d2d4"},
    )

    board_1 = client.get("/api/game/get_board?matchId=demo-1").get_json()
    board_2 = client.get("/api/game/get_board?matchId=demo-2").get_json()

    print("move statuses:", move_1.status_code, move_2.status_code)
    print("session boards are isolated:", board_1["board"] != board_2["board"])

    return 0 if move_1.status_code == 200 and move_2.status_code == 200 and board_1["board"] != board_2["board"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

