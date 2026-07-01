# AI Engine PvP Session Tests

These tests verify that PvP game state is isolated per `matchId` and that stale sessions are cleaned up.

## Run

```bash
cd /Users/daniellungu/Desktop/ChessEngine/ChessEngine/ai-engine
./venv/bin/python -m pytest -q tests/test_pvp_sessions.py
```

## Quick Smoke Runner

```bash
cd /Users/daniellungu/Desktop/ChessEngine/ChessEngine/ai-engine
./venv/bin/python scripts/pvp_session_smoke.py
```

## TTL Configuration

Session cleanup behavior can be tuned with environment variables:

- `GAME_SESSION_TTL_SECONDS` (default: `1800`)
- `GAME_SESSION_CLEANUP_INTERVAL_SECONDS` (default: `60`)

