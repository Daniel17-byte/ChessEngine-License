import time

import pytest

import app as app_module


@pytest.fixture()
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_match_sessions():
    # Keep only the default AI session between tests.
    with app_module.session_lock:
        keep_default = app_module.game_sessions.get(app_module.DEFAULT_SESSION_ID)
        app_module.game_sessions.clear()
        app_module.game_sessions[app_module.DEFAULT_SESSION_ID] = keep_default or app_module.create_session()



def test_pvp_sessions_are_isolated_between_matches(client):
    start_m1_white = client.post(
        "/api/game/start_new_game",
        json={"gameType": "pvp", "playerColor": "white", "matchId": "m1", "playerId": "p1"},
    )
    start_m1_black = client.post(
        "/api/game/start_new_game",
        json={"gameType": "pvp", "playerColor": "black", "matchId": "m1", "playerId": "p2"},
    )
    start_m2_white = client.post(
        "/api/game/start_new_game",
        json={"gameType": "pvp", "playerColor": "white", "matchId": "m2", "playerId": "p3"},
    )
    start_m2_black = client.post(
        "/api/game/start_new_game",
        json={"gameType": "pvp", "playerColor": "black", "matchId": "m2", "playerId": "p4"},
    )

    assert start_m1_white.status_code == 200
    assert start_m1_black.status_code == 200
    assert start_m2_white.status_code == 200
    assert start_m2_black.status_code == 200

    move_m1 = client.post(
        "/api/game/make_move",
        json={"matchId": "m1", "playerId": "p1", "move": "e2e4"},
    )
    move_m2 = client.post(
        "/api/game/make_move",
        json={"matchId": "m2", "playerId": "p3", "move": "d2d4"},
    )

    assert move_m1.status_code == 200
    assert move_m2.status_code == 200

    board_m1 = client.get("/api/game/get_board?matchId=m1").get_json()
    board_m2 = client.get("/api/game/get_board?matchId=m2").get_json()

    assert board_m1["board"] != board_m2["board"]
    assert board_m1["turn"] == "black"
    assert board_m2["turn"] == "black"



def test_pvp_move_requires_registered_player(client):
    client.post(
        "/api/game/start_new_game",
        json={"gameType": "pvp", "playerColor": "white", "matchId": "m3", "playerId": "owner"},
    )

    unauthorized = client.post(
        "/api/game/make_move",
        json={"matchId": "m3", "playerId": "intruder", "move": "e2e4"},
    )

    assert unauthorized.status_code == 403
    assert "not registered" in unauthorized.get_json()["error"]



def test_cleanup_removes_stale_sessions():
    with app_module.session_lock:
        app_module.game_sessions["fresh"] = app_module.create_session(game_type="pvp", player_color=app_module.chess.WHITE)
        app_module.game_sessions["stale"] = app_module.create_session(game_type="pvp", player_color=app_module.chess.WHITE)
        app_module.game_sessions["stale"]["last_accessed"] = (
            time.time() - app_module.SESSION_TTL_SECONDS - 10
        )

    removed = app_module.cleanup_expired_sessions(now=time.time())

    assert "stale" in removed
    assert "fresh" not in removed
    assert "stale" not in app_module.game_sessions
    assert "fresh" in app_module.game_sessions

