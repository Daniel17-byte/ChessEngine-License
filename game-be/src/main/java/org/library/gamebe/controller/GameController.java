package org.library.gamebe.controller;

import lombok.RequiredArgsConstructor;
import org.library.gamebe.dto.ApiResponse;
import org.library.gamebe.dto.MoveResult;
import org.library.gamebe.model.Board;
import org.library.gamebe.model.Move;
import org.library.gamebe.service.GameService;
import org.library.gamebe.util.MoveStatus;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/game")
@RequiredArgsConstructor
public class GameController {

    private final GameService gameService;

    @PostMapping("/move")
    public ResponseEntity<ApiResponse<Board>> makeMove(@RequestBody Move move) {
        MoveResult valid = gameService.processMove(move);
        if (valid.getStatus().name().equals(MoveStatus.VALID.name())) {
            return ResponseEntity
                    .status(HttpStatus.OK)
                    .body(new ApiResponse<>(
                            "Move accepted",
                            gameService.getBoard()
                    ));
        } else {
            return ResponseEntity
                    .status(HttpStatus.BAD_REQUEST)
                    .body(new ApiResponse<>(
                            "Invalid move",
                            null
                    ));
        }
    }

    @GetMapping("/board")
    public ResponseEntity<ApiResponse<Board>> getBoard() {
        return ResponseEntity
                .ok(new ApiResponse<>(
                        "Current board state",
                        gameService.getBoard()
                ));
    }

    @PostMapping("/reset")
    public ResponseEntity<ApiResponse<Void>> resetBoard() {
        gameService.resetGame();
        return ResponseEntity
                .ok(new ApiResponse<>(
                        "Game has been reset",
                        null
                ));
    }
}