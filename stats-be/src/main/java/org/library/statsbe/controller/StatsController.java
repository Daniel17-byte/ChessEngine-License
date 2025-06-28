package org.library.statsbe.controller;

import lombok.RequiredArgsConstructor;
import org.library.statsbe.model.GameStat;
import org.library.statsbe.service.StatsService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/stats")
@RequiredArgsConstructor
public class StatsController {
    private final StatsService service;

    @PostMapping("/win/{userId}")
    public ResponseEntity<GameStat> addWin(@PathVariable String userId) {
        return ResponseEntity.ok(service.updateWin(userId));
    }

    @PostMapping("/loss/{userId}")
    public ResponseEntity<GameStat> addLoss(@PathVariable String userId) {
        return ResponseEntity.ok(service.updateLoss(userId));
    }

    @PostMapping("/draw/{userId}")
    public ResponseEntity<GameStat> addDraw(@PathVariable String userId) {
        return ResponseEntity.ok(service.updateDraw(userId));
    }

    @GetMapping("/all")
    public ResponseEntity<List<GameStat>> getAll() {
        return ResponseEntity.ok(service.getAllStats());
    }

    @GetMapping("/{userId}")
    public ResponseEntity<GameStat> getByUser(@PathVariable String userId) {
        return ResponseEntity.ok(service.getByUserId(userId));
    }
}