package org.library.matchmakingbe.controller;

import lombok.RequiredArgsConstructor;
import org.library.matchmakingbe.model.Match;
import org.library.matchmakingbe.service.MatchService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/matches")
@RequiredArgsConstructor
public class MatchController {
    private final MatchService matchService;

    @PostMapping("/create")
    public Match create(@RequestParam String playerOneId) {
        return matchService.createMatch(playerOneId);
    }

    @PostMapping("/join/{id}")
    public Match join(@PathVariable Long id, @RequestParam String playerTwoId) {
        return matchService.joinMatch(id, playerTwoId);
    }

    @GetMapping
    public List<Match> all() {
        return matchService.getAllMatches();
    }
}
