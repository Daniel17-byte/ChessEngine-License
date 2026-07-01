package org.library.matchmakingbe.service;

import lombok.RequiredArgsConstructor;
import org.library.matchmakingbe.model.Match;
import org.library.matchmakingbe.util.MatchStatus;
import org.library.matchmakingbe.repository.MatchRepository;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;

import static org.springframework.http.HttpStatus.NOT_FOUND;

@Service
@RequiredArgsConstructor
public class MatchService {
    private final MatchRepository matchRepository;

    public Match createMatch(String playerOneId) {
        Match match = new Match();
        match.setPlayerOneId(playerOneId);
        match.setStatus(MatchStatus.WAITING);
        return matchRepository.save(match);
    }

    public Match joinMatch(Long matchId, String playerTwoId) {
        Match match = matchRepository.findById(matchId).orElseThrow();
        match.setPlayerTwoId(playerTwoId);
        match.setStatus(MatchStatus.ONGOING);
        return matchRepository.save(match);
    }

    public Match getMatchById(Long matchId) {
        return matchRepository.findById(matchId)
                .orElseThrow(() -> new ResponseStatusException(NOT_FOUND, "Match not found"));
    }

    public List<Match> getAllMatches() {
        return matchRepository.findAll();
    }
}