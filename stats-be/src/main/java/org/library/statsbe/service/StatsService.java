package org.library.statsbe.service;

import lombok.RequiredArgsConstructor;
import org.library.statsbe.model.GameStat;
import org.library.statsbe.repository.StatsRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class StatsService {
    private final StatsRepository repository;

    public GameStat updateWin(String userId) {
        GameStat stat = repository.findByUserId(userId).orElseGet(() -> new GameStat(null, userId, 0, 0, 0));
        stat.setWins(stat.getWins() + 1);
        return repository.save(stat);
    }

    public GameStat updateLoss(String userId) {
        GameStat stat = repository.findByUserId(userId).orElseGet(() -> new GameStat(null, userId, 0, 0, 0));
        stat.setLosses(stat.getLosses() + 1);
        return repository.save(stat);
    }

    public GameStat updateDraw(String userId) {
        GameStat stat = repository.findByUserId(userId).orElseGet(() -> new GameStat(null, userId, 0, 0, 0));
        stat.setDraws(stat.getDraws() + 1);
        return repository.save(stat);
    }

    public List<GameStat> getAllStats() {
        return repository.findAll();
    }

    public GameStat getByUserId(String userId) {
        return repository.findByUserId(userId).orElse(null);
    }
}