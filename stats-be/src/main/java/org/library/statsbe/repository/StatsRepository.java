package org.library.statsbe.repository;

import org.library.statsbe.model.GameStat;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface StatsRepository extends JpaRepository<GameStat, Long> {
    Optional<GameStat> findByUserId(String userId);
}