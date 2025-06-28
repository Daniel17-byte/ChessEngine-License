package org.library.statsbe.model;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.Data;

@Entity
@Data
public class GameStat {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String userId;
    private int wins;
    private int losses;
    private int draws;
    public GameStat(Long id, String userId, int wins, int losses, int draws) {
        this.id = id;
        this.userId = userId;
        this.wins = wins;
        this.losses = losses;
        this.draws = draws;
    }

    public GameStat() {}
}