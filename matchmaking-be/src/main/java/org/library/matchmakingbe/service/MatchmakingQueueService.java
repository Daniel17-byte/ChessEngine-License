package org.library.matchmakingbe.service;

import org.library.matchmakingbe.config.RabbitMQConfig;
import org.library.matchmakingbe.dto.MatchFoundEvent;
import org.library.matchmakingbe.dto.QueueEntry;
import org.library.matchmakingbe.dto.QueueStatusResponse;
import org.library.matchmakingbe.model.Match;
import org.library.matchmakingbe.repository.MatchRepository;
import org.library.matchmakingbe.util.MatchStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

@Service
public class MatchmakingQueueService {

    private static final Logger log = LoggerFactory.getLogger(MatchmakingQueueService.class);

    private final RabbitTemplate rabbitTemplate;
    private final MatchRepository matchRepository;

    // In-memory queue of waiting players (ordered by join time)
    private final ConcurrentLinkedQueue<QueueEntry> waitingQueue = new ConcurrentLinkedQueue<>();

    // Track which players are currently in queue (playerId → entry)
    private final ConcurrentHashMap<String, QueueEntry> playerIndex = new ConcurrentHashMap<>();

    // Track matched players (playerId → matchId) until they acknowledge
    private final ConcurrentHashMap<String, Long> matchedPlayers = new ConcurrentHashMap<>();

    public MatchmakingQueueService(RabbitTemplate rabbitTemplate, MatchRepository matchRepository) {
        this.rabbitTemplate = rabbitTemplate;
        this.matchRepository = matchRepository;
    }

    /**
     * Player joins the matchmaking queue.
     * Publishes a message to RabbitMQ and adds to in-memory waiting list.
     */
    public QueueStatusResponse joinQueue(String playerId, String playerName, int rating) {
        // Already in queue?
        if (playerIndex.containsKey(playerId)) {
            log.info("Player {} already in queue, returning current status", playerId);
            return getQueueStatus(playerId);
        }

        // Already matched?
        if (matchedPlayers.containsKey(playerId)) {
            Long matchId = matchedPlayers.get(playerId);
            return new QueueStatusResponse(playerId, "MATCHED", 0, waitingQueue.size(), matchId);
        }

        QueueEntry entry = new QueueEntry(playerId, playerName, rating);

        // Publish to RabbitMQ
        try {
            rabbitTemplate.convertAndSend(
                    RabbitMQConfig.EXCHANGE_NAME,
                    RabbitMQConfig.ROUTING_KEY,
                    entry
            );
            log.info("Published queue entry for player {} to RabbitMQ", playerId);
        } catch (Exception ex) {
            log.warn("RabbitMQ unavailable while publishing queue entry for player {}. Continuing with in-memory matchmaking.", playerId, ex);
        }

        // Add to in-memory queue
        waitingQueue.add(entry);
        playerIndex.put(playerId, entry);

        // Try to match immediately
        tryMatch();

        return getQueueStatus(playerId);
    }

    /**
     * Player leaves the matchmaking queue.
     */
    public QueueStatusResponse leaveQueue(String playerId) {
        QueueEntry removed = playerIndex.remove(playerId);
        if (removed != null) {
            waitingQueue.remove(removed);
            log.info("Player {} left the queue", playerId);
            return new QueueStatusResponse(playerId, "NOT_IN_QUEUE", 0, waitingQueue.size(), null);
        }

        // Also check if player was matched
        Long matchId = matchedPlayers.remove(playerId);
        if (matchId != null) {
            log.info("Player {} left after being matched (matchId={})", playerId, matchId);
        }

        return new QueueStatusResponse(playerId, "NOT_IN_QUEUE", 0, waitingQueue.size(), null);
    }

    /**
     * Get queue status for a player.
     */
    public QueueStatusResponse getQueueStatus(String playerId) {
        // Check if matched
        Long matchId = matchedPlayers.get(playerId);
        if (matchId != null) {
            return new QueueStatusResponse(playerId, "MATCHED", 0, waitingQueue.size(), matchId);
        }

        // Check if in queue
        if (playerIndex.containsKey(playerId)) {
            int position = getQueuePosition(playerId);
            return new QueueStatusResponse(playerId, "QUEUED", position, waitingQueue.size(), null);
        }

        return new QueueStatusResponse(playerId, "NOT_IN_QUEUE", 0, waitingQueue.size(), null);
    }

    /**
     * Get the total number of players waiting in queue.
     */
    public int getQueueSize() {
        return waitingQueue.size();
    }

    /**
     * Try to match two players from the queue.
     * Called after every join. Simple FIFO matching.
     */
    private synchronized void tryMatch() {
        while (waitingQueue.size() >= 2) {
            QueueEntry playerOne = waitingQueue.poll();
            QueueEntry playerTwo = waitingQueue.poll();

            if (playerOne == null || playerTwo == null) break;

            // Remove from index
            playerIndex.remove(playerOne.getPlayerId());
            playerIndex.remove(playerTwo.getPlayerId());

            // Create match in DB
            Match match = new Match();
            match.setPlayerOneId(playerOne.getPlayerId());
            match.setPlayerTwoId(playerTwo.getPlayerId());
            match.setStatus(MatchStatus.ONGOING);
            match = matchRepository.save(match);

            // Track as matched
            matchedPlayers.put(playerOne.getPlayerId(), match.getId());
            matchedPlayers.put(playerTwo.getPlayerId(), match.getId());

            // Publish match-found event to RabbitMQ
            MatchFoundEvent event = new MatchFoundEvent(
                    match.getId(),
                    playerOne.getPlayerId(), playerOne.getPlayerName(),
                    playerTwo.getPlayerId(), playerTwo.getPlayerName()
            );

            try {
                rabbitTemplate.convertAndSend(
                        RabbitMQConfig.EXCHANGE_NAME,
                        RabbitMQConfig.MATCH_FOUND_ROUTING_KEY,
                        event
                );
            } catch (Exception ex) {
                log.warn("RabbitMQ unavailable while publishing match-found event for matchId={}. Continuing.", match.getId(), ex);
            }

            log.info("Match found! {} vs {} → matchId={}",
                    playerOne.getPlayerName(), playerTwo.getPlayerName(), match.getId());
        }
    }

    /**
     * Get position of a player in the queue (1-based).
     */
    private int getQueuePosition(String playerId) {
        int pos = 1;
        for (QueueEntry entry : waitingQueue) {
            if (entry.getPlayerId().equals(playerId)) return pos;
            pos++;
        }
        return 0;
    }
}

