package org.library.matchmakingbe.controller;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.library.matchmakingbe.model.Match;
import org.library.matchmakingbe.service.MatchService;
import org.library.matchmakingbe.util.MatchStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;

import static org.mockito.Mockito.when;
import static org.springframework.http.HttpStatus.NOT_FOUND;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(MatchController.class)
class MatchControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private MatchService matchService;

    @Test
    @DisplayName("GET /api/matches/{id} returns the requested match")
    void getById_existingMatch_returnsMatch() throws Exception {
        Match match = new Match();
        match.setId(10L);
        match.setPlayerOneId("p1");
        match.setPlayerTwoId("p2");
        match.setStatus(MatchStatus.ONGOING);

        when(matchService.getMatchById(10L)).thenReturn(match);

        mockMvc.perform(get("/api/matches/10"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(10))
                .andExpect(jsonPath("$.playerOneId").value("p1"))
                .andExpect(jsonPath("$.playerTwoId").value("p2"))
                .andExpect(jsonPath("$.status").value("ONGOING"));
    }

    @Test
    @DisplayName("GET /api/matches/{id} returns 404 when match is missing")
    void getById_missingMatch_returnsNotFound() throws Exception {
        when(matchService.getMatchById(999L))
                .thenThrow(new ResponseStatusException(NOT_FOUND, "Match not found"));

        mockMvc.perform(get("/api/matches/999"))
                .andExpect(status().isNotFound());
    }

    @Test
    @DisplayName("GET /api/matches returns all matches")
    void getAll_returnsMatches() throws Exception {
        Match match = new Match();
        match.setId(20L);
        match.setPlayerOneId("p3");
        match.setPlayerTwoId("p4");
        match.setStatus(MatchStatus.ONGOING);

        when(matchService.getAllMatches()).thenReturn(List.of(match));

        mockMvc.perform(get("/api/matches"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].id").value(20))
                .andExpect(jsonPath("$[0].playerOneId").value("p3"));
    }
}

