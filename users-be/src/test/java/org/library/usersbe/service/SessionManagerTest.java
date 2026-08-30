package org.library.usersbe.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.library.usersbe.model.User;

import static org.junit.jupiter.api.Assertions.*;

class SessionManagerTest {

    private SessionManager sessionManager;

    @BeforeEach
    void setUp() {
        sessionManager = new SessionManager();
    }

    @Test
    @DisplayName("createSession → returns non-null session ID")
    void createSession_returnsSessionId() {
        User user = new User();
        user.setUsername("alice");

        String sessionId = sessionManager.createSession(user);

        assertNotNull(sessionId);
        assertFalse(sessionId.isBlank());
    }

    @Test
    @DisplayName("createSession → session is valid afterwards")
    void createSession_sessionIsValid() {
        User user = new User();
        user.setUsername("alice");

        String sessionId = sessionManager.createSession(user);

        assertTrue(sessionManager.isValidSession(sessionId));
    }

    @Test
    @DisplayName("isValidSession → invalid session → returns false")
    void isValidSession_invalidSession_returnsFalse() {
        assertFalse(sessionManager.isValidSession("non-existent"));
    }

    @Test
    @DisplayName("isValidSession → null session → returns false")
    void isValidSession_nullSession_returnsFalse() {
        assertFalse(sessionManager.isValidSession(null));
    }

    @Test
    @DisplayName("getUsernameForSession → returns correct user")
    void getUsernameForSession_returnsUser() {
        User user = new User();
        user.setUsername("bob");

        String sessionId = sessionManager.createSession(user);
        User result = sessionManager.getUsernameForSession(sessionId);

        assertNotNull(result);
        assertEquals("bob", result.getUsername());
    }

    @Test
    @DisplayName("getUsernameForSession → invalid session → returns null")
    void getUsernameForSession_invalidSession_returnsNull() {
        assertNull(sessionManager.getUsernameForSession("fake-id"));
    }

    @Test
    @DisplayName("Multiple sessions → each tracks its own user")
    void multipleSessions_trackOwnUsers() {
        User alice = new User();
        alice.setUsername("alice");
        User bob = new User();
        bob.setUsername("bob");

        String s1 = sessionManager.createSession(alice);
        String s2 = sessionManager.createSession(bob);

        assertNotEquals(s1, s2);
        assertEquals("alice", sessionManager.getUsernameForSession(s1).getUsername());
        assertEquals("bob", sessionManager.getUsernameForSession(s2).getUsername());
    }
}

