package org.library.usersbe.service;

import org.library.usersbe.UsersBeApplication;
import org.library.usersbe.model.User;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@Service
public class SessionManager {
    private final Map<String, User> sessions = new HashMap<>();

    public String createSession(User user) {
        String sessionId = generateSessionId();
        sessions.put(sessionId, user);
        return sessionId;
    }

    public boolean isValidSession(String sessionId) {
        return sessions.containsKey(sessionId);
    }

    public User getUsernameForSession(String sessionId) {
        return sessions.get(sessionId);
    }

    private String generateSessionId() {
        return UUID.randomUUID().toString();
    }

    public User getUserFromSession() {
        if (!isValidSession(UsersBeApplication.sessionID)){
            return null;
        }

        return sessions.get(UsersBeApplication.sessionID);

    }
}
