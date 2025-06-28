package org.library.usersbe.service;

import org.library.usersbe.UsersBeApplication;
import org.library.usersbe.model.User;
import org.library.usersbe.repository.UserRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;

@Service
public class UserService {
    private final UserRepository userRepository;
    private final PBKDF2PasswordEncoderHelper passwordEncoderHelper;
    private final SessionManager sessionManager;

    public UserService(UserRepository userRepository, PBKDF2PasswordEncoderHelper passwordEncoderHelper, SessionManager sessionManager) {
        this.userRepository = userRepository;
        this.passwordEncoderHelper = passwordEncoderHelper;
        this.sessionManager = sessionManager;
    }

    public User createUser(User user) {
        user.setPassword(passwordEncoderHelper.encode(user.getPassword()));
        userRepository.save(user);
        return user;
    }

    public User updateUser(User user) {
        user.setPassword(passwordEncoderHelper.encode(user.getPassword()));
        userRepository.updateUserByUuid(user.getUuid(), user);
        return user;
    }

    public User getUserByUUID(UUID uuid) {
        if (userRepository.existsUserByUuid(uuid)) {
            return userRepository.findByUuid(uuid);
        }
        return null;
    }

    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    public void deleteUser(UUID uuid) {
        User user = getUserByUUID(uuid);
        userRepository.delete(user);
    }

    public boolean authenticate(String username, String password) {
        User user = userRepository.getUserByUsername(username);

        if (user == null) {
            return false;
        }

        if (passwordEncoderHelper.matches(password, user.getPassword())) {
            UsersBeApplication.sessionID = sessionManager.createSession(user);
            return true;
        } else
            return false;

    }

    public User getUserFromSession() {
        return sessionManager.getUserFromSession();
    }

    public User getUserByUsername(String username) {
        return userRepository.getUserByUsername(username);
    }
}
