package org.library.usersbe.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.library.usersbe.model.Role;
import org.library.usersbe.model.User;
import org.library.usersbe.repository.UserRepository;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private PBKDF2PasswordEncoderHelper passwordEncoderHelper;

    @Mock
    private SessionManager sessionManager;

    private UserService userService;

    @BeforeEach
    void setUp() {
        userService = new UserService(userRepository, passwordEncoderHelper, sessionManager);
    }

    // ─── Create User ────────────────────────────────────────────────────

    @Test
    @DisplayName("createUser → encodes password and saves")
    void createUser_encodesPasswordAndSaves() {
        User user = new User();
        user.setUsername("alice");
        user.setPassword("plaintext123");
        user.setRole(Role.CLIENT);

        when(passwordEncoderHelper.encode("plaintext123")).thenReturn("encoded_hash");

        User result = userService.createUser(user);

        assertEquals("encoded_hash", result.getPassword());
        verify(userRepository).save(user);
    }

    @Test
    @DisplayName("createUser → returns the saved user")
    void createUser_returnsUser() {
        User user = new User();
        user.setUsername("bob");
        user.setPassword("pass");

        when(passwordEncoderHelper.encode(any())).thenReturn("hashed");

        User result = userService.createUser(user);

        assertNotNull(result);
        assertEquals("bob", result.getUsername());
    }

    // ─── Update User ────────────────────────────────────────────────────

    @Test
    @DisplayName("updateUser → encodes password and calls repository update")
    void updateUser_encodesAndUpdates() {
        UUID uuid = UUID.randomUUID();
        User user = new User();
        user.setUuid(uuid);
        user.setUsername("alice");
        user.setPassword("newpass");

        when(passwordEncoderHelper.encode("newpass")).thenReturn("new_hash");

        User result = userService.updateUser(user);

        assertEquals("new_hash", result.getPassword());
        verify(userRepository).updateUserByUuid(eq(uuid), eq(user));
    }

    // ─── Get User By UUID ───────────────────────────────────────────────

    @Test
    @DisplayName("getUserByUUID → existing user → returns user")
    void getUserByUUID_existingUser_returnsUser() {
        UUID uuid = UUID.randomUUID();
        User user = new User();
        user.setUuid(uuid);
        user.setUsername("alice");

        when(userRepository.existsUserByUuid(uuid)).thenReturn(true);
        when(userRepository.findByUuid(uuid)).thenReturn(user);

        User result = userService.getUserByUUID(uuid);

        assertNotNull(result);
        assertEquals("alice", result.getUsername());
    }

    @Test
    @DisplayName("getUserByUUID → non-existing user → returns null")
    void getUserByUUID_nonExisting_returnsNull() {
        UUID uuid = UUID.randomUUID();
        when(userRepository.existsUserByUuid(uuid)).thenReturn(false);

        User result = userService.getUserByUUID(uuid);

        assertNull(result);
        verify(userRepository, never()).findByUuid(any());
    }

    // ─── Get All Users ──────────────────────────────────────────────────

    @Test
    @DisplayName("getAllUsers → returns list from repository")
    void getAllUsers_returnsList() {
        User u1 = new User();
        u1.setUsername("alice");
        User u2 = new User();
        u2.setUsername("bob");

        when(userRepository.findAll()).thenReturn(Arrays.asList(u1, u2));

        List<User> result = userService.getAllUsers();

        assertEquals(2, result.size());
    }

    // ─── Delete User ────────────────────────────────────────────────────

    @Test
    @DisplayName("deleteUser → finds and deletes user")
    void deleteUser_findsAndDeletes() {
        UUID uuid = UUID.randomUUID();
        User user = new User();
        user.setUuid(uuid);

        when(userRepository.existsUserByUuid(uuid)).thenReturn(true);
        when(userRepository.findByUuid(uuid)).thenReturn(user);

        userService.deleteUser(uuid);

        verify(userRepository).delete(user);
    }

    // ─── Authenticate ───────────────────────────────────────────────────

    @Test
    @DisplayName("authenticate → valid credentials → returns authenticated user")
    void authenticate_validCredentials_returnsUser() {
        User user = new User();
        user.setUsername("alice");
        user.setPassword("encoded_pass");

        when(userRepository.getUserByUsername("alice")).thenReturn(user);
        when(passwordEncoderHelper.matches("rawpass", "encoded_pass")).thenReturn(true);
        when(sessionManager.createSession(user)).thenReturn("session-123");

        User result = userService.authenticate("alice", "rawpass");

        assertNotNull(result);
        assertEquals("alice", result.getUsername());
        verify(sessionManager).createSession(user);
    }

    @Test
    @DisplayName("authenticate → wrong password → returns null")
    void authenticate_wrongPassword_returnsNull() {
        User user = new User();
        user.setUsername("alice");
        user.setPassword("encoded_pass");

        when(userRepository.getUserByUsername("alice")).thenReturn(user);
        when(passwordEncoderHelper.matches("wrongpass", "encoded_pass")).thenReturn(false);

        User result = userService.authenticate("alice", "wrongpass");

        assertNull(result);
    }

    @Test
    @DisplayName("authenticate → unknown user → returns null")
    void authenticate_unknownUser_returnsNull() {
        when(userRepository.getUserByUsername("unknown")).thenReturn(null);

        User result = userService.authenticate("unknown", "anypass");

        assertNull(result);
    }

    // ─── Get User By Username ───────────────────────────────────────────

    @Test
    @DisplayName("getUserByUsername → returns user from repository")
    void getUserByUsername_returnsUser() {
        User user = new User();
        user.setUsername("alice");

        when(userRepository.getUserByUsername("alice")).thenReturn(user);

        User result = userService.getUserByUsername("alice");

        assertNotNull(result);
        assertEquals("alice", result.getUsername());
    }

    @Test
    @DisplayName("getUserByUsername → unknown → returns null")
    void getUserByUsername_unknown_returnsNull() {
        when(userRepository.getUserByUsername("unknown")).thenReturn(null);

        assertNull(userService.getUserByUsername("unknown"));
    }

}

