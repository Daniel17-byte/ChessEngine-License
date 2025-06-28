package org.library.usersbe.service;

import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.crypto.password.Pbkdf2PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class PBKDF2PasswordEncoderHelper {

    @Value("${password.encoder.secret}")
    private String secret;

    @Value("${password.encoder.salt-length}")
    private int saltLength;

    @Value("${password.encoder.iterations}")
    private int iterations;

    @Value("${password.encoder.hash-width}")
    private int hashWidth;

    private Pbkdf2PasswordEncoder passwordEncoder;

    @PostConstruct
    public void init() {
        this.passwordEncoder = new Pbkdf2PasswordEncoder(secret, saltLength, iterations, hashWidth);
    }

    public String encode(String rawPassword) {
        return passwordEncoder.encode(rawPassword);
    }

    public boolean matches(String rawPassword, String encodedPassword) {
        return passwordEncoder.matches(rawPassword, encodedPassword);
    }
}