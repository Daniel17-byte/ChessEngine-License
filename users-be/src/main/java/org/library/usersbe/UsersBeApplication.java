package org.library.usersbe;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class UsersBeApplication {
    public static String sessionID;

    public static void main(String[] args) {
        SpringApplication.run(UsersBeApplication.class, args);
    }

}
