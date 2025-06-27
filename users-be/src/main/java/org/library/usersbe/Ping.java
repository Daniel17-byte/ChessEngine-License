package org.library.usersbe;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/ping")
public class Ping {

    @GetMapping("/ping")
    public String ping() {
        return "pong";
    }
}