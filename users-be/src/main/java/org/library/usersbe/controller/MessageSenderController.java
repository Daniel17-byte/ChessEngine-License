package org.library.usersbe.controller;

import org.library.usersbe.service.MessageProducer;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.UUID;

@RestController
@RequestMapping("/api/users")
@CrossOrigin(origins = {"http://gateway:8080"})
public class MessageSenderController {

    private final MessageProducer messageProducer;

    public MessageSenderController(MessageProducer messageProducer) {
        this.messageProducer = messageProducer;
    }

    @PostMapping("/simulate/{deviceId}")
    public String uploadFile(@RequestParam("file") MultipartFile file, @PathVariable UUID deviceId) {
        long predefinedTimestamp = System.currentTimeMillis();
        String predefinedDeviceId = deviceId.toString();

        try (BufferedReader br = new BufferedReader(new InputStreamReader(file.getInputStream()))) {
            String line;
            while ((line = br.readLine()) != null) {
                try {
                    double measurementValue = Double.parseDouble(line.trim());

                    String jsonPayload = String.format(
                            "{\"timestamp\": %d, \"deviceId\": \"%s\", \"measurementValue\": %.4f}",
                            predefinedTimestamp, predefinedDeviceId, measurementValue
                    );

                    messageProducer.sendMessage(jsonPayload);
                    System.out.println(jsonPayload);
                } catch (Exception ex) {
                    System.out.println("Error processing line: " + ex.getMessage());
                }
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return "Failed to process file: " + e.getMessage();
        }

        return "File processed successfully!";
    }
}