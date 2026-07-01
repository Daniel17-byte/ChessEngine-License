package org.library.gatewayapi;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.RestTemplate;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

@RestController
public class ApiProxyController {

    private static final List<String> SKIPPED_REQUEST_HEADERS = List.of("host", "content-length", "connection");
    private static final List<String> SKIPPED_RESPONSE_HEADERS = List.of("transfer-encoding", "connection");

    private final RestTemplate restTemplate;
    private final String usersServiceBaseUrl;
    private final String statsServiceBaseUrl;
    private final String matchmakingServiceBaseUrl;
    private final String paymentServiceBaseUrl;
    private final String aiEngineServiceBaseUrl;

    public ApiProxyController(
            RestTemplate restTemplate,
            @Value("${users.service.base-url}") String usersServiceBaseUrl,
            @Value("${stats.service.base-url}") String statsServiceBaseUrl,
            @Value("${matchmaking.service.base-url}") String matchmakingServiceBaseUrl,
            @Value("${payment.service.base-url}") String paymentServiceBaseUrl,
            @Value("${ai.engine.service.base-url}") String aiEngineServiceBaseUrl
    ) {
        this.restTemplate = restTemplate;
        this.usersServiceBaseUrl = trimTrailingSlash(usersServiceBaseUrl);
        this.statsServiceBaseUrl = trimTrailingSlash(statsServiceBaseUrl);
        this.matchmakingServiceBaseUrl = trimTrailingSlash(matchmakingServiceBaseUrl);
        this.paymentServiceBaseUrl = trimTrailingSlash(paymentServiceBaseUrl);
        this.aiEngineServiceBaseUrl = trimTrailingSlash(aiEngineServiceBaseUrl);
    }

    @RequestMapping(
            value = {
                    "/api/users/**",
                    "/api/stats/**",
                    "/api/matchmaking/**",
                    "/api/matches/**",
                    "/api/game/**",
                    "/api/admin/**",
                    "/api/payment/**",
                    "/api/payments/**"
            },
            method = {
                    RequestMethod.GET,
                    RequestMethod.POST,
                    RequestMethod.PUT,
                    RequestMethod.PATCH,
                    RequestMethod.DELETE,
                    RequestMethod.OPTIONS
            }
    )
    public ResponseEntity<byte[]> proxyRequest(HttpServletRequest request, @RequestBody(required = false) byte[] body) {
        HttpMethod method = HttpMethod.valueOf(request.getMethod());
        String path = request.getRequestURI();

        String targetBaseUrl = resolveTargetBaseUrl(path);
        if (targetBaseUrl == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body("No route configured for this API path".getBytes(StandardCharsets.UTF_8));
        }

        String query = request.getQueryString();
        String targetUrl = targetBaseUrl + path + (query == null ? "" : "?" + query);

        HttpHeaders headers = copyRequestHeaders(request);
        HttpEntity<byte[]> entity = new HttpEntity<>(body == null ? new byte[0] : body, headers);

        try {
            ResponseEntity<byte[]> upstreamResponse = restTemplate.exchange(targetUrl, method, entity, byte[].class);
            return ResponseEntity.status(upstreamResponse.getStatusCode())
                    .headers(copyResponseHeaders(upstreamResponse.getHeaders()))
                    .body(upstreamResponse.getBody());
        } catch (HttpStatusCodeException ex) {
            return ResponseEntity.status(ex.getStatusCode())
                    .headers(copyResponseHeaders(ex.getResponseHeaders()))
                    .body(ex.getResponseBodyAsByteArray());
        } catch (Exception ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(("Gateway proxy error: " + ex.getMessage()).getBytes(StandardCharsets.UTF_8));
        }
    }

    private String resolveTargetBaseUrl(String path) {
        if (path.startsWith("/api/users/")) return usersServiceBaseUrl;
        if (path.equals("/api/users")) return usersServiceBaseUrl;

        if (path.startsWith("/api/stats/")) return statsServiceBaseUrl;
        if (path.equals("/api/stats")) return statsServiceBaseUrl;

        if (path.startsWith("/api/matchmaking/") || path.equals("/api/matchmaking")) return matchmakingServiceBaseUrl;
        if (path.startsWith("/api/matches/") || path.equals("/api/matches")) return matchmakingServiceBaseUrl;

        if (path.startsWith("/api/game/") || path.equals("/api/game")) return aiEngineServiceBaseUrl;
        if (path.startsWith("/api/admin/") || path.equals("/api/admin")) return aiEngineServiceBaseUrl;

        if (path.startsWith("/api/payment/") || path.equals("/api/payment")) return paymentServiceBaseUrl;
        if (path.startsWith("/api/payments/") || path.equals("/api/payments")) return paymentServiceBaseUrl;

        return null;
    }

    private HttpHeaders copyRequestHeaders(HttpServletRequest request) {
        HttpHeaders headers = new HttpHeaders();
        request.getHeaderNames().asIterator().forEachRemaining(headerName -> {
            if (SKIPPED_REQUEST_HEADERS.contains(headerName.toLowerCase())) {
                return;
            }
            request.getHeaders(headerName).asIterator().forEachRemaining(value -> headers.add(headerName, value));
        });
        return headers;
    }

    private HttpHeaders copyResponseHeaders(HttpHeaders upstreamHeaders) {
        HttpHeaders headers = new HttpHeaders();
        if (upstreamHeaders == null) {
            return headers;
        }

        for (Map.Entry<String, List<String>> entry : upstreamHeaders.entrySet()) {
            String headerName = entry.getKey();
            if (headerName == null || SKIPPED_RESPONSE_HEADERS.contains(headerName.toLowerCase())) {
                continue;
            }
            headers.put(headerName, entry.getValue());
        }
        return headers;
    }

    private String trimTrailingSlash(String url) {
        if (url.endsWith("/")) {
            return url.substring(0, url.length() - 1);
        }
        return url;
    }
}

