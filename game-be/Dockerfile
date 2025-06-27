# syntax=docker/dockerfile:1

# === Build stage ===
FROM eclipse-temurin:21-jdk AS build

WORKDIR /app

# Copiem fișierele necesare pentru a instala dependențele
COPY pom.xml .
COPY mvnw .
COPY .mvn/ .mvn/
COPY src ./src

RUN chmod +x mvnw

# Compilăm proiectul (fără teste, pentru performanță)
RUN ./mvnw clean package -DskipTests

# === Final image ===
FROM eclipse-temurin:21-jre AS final

WORKDIR /app

# Copiem JAR-ul din imaginea de build
COPY --from=build /app/target/*.jar app.jar

# Expunem portul default Spring Boot
EXPOSE 8080

# Pornim aplicația
ENTRYPOINT ["java", "-jar", "app.jar"]