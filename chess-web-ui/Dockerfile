# syntax=docker/dockerfile:1

# ===== Build stage =====
FROM node:20-alpine AS builder

# Director de lucru în container
WORKDIR /app

# Copiem fișierele de configurare și instalăm dependențele
COPY package.json package-lock.json ./
RUN npm install

# Copiem restul codului sursă
COPY . .

# Compilăm aplicația
RUN npm run build

# ===== Runtime stage =====
FROM node:20-alpine AS runner

WORKDIR /app

# Setăm variabile de mediu pentru producție
ENV NODE_ENV production

# Copiem doar fișierele necesare pentru rulare
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Expunem portul Next.js
EXPOSE 3000

# Pornim aplicația
CMD ["npm", "start"]