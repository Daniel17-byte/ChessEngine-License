# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Setăm directorul de lucru în container
WORKDIR /app

# Copiem fișierele în container
COPY . .

# Instalăm dependențele (folosind requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Expunem portul Flask (default)
EXPOSE 5000

# Pornim aplicația
CMD ["python", "app.py"]