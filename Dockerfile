# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/

# Создание точки входа
CMD ["python", "-m", "src.main", "--particles", "1000", "--steps", "100", "--output-dir", "/app/output"]