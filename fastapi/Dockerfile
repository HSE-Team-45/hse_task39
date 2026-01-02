# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем системные зависимости (если нужны для pandas/numpy)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Создаем необходимые директории
RUN mkdir -p outputs

# Открываем порт
EXPOSE 8000

# Переменная для порта (можно переопределить при запуске)
ENV PORT=8000

# Команда запуска с поддержкой переменной PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]