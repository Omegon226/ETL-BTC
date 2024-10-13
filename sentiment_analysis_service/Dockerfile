FROM python:3.11
USER root

# Установите рабочий каталог
WORKDIR /app

# Обновдление установщиков
RUN apt-get update && \
  pip install --upgrade pip

# Копирование requirements.txt в контейнер
COPY requirements.txt .

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the webserver port
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]