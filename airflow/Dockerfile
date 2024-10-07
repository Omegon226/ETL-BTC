FROM apache/airflow:2.7.1-python3.11

# Обновдление установщиков
RUN pip install --upgrade pip

# Copy requirements.txt to the container
COPY requirements.txt /requirements.txt

# Установка зависимотсей из файла
RUN pip install --no-cache-dir -r /requirements.txt