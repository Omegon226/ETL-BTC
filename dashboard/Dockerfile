FROM python:3.11
USER root

# Установка JupyterLab и необходимых пакетов
RUN pip install --upgrade pip

# Копирование requirements.txt в контейнер
COPY requirements.txt /tmp/

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Создание рабочей директории
WORKDIR /home/jovyan/work

COPY . .

# Expose the webserver port
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Hello.py", "--server.port=8501", "--server.address=0.0.0.0"]