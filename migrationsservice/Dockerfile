FROM python:3.11
USER root

# Установите рабочий каталог
WORKDIR /app

# Обновдление установщиков
RUN apt-get update
RUN pip install --upgrade pip

# Подгрузка TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
  make && \
  make install
RUN pip install TA-Lib
RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz

# Копирование requirements.txt в контейнер
COPY requirements.txt .

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируйте остальную часть приложения
COPY . .

# Запуск JupyterLab
ENTRYPOINT ["python", "init_system.py"]