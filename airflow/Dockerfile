FROM apache/airflow:2.7.1-python3.11

# Switch to root to install system packages
USER root

# Update sources to use HTTPS and install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Replace http with https in sources.list to ensure secure connections
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list

# Now perform the update and install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        libtool \
        autoconf \
        automake \
        pkg-config \
        libffi-dev \
        python3-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Подгрузка TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
  make && \
  make install

USER airflow

# Обновдление установщиков
RUN pip install --upgrade pip && \
  pip install TA-Lib
#RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements.txt to the container
COPY requirements.txt /requirements.txt
# Копирование файла .env в домашнюю директорию airflow
COPY .env /home/airflow/.env

# Установка зависимотсей из файла
RUN pip install --no-cache-dir -r /requirements.txt