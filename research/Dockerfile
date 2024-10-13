FROM python:3.11
USER root

# Обновдление установщиков
RUN apt-get update && \
  pip install --upgrade pip

# Подгрузка TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
  make && \
  make install && \
  pip install TA-Lib && \
  rm -R ta-lib ta-lib-0.4.0-src.tar.gz && \
  pip install jupyterlab

# Копирование requirements.txt в контейнер
COPY requirements.txt /tmp/ && \
 .env /tmp/

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Создание рабочей директории
WORKDIR /home/jovyan/work

# Запуск JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]