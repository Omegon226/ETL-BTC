import numpy as np
import pandas as pd
import datetime
import ccxt
import logging
import random
import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
from influxdb_client.rest import ApiException
from influxdb_client.client.write_api import SYNCHRONOUS


from timeseries_utils.filters import apply_kalman_filtering, apply_wavelet_filtering, apply_savitzky_golay_filtering
from timeseries_utils.smoothers import apply_gaussian_smoothing, apply_sma_smoothing, apply_exponential_smoothing
from timeseries_utils.ta_signals import adx_signal, rsi_signal, ppo_signal, macd_signal, bbands_signal


random.seed(0)
np.random.seed()
load_dotenv()


INFLUXDB_URL = "http://influxdb:8086" # Прод
#INFLUXDB_URL = "http://localhost:8010" # Локальный дебаг
INFLUXDB_TOKEN = os.getenv('INFLUXDB_INIT_ADMIN_TOKEN')
INFLUXDB_ORG = os.getenv('INFLUXDB_INIT_ORG')
INFLUXDB_BUCKET_NAME = 'btc-usdt-bucket'
INFLUXDB_MEASUREMENT_NAME = '1h-timeframe'


def get_historical_data():
    # Биржа из которой будут браться данные с помощью CCXT
    EXCHANGE = ccxt.okx()
    # Инструмент в формате символа для обработки
    SYMBOL = "BTC/USDT"
    # Таймфрейм свеч
    TIMEFRAME = "1h"

    # Получение данных
    from_ts = EXCHANGE.parse8601('2023-01-10 00:00:00')

    ohlcv_list = []
    ohlcv = EXCHANGE.fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, since=from_ts, limit=100)
    ohlcv_list.append(ohlcv)

    while True:
        from_ts = ohlcv[-1][0]
        new_ohlcv = EXCHANGE.fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, since=from_ts, limit=100)
        ohlcv.extend(new_ohlcv)

        print(f"Полученны данные до {EXCHANGE.iso8601(from_ts)}")

        if len(new_ohlcv) <= 1:
            break

    ohlcv = pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    ohlcv["Date"] = ohlcv["Date"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))

    return ohlcv


def clean_historical_data(ohlcv):
    # Удаление дубликатов
    ohlcv = ohlcv.drop_duplicates()

    # Замена отсутствующих значений с помощью скользящей медианы
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        ohlcv[column] = ohlcv[column].fillna(
            ohlcv[column].rolling(window=5, min_periods=1).median()
        )

    # Оставшиеся nan значения удаляем
    ohlcv = ohlcv.dropna()

    return ohlcv


def add_filters_and_smoothers(ohlcv):
    with pd.option_context('mode.chained_assignment', None):
        ohlcv["Close_kalman_filter"] = apply_kalman_filtering(ohlcv["Close"])
        ohlcv["Close_savitzky_golay_filter"] = apply_savitzky_golay_filtering(ohlcv["Close"])
        ohlcv["Close_wavelet_filter"] = apply_wavelet_filtering(ohlcv["Close"])
        ohlcv["Close_gaussian_smoothing"] = apply_gaussian_smoothing(ohlcv["Close"])
        ohlcv["Close_sma_smoothing"] = apply_sma_smoothing(ohlcv["Close"])
        ohlcv["Close_exponential_smoothing"] = apply_exponential_smoothing(ohlcv["Close"])

        return ohlcv


def add_ta_signals(ohlcv):
    # Отключение warning-ов
    with pd.option_context('mode.chained_assignment', None):
        rsi_signals = rsi_signal(ohlcv)
        ohlcv["RSI_buy_signal"] = rsi_signals[0]
        ohlcv["RSI_sell_signal"] = rsi_signals[1]

        bbands_signals = bbands_signal(ohlcv)
        ohlcv["BBANDS_buy_signal"] = bbands_signals[0]
        ohlcv["BBANDS_sell_signal"] = bbands_signals[1]

        macd_signals = macd_signal(ohlcv)
        ohlcv["MACD_buy_signal"] = macd_signals[0]
        ohlcv["MACD_sell_signal"] = macd_signals[1]

        ppo_signals = ppo_signal(ohlcv)
        ohlcv["PPO_buy_signal"] = ppo_signals[0]
        ohlcv["PPO_sell_signal"] = ppo_signals[1]

        adx_signals = adx_signal(ohlcv)
        ohlcv["ADX_buy_signal"] = adx_signals[0]
        ohlcv["ADX_sell_signal"] = adx_signals[1]

        return ohlcv


def create_bucket():
    global INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET_NAME

    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

    try:
        bucket = client.buckets_api().create_bucket(
            org_id=INFLUXDB_ORG,
            bucket_name=INFLUXDB_BUCKET_NAME,
            description="Данные BTC/USDT"
        )
        print(f"Бакет '{bucket.name}' успешно создан")
    except ApiException:
        print(f"Бакет '{INFLUXDB_BUCKET_NAME}' уже существует")

    client.close()


def write_data_in_bucket(ohlcv):
    global INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET_NAME, INFLUXDB_MEASUREMENT_NAME

    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

    write_api = client.write_api(write_options=SYNCHRONOUS)

    write_api.write(
        bucket=INFLUXDB_BUCKET_NAME,
        record=ohlcv,
        data_frame_measurement_name=INFLUXDB_MEASUREMENT_NAME,
        data_frame_timestamp_column="Date",
    )

    client.close()


def init_influxdb():
    print("Получение историческихданных")
    ohlcv = get_historical_data()
    print("Обработка исторических данных")
    ohlcv = clean_historical_data(ohlcv)
    print("Добавление сглаживателей и фильтров на основе Close")
    ohlcv = add_filters_and_smoothers(ohlcv)
    print("Добавление сигналов на основе тех. анализа")
    ohlcv = add_ta_signals(ohlcv)
    print("Создание бакета")
    create_bucket()
    print("Добавление данных в бакет")
    write_data_in_bucket(ohlcv)
