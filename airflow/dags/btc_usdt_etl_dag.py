from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
import ccxt
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
import talib
from dotenv import load_dotenv
from pathlib import Path
import os

from utils.filters import apply_kalman_filtering, apply_wavelet_filtering, apply_savitzky_golay_filtering
from utils.smoothers import apply_gaussian_smoothing, apply_sma_smoothing, apply_exponential_smoothing
from utils.ta_signals import adx_signal, rsi_signal, ppo_signal, macd_signal, bbands_signal


load_dotenv(dotenv_path=Path('/home/airflow/.env'))

INFLUXDB_URL = "http://influxdb:8086"
INFLUXDB_TOKEN = os.getenv('INFLUXDB_INIT_ADMIN_TOKEN')
INFLUXDB_ORG = os.getenv('INFLUXDB_INIT_ORG')
INFLUXDB_BUCKET_NAME = 'btc-usdt-bucket'
INFLUXDB_MEASUREMENT_NAME = '1h-timeframe'

EXCHANGE = ccxt.okx()
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LAST_CLOSE_POINT_COUNT = 299

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
} 


# Функция для считывания данных из OKX
def fetch_okx_data(**kwargs):
    logging.info("Считывание данных из OKX")
    try:
        bars = EXCHANGE.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1)
        ohlcv = pd.DataFrame(bars, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        ohlcv["Date"] = ohlcv["Date"].apply(lambda x: datetime.fromtimestamp(x / 1000))
        kwargs['ti'].xcom_push(key='current_data', value=ohlcv.to_json(date_format='iso', orient='split'))
        logging.info("Данные успешно считаны и сохранены в XCom")
    except Exception as e:
        logging.error(f"Ошибка при считывании данных из OKX: {e}")
        raise

# Функция для загрузки последних 299 точек из InfluxDB
def load_last_data_from_influxdb(**kwargs):
    logging.info("Загрузка последних 299 точек из InfluxDB")
    try:
        query_api = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG).query_api()
        
        # Получение последней точки
        check_query = f'''
        from(bucket:"{INFLUXDB_BUCKET_NAME}")
            |> range(start: -inf)
            |> filter(fn: (r) => r["_measurement"] == "{INFLUXDB_MEASUREMENT_NAME}")
            |> last()
            |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        result = query_api.query_data_frame(check_query)
        if result.empty:
            raise Exception("В InfluxDB нет данных, заполните её")
        else:
            start_query_time = result["_time"].copy() - pd.Timedelta(hours=299-1)
            start_query_time = start_query_time[0]
            start_query_time = start_query_time.isoformat()
        
        # Запрос последних 299 точек
        last_data_query = f'''
        from(bucket:"{INFLUXDB_BUCKET_NAME}")
            |> range(start: {start_query_time})
            |> filter(fn: (r) => r["_measurement"] == "{INFLUXDB_MEASUREMENT_NAME}")
            |> filter(fn: (r) => r["_field"] == "Close" or r["_field"] == "Open" or r["_field"] == "Low" or r["_field"] == "High" or r["_field"] == "Volume")
            |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"], desc: false)
        '''
        
        last_data_df = query_api.query_data_frame(last_data_query)
        if not last_data_df.empty:
            last_data_df = last_data_df[["_time", "Open", "High", "Low", "Close", "Volume"]]
            last_data_df = last_data_df.rename(columns={"_time": "Date"})
            last_data_df['Date'] = pd.to_datetime(last_data_df['Date'])
        else:
            raise Exception("При получении последних точек, ничего не было получено")
        
        # Получение текущих данных из XCom
        ti = kwargs['ti']
        current_data_json = ti.xcom_pull(key='current_data', task_ids='fetch_okx_data')
        current_data = pd.read_json(current_data_json, orient='split')
        
        # Объединение данных
        #df_for_calculations =pd.concat([last_data_df, current_data.iloc[0].to_frame().T], axis=0).reset_index(drop=True)
        df_for_calculations = pd.concat([last_data_df, current_data], axis=0).reset_index(drop=True)
        kwargs['ti'].xcom_push(key='combined_data', value=df_for_calculations.to_json(date_format='iso', orient='split'))
        logging.info("Данные успешно загружены и объединены")
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных из InfluxDB: {e}")
        raise

# Функция для добавления новых фичей на основе Close данных
def add_features(**kwargs):
    logging.info("Добавление новых фичей на основе Close данных")
    try:
        # Получение объединенных данных из XCom
        ti = kwargs['ti']
        combined_data_json = ti.xcom_pull(key='combined_data', task_ids='load_last_data_from_influxdb')
        df = pd.read_json(combined_data_json, orient='split')
        
        with pd.option_context('mode.chained_assignment', None):
            df["Close_kalman_filter"] = apply_kalman_filtering(df["Close"])
            df["Close_savitzky_golay_filter"] = apply_savitzky_golay_filtering(df["Close"])
            df["Close_wavelet_filter"] = apply_wavelet_filtering(df["Close"])
            df["Close_gaussian_smoothing"] = apply_gaussian_smoothing(df["Close"])
            df["Close_sma_smoothing"] = apply_sma_smoothing(df["Close"])
            df["Close_exponential_smoothing"] = apply_exponential_smoothing(df["Close"])
        
        # Сохранение промежуточных данных в XCom
        kwargs['ti'].xcom_push(key='features_data', value=df.to_json(date_format='iso', orient='split'))
        logging.info("Добавление фичей завершено")
    except Exception as e:
        logging.error(f"Ошибка при добавлении фичей: {e}")
        raise

# Функция для создания технических индикаторов на основе OHLCV данных
def add_signal(**kwargs):
    logging.info("Создание технических индикаторов на основе OHLCV данных")
    try:
        # Получение данных с добавленными фичами из XCom
        ti = kwargs['ti']
        combined_data_json = ti.xcom_pull(key='combined_data', task_ids='load_last_data_from_influxdb')
        df = pd.read_json(combined_data_json, orient='split')
        
        # Создание индикаторов теханализа
        # RSI
        rsi_signals = rsi_signal(df)
        df['RSI_buy_signal'] = rsi_signals[0]
        df['RSI_sell_signal'] = rsi_signals[1]

        # Bollinger Bands
        bbands_signals = bbands_signal(df)
        df['BBANDS_buy_signal'] = bbands_signals[0]
        df['BBANDS_sell_signal'] = bbands_signals[1]

        # MACD
        macd_signals = macd_signal(df)
        df['MACD_buy_signal'] = macd_signals[0]
        df['MACD_sell_signal'] = macd_signals[1]

        # PPO
        ppo_signals = ppo_signal(df)
        df['PPO_buy_signal'] = ppo_signals[0]
        df['PPO_sell_signal'] = ppo_signals[1]

        # ADX
        adx_signals = adx_signal(df)
        df['ADX_buy_signal'] = adx_signals[0]
        df['ADX_sell_signal'] = adx_signals[1]

        # Сохранение обработанных данных в XCom
        kwargs['ti'].xcom_push(key='signals_data', value=df.to_json(date_format='iso', orient='split'))
        logging.info("Создание сигналов на основе индикаторов ТА завершено")
    except Exception as e:
        logging.error(f"Ошибка при создании индикаторов: {e}")
        raise

# Функция для сохранения данных в InfluxDB
def save_to_influxdb(**kwargs):
    logging.info("Сохранение данных в InfluxDB")
    try:
        ti = kwargs['ti']
        # Получение данных из XCom
        features_data_json = ti.xcom_pull(key='features_data', task_ids='add_features')
        signals_data_json = ti.xcom_pull(key='signals_data', task_ids='add_signal')
        
        if not features_data_json or not signals_data_json:
            raise ValueError("Отсутствуют необходимые данные для сохранения")
        
        # Преобразование JSON обратно в DataFrame
        features_df = pd.read_json(features_data_json, orient='split')
        signals_df = pd.read_json(signals_data_json, orient='split')
        
        # Объединение DataFrame по времени (предполагается, что даты совпадают)
        combined_df = pd.merge(features_df, signals_df, on='Date', how='inner')

        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)

        write_api.write(
            bucket=INFLUXDB_BUCKET_NAME,
            record=combined_df.iloc[-1:],
            data_frame_measurement_name=INFLUXDB_MEASUREMENT_NAME,
            data_frame_timestamp_column="Date",
        )

        client.close()

        logging.info("Данные успешно сохранены в InfluxDB")
    except Exception as e:
        logging.error(f"Ошибка при сохранении в InfluxDB: {e}")
        raise
    finally:
        client.close()



with DAG(
    'btc_usdt_etl_dag',
    default_args=default_args,
    description='ETL процесс для получения новостей за последний час, их анализа и сохранения в Qdrant',
    schedule_interval=timedelta(hours=1),  # Запуск каждый час
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Определение задач
    fetch_data = PythonOperator(
        task_id='fetch_okx_data',
        python_callable=fetch_okx_data,
        provide_context=True,
    )
    
    load_data = PythonOperator(
        task_id='load_last_data_from_influxdb',
        python_callable=load_last_data_from_influxdb,
        provide_context=True,
    )
    
    add_features_task = PythonOperator(
        task_id='add_features',
        python_callable=add_features,
        provide_context=True,
    )
    
    add_signal_task = PythonOperator(
        task_id='add_signal',
        python_callable=add_signal,
        provide_context=True,
    )
    
    save_data = PythonOperator(
        task_id='save_to_influxdb',
        python_callable=save_to_influxdb,
        provide_context=True,
    )

    # Определение последовательности задач
    fetch_data >> load_data >> [add_features_task, add_signal_task] >> save_data