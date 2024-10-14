import streamlit as st
import os
from influxdb_client import InfluxDBClient
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from styles.page_style import set_page_config_wide, disable_header_and_footer


# Настройка страницы
set_page_config_wide()
disable_header_and_footer()


INFLUXDB_URL = "http://influxdb:8086"
#INFLUXDB_URL = "http://localhost:8010"
INFLUXDB_TOKEN = os.getenv('INFLUXDB_INIT_ADMIN_TOKEN')
INFLUXDB_ORG = os.getenv('INFLUXDB_INIT_ORG')
INFLUXDB_BUCKET_NAME = 'btc-usdt-bucket'
INFLUXDB_MEASUREMENT_NAME = '1h-timeframe'


st.markdown(
    """
    # Цены закрытия BTC-USDT
    """
)

user_input = st.number_input("Введите кол-во точек данных для визуализации:", min_value=30, max_value=3000)

last_data_query = f'''
from(bucket:"{INFLUXDB_BUCKET_NAME}")
    |> range(start: -{user_input}h)
    |> filter(fn: (r) => r["_measurement"] == "{INFLUXDB_MEASUREMENT_NAME}")
    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> sort(columns: ["_time"], desc: false)
'''

query_api = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG).query_api()
last_data_df = query_api.query_data_frame(last_data_query)

last_data_df['_time'] = pd.to_datetime(last_data_df['_time'])
df = last_data_df

# Создание фигуры Plotly
fig = go.Figure()

# Добавление трасс для каждого столбца
fig.add_trace(go.Scatter(x=df['_time'], y=df['Close'],
                         mode='lines',
                         name='Close'))
fig.add_trace(go.Scatter(x=df['_time'], y=df['Close_kalman_filter'],
                         mode='lines',
                         name='Kalman Filter'))
fig.add_trace(go.Scatter(x=df['_time'], y=df['Close_savitzky_golay_filter'],
                         mode='lines',
                         name='Savitzky-Golay Filter'))
fig.add_trace(go.Scatter(x=df['_time'], y=df['Close_wavelet_filter'],
                         mode='lines',
                         name='Wavelet Filter'))
fig.add_trace(go.Scatter(x=df['_time'], y=df['Close_gaussian_smoothing'],
                         mode='lines',
                         name='Gaussian Smoothing'))
fig.add_trace(go.Scatter(x=df['_time'], y=df['Close_sma_smoothing'],
                         mode='lines',
                         name='SMA Smoothing'))
fig.add_trace(go.Scatter(x=df['_time'], y=df['Close_exponential_smoothing'],
                         mode='lines',
                         name='Exponential Smoothing'))

# Настройка макета
fig.update_layout(
    title="Сравнение различных фильтров для цен Close",
    xaxis_title="Время",
    yaxis_title="Цена Close",
    hovermode="x unified"
)

# Отображение графика
st.plotly_chart(fig, use_container_width=True)
