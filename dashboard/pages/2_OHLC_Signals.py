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
    # OHLC данные с сигналами
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

required_columns = ['_time', 'Open', 'High', 'Low', 'Close']
signal_columns = [
    'RSI_buy_signal', 'RSI_sell_signal',
    'BBANDS_buy_signal', 'BBANDS_sell_signal',
    'MACD_buy_signal', 'MACD_sell_signal',
    'PPO_buy_signal', 'PPO_sell_signal',
    'ADX_buy_signal', 'ADX_sell_signal'
]

# Определение сигналов покупки и продажи
buy_signals = {
    'RSI Покупка': 'RSI_buy_signal',
    'BBANDS Покупка': 'BBANDS_buy_signal',
    'MACD Покупка': 'MACD_buy_signal',
    'PPO Покупка': 'PPO_buy_signal',
    'ADX Покупка': 'ADX_buy_signal'
}

sell_signals = {
    'RSI Продажа': 'RSI_sell_signal',
    'BBANDS Продажа': 'BBANDS_sell_signal',
    'MACD Продажа': 'MACD_sell_signal',
    'PPO Продажа': 'PPO_sell_signal',
    'ADX Продажа': 'ADX_sell_signal'
}

# Создание фигуры Plotly с OHLC
fig = go.Figure()

fig.add_trace(go.Ohlc(
    x=df['_time'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='OHLC',
    increasing_line_color='green',
    decreasing_line_color='red'
))


# Функция для добавления сигналов
def add_signals(fig, signals, signal_type='Покупка'):
    for label, column in signals.items():
        signal_points = df[df[column] == True]
        if signal_type == 'Покупка':
            symbol = 'triangle-up'
            color = 'green'
            y = signal_points['Open'] - signal_points['Open'] * 0.01
        else:
            symbol = 'triangle-down'
            color = 'red'
            y = signal_points['Close'] + signal_points['Close'] * 0.01

        fig.add_trace(go.Scatter(
            x=signal_points['_time'],
            y=y,
            mode='markers',
            marker=dict(
                symbol=symbol,
                color=color,
                size=12,
                line=dict(width=1, color='Black')
            ),
            name=label,
            text=label,
            hoverinfo='text'
        ))


# Добавление сигналов покупки
add_signals(fig, buy_signals, signal_type='Покупка')

# Добавление сигналов продажи
add_signals(fig, sell_signals, signal_type='Продажа')

# Настройка макета
fig.update_layout(
    title="OHLC График с Сигналами Покупки и Продажи",
    xaxis_title="Время",
    yaxis_title="Цена",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0)')
)

# Отображение графика
st.plotly_chart(fig, use_container_width=True)