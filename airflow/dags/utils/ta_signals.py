import numpy as np
import pandas as pd
import talib


def rsi_signal(ohlcv, time_period=14, overbought=70, oversold=30):
    close = ohlcv['Close']
    rsi = talib.RSI(close, timeperiod=time_period)

    # Определение сигналов
    buy_signals = (rsi.shift(1) < oversold) & (rsi > oversold)
    sell_signals = (rsi.shift(1) > overbought) & (rsi < overbought)

    return buy_signals, sell_signals


def bbands_signal(ohlcv, time_period=20, nbdevup=2, nbdevdn=2, matype=0):
    close = ohlcv['Close']
    upper, middle, lower = talib.BBANDS(close, timeperiod=time_period, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    # Определение сигналов
    buy_signals = (close.shift(1) > lower.shift(1)) & (close < lower)
    sell_signals = (close.shift(1) < upper.shift(1)) & (close > upper)

    return buy_signals, sell_signals


def macd_signal(ohlcv, fastperiod=12, slowperiod=26, signalperiod=9):
    close = ohlcv['Close']
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod,
                                            signalperiod=signalperiod)

    # Определение сигналов
    buy_signals = (macd.shift(1) < macdsignal.shift(1)) & (macd > macdsignal)
    sell_signals = (macd.shift(1) > macdsignal.shift(1)) & (macd < macdsignal)

    return buy_signals, sell_signals


def ppo_signal(ohlcv, fastperiod=12, slowperiod=26, signal_period=9, matype=0):
    close = ohlcv['Close']

    # Рассчитываем PPO
    ppo = talib.PPO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    # Рассчитываем сигнальную линию PPO как EMA от PPO
    ppo_signal = talib.EMA(ppo, timeperiod=signal_period)

    # Определяем сигналы покупки и продажи
    buy_signals = (ppo.shift(1) < ppo_signal.shift(1)) & (ppo > ppo_signal)
    sell_signals = (ppo.shift(1) > ppo_signal.shift(1)) & (ppo < ppo_signal)

    return buy_signals, sell_signals


def adx_signal(ohlcv, time_period=14, adx_threshold=25):
    high = ohlcv['High']
    low = ohlcv['Low']
    close = ohlcv['Close']

    adx = talib.ADX(high, low, close, timeperiod=time_period)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=time_period)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=time_period)

    # Определение сигналов
    buy_signals = (plus_di.shift(1) < minus_di.shift(1)) & (plus_di > minus_di) & (adx > adx_threshold)
    sell_signals = (minus_di.shift(1) < plus_di.shift(1)) & (minus_di > plus_di) & (adx > adx_threshold)

    return buy_signals, sell_signals
