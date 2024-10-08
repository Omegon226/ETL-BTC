import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
import pywt


def apply_kalman_filtering(df, q=1e-4, r=1e-3):
    data = df.to_numpy()
    df = df.to_frame().copy()

    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([data[0]])
    kf.P = np.array([[1.]])
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    kf.R = r
    kf.Q = q

    filtered = []

    for z in data:
        kf.predict()
        kf.update(z)
        filtered.append(kf.x[0])

    return filtered


def apply_savitzky_golay_filtering(df, win=21, poly=4):
    close_prices = df.to_numpy()
    df = df.to_frame().copy()
    filtered = savgol_filter(close_prices, window_length=win, polyorder=poly)

    return filtered


def apply_wavelet_filtering(df, wavelet_name='sym4', level=2):
    # Преобразуем данные в массив NumPy и выпрямляем его
    data = df.to_numpy().flatten()
    # Создаём объект вейвлета
    wavelet = pywt.Wavelet(wavelet_name)

    # Определяем максимальный возможный уровень декомпозиции
    max_level = pywt.dwt_max_level(data_len=len(data), filter_len=wavelet.dec_len)
    if level > max_level:
        raise ValueError(
            f"Уровень декомпозиции {level} превышает максимальный {max_level} для вейвлета '{wavelet_name}'."
        )

    # Выполняем вейвлет-декомпозицию
    coeffs = pywt.wavedec(data=data, wavelet=wavelet_name, level=level)
    # Фильтруем детальные коэффициенты, установив их в ноль
    coeffs_filtered = [coeffs[0]] + [np.zeros_like(detail) for detail in coeffs[1:]]
    # Реконструируем сигнал из отфильтрованных коэффициентов
    reconstructed_signal = pywt.waverec(coeffs_filtered, wavelet_name)
    # Обрезаем сигнал до исходной длины
    filtered = reconstructed_signal[:len(data)]

    return filtered
