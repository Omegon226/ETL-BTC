import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.api import SimpleExpSmoothing


def apply_gaussian_smoothing(df, window_size=20, sigma=None):
    df = df.to_frame().copy()
    y = df.iloc[:, 0].values

    # Если sigma не задан, устанавливаем его как window_size / 6
    if sigma is None:
        sigma = window_size / 6.0

    smoothed = gaussian_filter1d(y, sigma=sigma)
    return smoothed


def apply_sma_smoothing(df, window=30):
    df = df.to_frame().copy()

    smoothed = df.rolling(window=window, min_periods=1).mean()

    return smoothed


def apply_exponential_smoothing(df, alpha=0.2):
    data = df.copy()

    ses_model = SimpleExpSmoothing(data).fit(smoothing_level=alpha, optimized=False)
    smoothed = ses_model.fittedvalues

    return smoothed
