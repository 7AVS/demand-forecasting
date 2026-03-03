"""Time series EDA — stationarity tests, ACF/PACF, decomposition."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL


def adf_test(series: pd.Series) -> dict:
    """Run Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def compute_acf_pacf(series: pd.Series, nlags: int = 40) -> tuple:
    """Compute ACF and PACF values with confidence intervals."""
    acf_vals, acf_ci = acf(series.dropna(), nlags=nlags, alpha=0.05)
    pacf_vals, pacf_ci = pacf(series.dropna(), nlags=nlags, alpha=0.05)
    return acf_vals, acf_ci, pacf_vals, pacf_ci


def stl_decompose(series: pd.Series, period: int = 7) -> dict:
    """STL decomposition into trend, seasonal, and residual."""
    stl = STL(series.dropna(), period=period, robust=True)
    result = stl.fit()
    return {
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": result.resid,
        "observed": series.dropna(),
    }
