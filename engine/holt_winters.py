"""Holt-Winters (Exponential Smoothing) model."""

import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings


@st.cache_data(show_spinner="Fitting Holt-Winters model...")
def fit_holt_winters(
    train: pd.Series,
    seasonal_periods: int = 7,
    trend: str = "add",
    seasonal: str = "add",
) -> dict:
    """Fit Holt-Winters Exponential Smoothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        results = model.fit(optimized=True)

    return {
        "results": results,
        "aic": results.aic,
        "seasonal_periods": seasonal_periods,
    }


def forecast_holt_winters(
    fit_result: dict, steps: int, train: pd.Series,
) -> pd.DataFrame:
    """Generate forecast. HW doesn't natively produce intervals, so we bootstrap."""
    results = fit_result["results"]
    forecast = results.forecast(steps)

    # Simple residual-based prediction interval
    residuals = results.resid.dropna()
    std = residuals.std()

    last_date = train.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="D")
    forecast.index = future_dates

    out = pd.DataFrame({
        "forecast": forecast.values,
        "lower": forecast.values - 1.96 * std,
        "upper": forecast.values + 1.96 * std,
    }, index=future_dates)

    # Floor at zero
    out["lower"] = out["lower"].clip(lower=0)
    out["forecast"] = out["forecast"].clip(lower=0)

    return out
