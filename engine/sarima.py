"""SARIMA model fitting and forecasting."""

import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings


@st.cache_data(show_spinner="Fitting SARIMA model...")
def fit_sarima(
    train: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7),
) -> dict:
    """Fit SARIMA and return model results."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=200)

    return {
        "results": results,
        "aic": results.aic,
        "bic": results.bic,
        "order": order,
        "seasonal_order": seasonal_order,
    }


def forecast_sarima(
    fit_result: dict, steps: int, alpha: float = 0.05,
) -> pd.DataFrame:
    """Generate forecast with prediction intervals."""
    results = fit_result["results"]
    forecast = results.get_forecast(steps=steps)
    mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=alpha)

    out = pd.DataFrame({
        "forecast": mean,
        "lower": ci.iloc[:, 0],
        "upper": ci.iloc[:, 1],
    })
    return out
