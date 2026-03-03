"""XGBoost forecasting with lag and calendar features."""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def create_lag_features(series: pd.Series, lags: list[int] = None) -> pd.DataFrame:
    """Create lag, rolling, and calendar features from a time series."""
    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 28]

    df = pd.DataFrame({"y": series})

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Rolling statistics
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["y"].shift(1).rolling(window).std()

    # Calendar features
    df["dayofweek"] = series.index.dayofweek
    df["month"] = series.index.month
    df["day"] = series.index.day
    df["is_weekend"] = (series.index.dayofweek >= 5).astype(int)

    return df.dropna()


@st.cache_data(show_spinner="Training XGBoost forecasting model...")
def fit_xgboost(train: pd.Series) -> dict:
    """Fit XGBoost on lag features."""
    df = create_lag_features(train)
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols].values
    y = df["y"].values

    model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_series": train,
    }


def forecast_xgboost(fit_result: dict, steps: int) -> pd.DataFrame:
    """Recursive multi-step forecast."""
    model = fit_result["model"]
    feature_cols = fit_result["feature_cols"]
    train = fit_result["train_series"].copy()

    predictions = []
    current_series = train.copy()

    for i in range(steps):
        next_date = current_series.index[-1] + pd.Timedelta(days=1)

        # Create features for the next step
        extended = pd.concat([current_series, pd.Series([np.nan], index=[next_date])])
        df = create_lag_features(extended)

        if len(df) == 0:
            break

        last_row = df.iloc[[-1]]
        X = last_row[feature_cols].values
        pred = max(0, float(model.predict(X)[0]))
        predictions.append({"date": next_date, "forecast": pred})

        current_series = pd.concat([current_series, pd.Series([pred], index=[next_date])])

    out = pd.DataFrame(predictions).set_index("date")

    # Simple prediction interval from training residuals
    df_train = create_lag_features(train)
    X_train = df_train[[c for c in df_train.columns if c != "y"]].values
    y_train = df_train["y"].values
    train_preds = model.predict(X_train)
    residual_std = np.std(y_train - train_preds)

    out["lower"] = (out["forecast"] - 1.96 * residual_std).clip(lower=0)
    out["upper"] = out["forecast"] + 1.96 * residual_std

    return out
