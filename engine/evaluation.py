"""Walk-forward validation and model comparison."""

import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zeros."""
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAE, RMSE, MAPE for a forecast."""
    # Align on common dates
    common = actual.index.intersection(predicted.index)
    a = actual.loc[common].values.astype(float)
    p = predicted.loc[common].values.astype(float)

    return {
        "mae": mean_absolute_error(a, p),
        "rmse": np.sqrt(mean_squared_error(a, p)),
        "mape": mape(a, p),
        "n_points": len(common),
    }


def walk_forward_validation(
    series: pd.Series,
    fit_fn,
    forecast_fn,
    n_splits: int = 3,
    test_size: int = 30,
) -> list[dict]:
    """Walk-forward cross-validation.

    fit_fn(train_series) -> fit_result
    forecast_fn(fit_result, steps) -> DataFrame with 'forecast' column
    """
    results = []
    total_len = len(series)

    for i in range(n_splits):
        test_end = total_len - i * test_size
        test_start = test_end - test_size
        if test_start < 60:  # Need at least 60 days for training
            break

        train = series.iloc[:test_start]
        test = series.iloc[test_start:test_end]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_result = fit_fn(train)
            forecast_df = forecast_fn(fit_result, test_size)

        # Align forecast dates with test dates
        forecast_series = forecast_df["forecast"]
        forecast_series.index = test.index[:len(forecast_series)]

        metrics = evaluate_forecast(test, forecast_series)
        metrics["fold"] = i + 1
        metrics["train_end"] = train.index[-1]
        metrics["test_start"] = test.index[0]
        metrics["test_end"] = test.index[-1]
        results.append(metrics)

    return results
