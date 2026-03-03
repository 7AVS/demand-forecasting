# Demand Forecasting Engine

Three forecasting methods applied to real Vancouver 311 service request data. SARIMA, Holt-Winters, and XGBoost compared using walk-forward cross-validation with uncertainty quantification.

## What This Demonstrates

- **Time series analysis** — trend detection, seasonality decomposition, stationarity testing
- **Multiple forecasting methods** — SARIMA, Holt-Winters Exponential Smoothing, XGBoost with lag features
- **Walk-forward validation** — proper time series cross-validation, not random train/test splits
- **Prediction intervals** — uncertainty is part of every forecast, not just point estimates
- **Real public data** — Vancouver 311 service requests via Open Data API

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The Vancouver 311 dataset is downloaded automatically from the Open Data API on first run and cached locally.

## Architecture

```
engine/
  data_loader.py     — Download + cache Vancouver 311 data, aggregate to daily counts
  eda.py             — ADF stationarity test, ACF/PACF, STL decomposition
  sarima.py          — SARIMA model fitting and forecasting
  holt_winters.py    — Holt-Winters Exponential Smoothing
  ml_forecast.py     — XGBoost with lag + calendar features
  evaluation.py      — Walk-forward validation, MAPE/RMSE/MAE

dashboard/
  styles.py          — Shared design tokens (Desaturated Cool palette)
  overview.py        — Full time series, KPIs, day-of-week + monthly patterns
  decomposition.py   — STL decomposition, ACF/PACF, stationarity tests
  models.py          — Individual model forecasts with prediction intervals
  comparison.py      — Walk-forward validation, model comparison metrics
```

## Dataset

Vancouver 311 Service Requests from Vancouver Open Data (opendata.vancouver.ca). ~897K records from Aug 2022 to present, aggregated to daily request counts.

## Tech Stack

Python, statsmodels, XGBoost, Streamlit, Plotly
