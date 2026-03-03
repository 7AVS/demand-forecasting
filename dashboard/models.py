"""Tab 3: Forecast visualization — all models with prediction intervals."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from engine.sarima import fit_sarima, forecast_sarima
from engine.holt_winters import fit_holt_winters, forecast_holt_winters
from engine.ml_forecast import fit_xgboost, forecast_xgboost
from dashboard.styles import COLORS, CHART_COLORS, PLOTLY_LAYOUT, AXIS_DEFAULTS, CUSTOM_CSS


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert hex color to rgba string for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render(df: pd.DataFrame):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    series = df["request_count"]

    st.markdown('<div class="section-header">Forecast Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        horizon = st.slider("Forecast horizon (days)", min_value=14, max_value=90, value=30, step=7)
    with col2:
        holdout = st.slider("Holdout days (for visual comparison)", min_value=0, max_value=60, value=30, step=7)

    # Split
    if holdout > 0:
        train = series.iloc[:-holdout]
        test = series.iloc[-holdout:]
    else:
        train = series
        test = pd.Series(dtype=float)

    st.markdown("---")

    # Fit all models
    sarima_fit = fit_sarima(train)
    hw_fit = fit_holt_winters(train)
    xgb_fit = fit_xgboost(train)

    # Forecast
    sarima_fc = forecast_sarima(sarima_fit, horizon)
    hw_fc = forecast_holt_winters(hw_fit, horizon, train)
    xgb_fc = forecast_xgboost(xgb_fit, horizon)

    forecasts = {
        "SARIMA": (sarima_fc, CHART_COLORS[0]),
        "Holt-Winters": (hw_fc, CHART_COLORS[1]),
        "XGBoost": (xgb_fc, CHART_COLORS[2]),
    }

    # Plot each model
    for name, (fc, color) in forecasts.items():
        st.markdown(f'<div class="section-header">{name} Forecast</div>', unsafe_allow_html=True)

        fig = go.Figure()

        # Training data (last 90 days for context)
        context_days = min(90, len(train))
        train_context = train.iloc[-context_days:]

        fig.add_trace(go.Scatter(
            x=train_context.index, y=train_context.values,
            mode="lines", name="Historical",
            line=dict(color=COLORS["slate"], width=1.5),
        ))

        # Holdout actuals
        if len(test) > 0:
            fig.add_trace(go.Scatter(
                x=test.index, y=test.values,
                mode="lines", name="Actual (holdout)",
                line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
            ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc["forecast"],
            mode="lines", name=f"{name} forecast",
            line=dict(color=color, width=2),
        ))

        # Prediction interval
        fig.add_trace(go.Scatter(
            x=list(fc.index) + list(fc.index[::-1]),
            y=list(fc["upper"]) + list(fc["lower"][::-1]),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.15),
            line=dict(width=0),
            name="95% prediction interval",
            showlegend=True,
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT, height=350,
            xaxis=dict(**AXIS_DEFAULTS, title="Date"),
            yaxis=dict(**AXIS_DEFAULTS, title="Daily requests"),
            legend=dict(orientation="h", y=-0.2, x=0),
        )
        st.plotly_chart(fig, width="stretch")

    # Combined overlay
    st.markdown('<div class="section-header">All Models — Combined View</div>', unsafe_allow_html=True)

    fig_all = go.Figure()

    context_days = min(60, len(train))
    train_context = train.iloc[-context_days:]
    fig_all.add_trace(go.Scatter(
        x=train_context.index, y=train_context.values,
        mode="lines", name="Historical",
        line=dict(color=COLORS["slate"], width=1.5),
    ))

    if len(test) > 0:
        fig_all.add_trace(go.Scatter(
            x=test.index, y=test.values,
            mode="lines", name="Actual",
            line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
        ))

    for name, (fc, color) in forecasts.items():
        fig_all.add_trace(go.Scatter(
            x=fc.index, y=fc["forecast"],
            mode="lines", name=name,
            line=dict(color=color, width=2),
        ))

    fig_all.update_layout(
        **PLOTLY_LAYOUT, height=400,
        xaxis=dict(**AXIS_DEFAULTS, title="Date"),
        yaxis=dict(**AXIS_DEFAULTS, title="Daily requests"),
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    st.plotly_chart(fig_all, width="stretch")
