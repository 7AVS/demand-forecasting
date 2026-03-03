"""Tab 2: STL decomposition — trend, seasonal, residual."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from engine.eda import stl_decompose, adf_test, compute_acf_pacf
from dashboard.styles import COLORS, PLOTLY_LAYOUT, AXIS_DEFAULTS, CUSTOM_CSS


def render(df: pd.DataFrame):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    series = df["request_count"]

    # Stationarity test
    st.markdown('<div class="section-header">Stationarity Test (Augmented Dickey-Fuller)</div>', unsafe_allow_html=True)

    adf = adf_test(series)
    cols = st.columns(4)
    cols[0].metric("ADF Statistic", f"{adf['statistic']:.2f}")
    cols[1].metric("p-value", f"{adf['p_value']:.4f}")
    cols[2].metric("Lags Used", str(adf["lags_used"]))
    cols[3].metric("Stationary?", "Yes" if adf["is_stationary"] else "No")

    if adf["is_stationary"]:
        st.caption("The series is stationary (p < 0.05). No differencing required for modeling.")
    else:
        st.caption("The series is non-stationary. Differencing will be applied in SARIMA.")

    st.markdown("---")

    # STL Decomposition
    st.markdown('<div class="section-header">STL Decomposition (period=7 days)</div>', unsafe_allow_html=True)
    st.caption("Seasonal-Trend decomposition using LOESS separates the series into trend, "
               "weekly seasonal pattern, and residual components.")

    decomp = stl_decompose(series, period=7)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=["Observed", "Trend", "Seasonal (Weekly)", "Residual"],
        vertical_spacing=0.06,
    )

    components = [
        (decomp["observed"], COLORS["slate"]),
        (decomp["trend"], COLORS["teal"]),
        (decomp["seasonal"], COLORS["taupe"]),
        (decomp["resid"], COLORS["rose"]),
    ]

    for i, (comp, color) in enumerate(components, 1):
        fig.add_trace(go.Scatter(
            x=comp.index, y=comp.values,
            mode="lines", line=dict(color=color, width=1),
            showlegend=False,
        ), row=i, col=1)
        fig.update_yaxes(**AXIS_DEFAULTS, row=i, col=1)

    fig.update_layout(**PLOTLY_LAYOUT, height=700)
    fig.update_xaxes(**AXIS_DEFAULTS, row=4, col=1)
    st.plotly_chart(fig, width="stretch")

    # ACF / PACF
    st.markdown('<div class="section-header">Autocorrelation (ACF) and Partial Autocorrelation (PACF)</div>',
                unsafe_allow_html=True)
    st.caption("ACF and PACF plots help determine the order parameters for SARIMA. "
               "Significant spikes at lag 7, 14, 21 confirm weekly seasonality.")

    acf_vals, acf_ci, pacf_vals, pacf_ci = compute_acf_pacf(series, nlags=35)

    fig_acf = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])

    lags = list(range(len(acf_vals)))

    # ACF
    for lag, val in zip(lags, acf_vals):
        fig_acf.add_trace(go.Scatter(
            x=[lag, lag], y=[0, val],
            mode="lines", line=dict(color=COLORS["slate"], width=2),
            showlegend=False,
        ), row=1, col=1)

    ci_upper = acf_ci[:, 1] - acf_vals
    fig_acf.add_trace(go.Scatter(
        x=lags, y=[1.96 / (len(series) ** 0.5)] * len(lags),
        mode="lines", line=dict(color=COLORS["rose"], width=1, dash="dash"),
        showlegend=False,
    ), row=1, col=1)
    fig_acf.add_trace(go.Scatter(
        x=lags, y=[-1.96 / (len(series) ** 0.5)] * len(lags),
        mode="lines", line=dict(color=COLORS["rose"], width=1, dash="dash"),
        showlegend=False,
    ), row=1, col=1)

    # PACF
    pacf_lags = list(range(len(pacf_vals)))
    for lag, val in zip(pacf_lags, pacf_vals):
        fig_acf.add_trace(go.Scatter(
            x=[lag, lag], y=[0, val],
            mode="lines", line=dict(color=COLORS["teal"], width=2),
            showlegend=False,
        ), row=1, col=2)

    fig_acf.add_trace(go.Scatter(
        x=pacf_lags, y=[1.96 / (len(series) ** 0.5)] * len(pacf_lags),
        mode="lines", line=dict(color=COLORS["rose"], width=1, dash="dash"),
        showlegend=False,
    ), row=1, col=2)
    fig_acf.add_trace(go.Scatter(
        x=pacf_lags, y=[-1.96 / (len(series) ** 0.5)] * len(pacf_lags),
        mode="lines", line=dict(color=COLORS["rose"], width=1, dash="dash"),
        showlegend=False,
    ), row=1, col=2)

    fig_acf.update_layout(**PLOTLY_LAYOUT, height=300)
    fig_acf.update_xaxes(**AXIS_DEFAULTS, title="Lag")
    fig_acf.update_yaxes(**AXIS_DEFAULTS)
    st.plotly_chart(fig_acf, width="stretch")
