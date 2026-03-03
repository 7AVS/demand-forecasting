"""Tab 1: Time series overview — full history, KPIs, summary stats."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dashboard.styles import COLORS, PLOTLY_LAYOUT, AXIS_DEFAULTS, CUSTOM_CSS


def render(df: pd.DataFrame):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # KPI row
    total_requests = int(df["request_count"].sum())
    n_days = len(df)
    daily_avg = df["request_count"].mean()
    daily_max = int(df["request_count"].max())
    date_range = f"{df.index.min().strftime('%b %Y')} — {df.index.max().strftime('%b %Y')}"

    cols = st.columns(5)
    for col, label, value in zip(
        cols,
        ["Total Requests", "Days", "Daily Average", "Peak Day", "Date Range"],
        [f"{total_requests:,}", f"{n_days:,}", f"{daily_avg:.0f}", f"{daily_max:,}", date_range],
    ):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Full time series plot
    st.markdown('<div class="section-header">Daily 311 Service Requests</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["request_count"],
        mode="lines", name="Daily requests",
        line=dict(color=COLORS["slate"], width=1),
    ))

    # 7-day rolling average
    rolling = df["request_count"].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=rolling,
        mode="lines", name="7-day average",
        line=dict(color=COLORS["teal"], width=2),
    ))

    # 30-day rolling average
    rolling30 = df["request_count"].rolling(30).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=rolling30,
        mode="lines", name="30-day average",
        line=dict(color=COLORS["taupe"], width=2, dash="dash"),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT, height=400,
        xaxis=dict(**AXIS_DEFAULTS, title="Date"),
        yaxis=dict(**AXIS_DEFAULTS, title="Requests per day"),
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    # Day-of-week pattern
    st.markdown('<div class="section-header">Day-of-Week Pattern</div>', unsafe_allow_html=True)

    dow = df.copy()
    dow["dayofweek"] = dow.index.dayofweek
    dow_avg = dow.groupby("dayofweek")["request_count"].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig_dow = go.Figure(go.Bar(
        x=day_names,
        y=dow_avg.values,
        marker_color=[COLORS["slate"]] * 5 + [COLORS["teal"]] * 2,
        text=[f"{v:.0f}" for v in dow_avg.values],
        textposition="outside",
    ))
    fig_dow.update_layout(
        **PLOTLY_LAYOUT, height=300,
        xaxis=dict(**AXIS_DEFAULTS),
        yaxis=dict(**AXIS_DEFAULTS, title="Average daily requests"),
    )
    st.plotly_chart(fig_dow, width="stretch")

    # Monthly pattern
    st.markdown('<div class="section-header">Monthly Pattern</div>', unsafe_allow_html=True)

    monthly = df.resample("MS")["request_count"].sum()
    fig_monthly = go.Figure(go.Bar(
        x=monthly.index,
        y=monthly.values,
        marker_color=COLORS["slate"],
    ))
    fig_monthly.update_layout(
        **PLOTLY_LAYOUT, height=300,
        xaxis=dict(**AXIS_DEFAULTS, title="Month"),
        yaxis=dict(**AXIS_DEFAULTS, title="Total requests"),
    )
    st.plotly_chart(fig_monthly, width="stretch")
