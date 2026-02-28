"""Demand Forecasting Engine — Streamlit Dashboard.

Three forecasting methods on Vancouver 311 service request data.
Walk-forward validation and uncertainty quantification.
"""

import streamlit as st

st.set_page_config(
    page_title="Demand Forecasting Engine",
    page_icon=None,
    layout="wide",
)

from dashboard.styles import CUSTOM_CSS
from engine.data_loader import load_data

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="main-header">'
    "<h1>Demand Forecasting Engine</h1>"
    "<p>Vancouver 311 Service Requests | SARIMA, Holt-Winters, XGBoost | "
    "Walk-forward validation with prediction intervals</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Load data
df = load_data()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Decomposition",
    "Forecasts",
    "Model Comparison",
])

with tab1:
    from dashboard.overview import render as render_overview
    render_overview(df)

with tab2:
    from dashboard.decomposition import render as render_decomposition
    render_decomposition(df)

with tab3:
    from dashboard.models import render as render_models
    render_models(df)

with tab4:
    from dashboard.comparison import render as render_comparison
    render_comparison(df)
