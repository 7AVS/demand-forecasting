"""Tab 4: Model comparison — walk-forward validation, error metrics."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from engine.sarima import fit_sarima, forecast_sarima
from engine.holt_winters import fit_holt_winters, forecast_holt_winters
from engine.ml_forecast import fit_xgboost, forecast_xgboost
from engine.evaluation import walk_forward_validation
from dashboard.styles import COLORS, CHART_COLORS, PLOTLY_LAYOUT, AXIS_DEFAULTS, CUSTOM_CSS


def render(df: pd.DataFrame):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    series = df["request_count"]

    st.markdown('<div class="section-header">Walk-Forward Cross-Validation</div>', unsafe_allow_html=True)
    st.caption(
        "Walk-forward validation slides a test window backward through time. "
        "The training set always ends before the test set starts — no data leakage. "
        "This mimics how the model would be used in production."
    )

    n_folds = st.slider("Number of validation folds", min_value=2, max_value=5, value=3)
    test_days = st.slider("Test window size (days)", min_value=14, max_value=60, value=30, step=7)

    # Run walk-forward for each model
    model_configs = {
        "SARIMA": {
            "fit_fn": fit_sarima,
            "forecast_fn": lambda fit_result, steps: forecast_sarima(fit_result, steps),
        },
        "Holt-Winters": {
            "fit_fn": fit_holt_winters,
            "forecast_fn": lambda fit_result, steps, _fr=None: forecast_holt_winters(
                fit_result, steps, fit_result.get("_train", series)
            ),
        },
        "XGBoost": {
            "fit_fn": fit_xgboost,
            "forecast_fn": lambda fit_result, steps: forecast_xgboost(fit_result, steps),
        },
    }

    # Simplified walk-forward for HW (needs train series)
    def hw_fit_fn(train):
        result = fit_holt_winters(train)
        result["_train"] = train
        return result

    def hw_forecast_fn(fit_result, steps):
        return forecast_holt_winters(fit_result, steps, fit_result["_train"])

    with st.spinner("Running walk-forward validation (this may take a moment)..."):
        all_results = {}

        sarima_wf = walk_forward_validation(
            series, fit_sarima,
            lambda fr, s: forecast_sarima(fr, s),
            n_splits=n_folds, test_size=test_days,
        )
        all_results["SARIMA"] = sarima_wf

        hw_wf = walk_forward_validation(
            series, hw_fit_fn, hw_forecast_fn,
            n_splits=n_folds, test_size=test_days,
        )
        all_results["Holt-Winters"] = hw_wf

        xgb_wf = walk_forward_validation(
            series, fit_xgboost,
            lambda fr, s: forecast_xgboost(fr, s),
            n_splits=n_folds, test_size=test_days,
        )
        all_results["XGBoost"] = xgb_wf

    # Summary table
    st.markdown('<div class="section-header">Model Comparison — Average Across Folds</div>',
                unsafe_allow_html=True)

    summary_rows = []
    for name, wf_results in all_results.items():
        if not wf_results:
            continue
        avg_mae = np.mean([r["mae"] for r in wf_results])
        avg_rmse = np.mean([r["rmse"] for r in wf_results])
        avg_mape = np.mean([r["mape"] for r in wf_results if not np.isnan(r["mape"])])
        summary_rows.append({
            "Model": name,
            "MAE": f"{avg_mae:.1f}",
            "RMSE": f"{avg_rmse:.1f}",
            "MAPE": f"{avg_mape:.1f}%",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    # Bar chart comparison
    metrics_to_plot = ["MAE", "RMSE"]
    for metric in metrics_to_plot:
        values = []
        names = []
        for name, wf_results in all_results.items():
            if not wf_results:
                continue
            key = metric.lower()
            avg_val = np.mean([r[key] for r in wf_results])
            values.append(avg_val)
            names.append(name)

        fig = go.Figure(go.Bar(
            x=names, y=values,
            marker_color=CHART_COLORS[:len(names)],
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=300,
            title=dict(text=f"Average {metric} Across Folds", font=dict(size=14, color=COLORS["dark"])),
            yaxis=dict(**AXIS_DEFAULTS, title=metric),
            xaxis=dict(**AXIS_DEFAULTS),
        )
        st.plotly_chart(fig, width="stretch")

    # Per-fold detail
    st.markdown('<div class="section-header">Per-Fold Results</div>', unsafe_allow_html=True)

    for name, wf_results in all_results.items():
        if not wf_results:
            continue
        st.caption(f"**{name}**")
        fold_df = pd.DataFrame(wf_results)
        fold_df["train_end"] = fold_df["train_end"].astype(str)
        fold_df["test_start"] = fold_df["test_start"].astype(str)
        fold_df["test_end"] = fold_df["test_end"].astype(str)
        fold_df["mae"] = fold_df["mae"].round(1)
        fold_df["rmse"] = fold_df["rmse"].round(1)
        fold_df["mape"] = fold_df["mape"].round(1).astype(str) + "%"
        st.dataframe(
            fold_df[["fold", "train_end", "test_start", "test_end", "mae", "rmse", "mape"]],
            hide_index=True, use_container_width=True,
        )
