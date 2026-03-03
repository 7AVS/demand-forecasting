"""Load Vancouver 311 Service Requests and aggregate to daily counts."""

import os
import io
import pandas as pd
import requests
import streamlit as st


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "van311_daily.parquet")
RAW_CSV_PATH = os.path.join(DATA_DIR, "van311_raw.csv")

CSV_EXPORT_URL = (
    "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/"
    "3-1-1-service-requests/exports/csv?delimiter=%2C&limit=-1"
)


@st.cache_data(show_spinner="Loading Vancouver 311 data...")
def load_data() -> pd.DataFrame:
    """Load daily 311 request counts. Downloads full CSV if not cached."""
    if os.path.exists(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)

    df = _download_and_aggregate()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=True)
    return df


def _download_and_aggregate() -> pd.DataFrame:
    """Download full CSV export from Vancouver Open Data and aggregate to daily counts."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download full CSV in one request
    if not os.path.exists(RAW_CSV_PATH):
        resp = requests.get(CSV_EXPORT_URL, timeout=300, stream=True)
        resp.raise_for_status()
        with open(RAW_CSV_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    # Parse
    df = pd.read_csv(RAW_CSV_PATH, sep=",")

    # Find the timestamp column
    ts_col = None
    for candidate in ["service_request_open_timestamp", "Service Request Open Timestamp"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        # Fall back to first column with 'timestamp' in the name
        for col in df.columns:
            if "timestamp" in col.lower():
                ts_col = col
                break

    if ts_col is None:
        raise ValueError(f"Cannot find timestamp column. Columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df[ts_col], errors="coerce").dt.date
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"])

    # Aggregate to daily counts
    daily = df.groupby("date").size().reset_index(name="request_count")
    daily = daily.set_index("date").sort_index()

    # Fill missing dates
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    daily.index.name = "date"

    return daily
