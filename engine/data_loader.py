"""Load Vancouver 311 Service Requests and aggregate to daily counts."""

import os
import pandas as pd
import requests
import streamlit as st


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "van311_daily.parquet")

API_BASE = "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/3-1-1-service-requests/records"
BATCH_SIZE = 100


@st.cache_data(show_spinner="Loading Vancouver 311 data...")
def load_data() -> pd.DataFrame:
    """Load daily 311 request counts. Downloads from API if not cached."""
    if os.path.exists(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)

    df = _download_and_aggregate()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=True)
    return df


def _download_and_aggregate() -> pd.DataFrame:
    """Download 311 data from Vancouver Open Data API in batches."""
    all_records = []
    offset = 0
    max_records = 200_000  # Cap to keep within memory/time limits

    while offset < max_records:
        params = {
            "limit": BATCH_SIZE,
            "offset": offset,
            "order_by": "service_request_open_timestamp asc",
            "select": "service_request_open_timestamp,service_request_type,department",
        }
        resp = requests.get(API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        all_records.extend(results)
        offset += BATCH_SIZE

        if len(results) < BATCH_SIZE:
            break

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["service_request_open_timestamp"]).dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate to daily counts
    daily = df.groupby("date").size().reset_index(name="request_count")
    daily = daily.set_index("date").sort_index()

    # Fill missing dates
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    daily.index.name = "date"

    return daily


@st.cache_data(show_spinner="Loading category breakdown...")
def load_categories() -> pd.DataFrame:
    """Load category breakdown from cached data."""
    if not os.path.exists(CACHE_PATH):
        return pd.DataFrame()

    # Re-download a small sample for category info
    params = {
        "limit": 100,
        "select": "service_request_type,department",
        "group_by": "service_request_type",
    }
    try:
        resp = requests.get(API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        return pd.DataFrame(resp.json().get("results", []))
    except Exception:
        return pd.DataFrame()
