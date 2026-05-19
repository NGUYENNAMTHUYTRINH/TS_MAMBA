"""
utils.py
--------
Shared helpers for the Streamlit app and model pipelines.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_locations(value) -> list[str]:
    """Normalize a location input into list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x and x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    s = str(value).strip()
    return [s] if s else []


def format_time_utc_strings(values: pd.Series) -> pd.Series:
    """Format datetimes as YYYY-MM-DD HH:MM:SS+00:00 in UTC."""
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")


def get_timestamp_col(df: pd.DataFrame) -> str:
    """Return the timestamp column used by the AQI datasets."""
    col_map = {c.lower(): c for c in df.columns}
    ts_col = col_map.get("ts_utc") or col_map.get("time") or col_map.get("timestamp")
    if ts_col is None:
        raise ValueError("Dataset can co cot Time, ts_utc hoac timestamp.")
    return ts_col


def model_run_dir(project_root: Path, model_name: str, timestamp: str) -> str:
    """Build the standard runs/<model>/<timestamp> path."""
    return str(project_root / "runs" / model_name.lower() / timestamp)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON with stable UTF-8 formatting."""
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_train_module():
    """Load mamba/train_mamba_aqi.py dynamically for Streamlit helpers."""
    import sys

    try:
        project_root = Path(__file__).parent.parent
        mod_path = project_root / "mamba" / "train_mamba_aqi.py"
        if not mod_path.exists():
            raise FileNotFoundError(f"Khong tim thay: {mod_path}")

        for path in [project_root, project_root / "mamba"]:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        spec = importlib.util.spec_from_file_location(
            "train_mamba_aqi_for_streamlit", str(mod_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def build_future_24h_frame(
    df_valid: pd.DataFrame, feature_cols: list[str], target_col: str, hours: int = 24
) -> pd.DataFrame:
    """Build the next-day hourly frame used by forecast pipelines."""
    normalized = df_valid.copy()
    ts_col = get_timestamp_col(normalized)
    if ts_col != "ts_utc":
        normalized["ts_utc"] = normalized[ts_col]

    work = normalized.copy()
    work["ts_utc"] = pd.to_datetime(work["ts_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts_utc"]).copy()

    future_rows = []
    if "location_key" in work.columns:
        groups = [
            (loc, work.loc[work["location_key"].astype(str) == loc].sort_values("ts_utc").copy())
            for loc in sorted(work["location_key"].dropna().astype(str).unique().tolist())
        ]
    else:
        groups = [(None, work.sort_values("ts_utc").copy())]

    for loc, group in groups:
        if group.empty:
            continue

        last_ts = group["ts_utc"].iloc[-1]
        next_day_start = last_ts.normalize() + pd.Timedelta(days=1)
        template = group.tail(hours).copy()
        if len(template) < hours:
            template = pd.concat(
                [template] * (hours // len(template) + 1), ignore_index=True
            ).head(hours)

        for hour in range(hours):
            src = template.iloc[hour].copy()
            row = {col: src[col] for col in feature_cols if col in template.columns}
            if loc is not None:
                row["location_key"] = loc
            row["ts_utc"] = next_day_start + pd.Timedelta(hours=hour)
            row[target_col] = np.nan
            future_rows.append(row)

    if not future_rows:
        raise ValueError("Khong tao duoc du lieu du bao.")

    return pd.DataFrame(future_rows)


def split_data_by_timeline(
    x_seq: np.ndarray,
    loc_ids: np.ndarray,
    y: np.ndarray,
    y_ts: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    """Fallback timeline split when the training module cannot be loaded."""
    if len(y) < 3:
        raise ValueError("Need at least 3 samples for train/val/test split.")

    order = np.argsort(y_ts)
    n = len(order)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid timeline split sizes.")

    class _Split:
        def __init__(self, x, l, yy):
            self.x_seq = x
            self.loc_ids = l
            self.y = yy

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]
    return (
        _Split(x_seq[train_idx], loc_ids[train_idx], y[train_idx]),
        _Split(x_seq[val_idx], loc_ids[val_idx], y[val_idx]),
        _Split(x_seq[test_idx], loc_ids[test_idx], y[test_idx]),
    )
