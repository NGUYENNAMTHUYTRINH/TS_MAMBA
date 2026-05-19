"""
ui_components.py
----------------
Reusable Streamlit UI pieces for loading data, configuring Mamba, and showing
prediction/training outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from Utils import load_train_module, split_data_by_timeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_TEMP_DIR = PROJECT_ROOT / "runs" / "uploaded"


def _fmt_metric(value, fallback=np.nan) -> str:
    try:
        if value is None:
            value = fallback
        return f"{float(value):.4f}"
    except Exception:
        return f"{float(fallback):.4f}"


def render_sidebar() -> tuple[str, str, object | None]:
    """Render dataset controls and load the selected CSV into session state."""
    with st.sidebar:
        st.header("Nguon du lieu")

        active_path = st.session_state.get("data_path")
        active_df = st.session_state.get("df")
        if active_df is not None:
            name = Path(active_path).name if active_path else "uploaded file"
            st.success(f"Dataset dang dung: `{name}`\n\n{active_df.shape[0]:,} rows - {active_df.shape[1]} cols")
        else:
            st.warning("Chua load dataset.")

        st.divider()

        source = st.radio(
            "Nguon dataset",
            ["Duong dan trong workspace", "Upload file CSV"],
            index=0,
            help="CSV can co location_key va Time/ts_utc/timestamp.",
        )
        use_upload = source.startswith("Upload")

        data_path = ""
        uploaded = None
        if use_upload:
            uploaded = st.file_uploader("Chon file CSV", type=["csv"])
        else:
            data_path = st.text_input(
                "Path CSV",
                value=st.session_state.get("_sidebar_path_input", str(PROJECT_ROOT / "dataset" / "air_quality.csv")),
                key="_sidebar_path_input",
            )
            st.caption(f"Project root: `{PROJECT_ROOT}`")

        col1, col2 = st.columns(2)
        with col1:
            load_clicked = st.button("Load", use_container_width=True, type="primary")
        with col2:
            reset_clicked = st.button("Reset", use_container_width=True)

        if reset_clicked:
            for key in ["df", "data_path", "_upload_saved_path"]:
                st.session_state.pop(key, None)
            st.rerun()

        if load_clicked:
            _load_dataset(use_upload, data_path, uploaded)

    return source, data_path, uploaded


def _load_dataset(use_upload: bool, data_path: str, uploaded) -> None:
    try:
        if use_upload:
            if uploaded is None:
                st.sidebar.error("Ban chua chon file CSV.")
                return
            UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            saved_path = UPLOAD_TEMP_DIR / uploaded.name
            with open(saved_path, "wb") as f:
                f.write(uploaded.getbuffer())
            abs_path = saved_path.resolve()
        else:
            if not data_path or not data_path.strip():
                st.sidebar.error("Path CSV khong duoc de trong.")
                return
            raw_path = Path(data_path.strip())
            abs_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
            abs_path = abs_path.resolve()
            if not abs_path.exists():
                st.sidebar.error(f"Khong tim thay file: `{abs_path}`")
                return

        df = pd.read_csv(abs_path)
        st.session_state["df"] = df
        st.session_state["data_path"] = str(abs_path)
        st.sidebar.success(f"Load thanh cong: `{abs_path.name}`\n\n{df.shape[0]:,} rows - {df.shape[1]} cols")

    except Exception as exc:
        st.sidebar.error(f"Load dataset loi: {exc}")


def render_data_preview(df: pd.DataFrame) -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Preview du lieu")
        st.dataframe(df.head(20), use_container_width=True)
    with col2:
        st.subheader("Thong tin")
        st.write(f"Rows: {len(df):,}")
        st.write(f"Columns: {df.shape[1]}")


def render_location_selector(df: pd.DataFrame) -> list[str]:
    locations = sorted(df["location_key"].dropna().astype(str).unique().tolist()) if "location_key" in df.columns else []
    st.subheader("Chon dia diem")
    selected_locations = st.multiselect(
        "Chon dia diem de du doan hoac train lai",
        options=locations,
        default=locations[: min(3, len(locations))],
        help="Khi predict bang checkpoint, hay chon dung location da dung luc train. iTransformer hien chay 1 location moi lan.",
    )

    if selected_locations:
        _render_sample_count_preview(
            df,
            selected_locations,
            int(st.session_state.get("train_window_size", 72)),
            int(st.session_state.get("train_horizon", 12)),
            int(st.session_state.get("train_sample_stride", 4)),
        )

    return selected_locations


def _render_sample_count_preview(
    df: pd.DataFrame,
    selected_locations: list[str],
    window: int,
    horizon: int,
    sample_stride: int,
) -> None:
    try:
        df_sel = df.loc[df["location_key"].astype(str).isin([str(x) for x in selected_locations])].copy()
        default_target = "aqi" if "aqi" in df_sel.columns else None
        if default_target is None:
            numeric_cols = [c for c in df_sel.select_dtypes(include=["number"]).columns if c != "_loc_id"]
            default_target = numeric_cols[0] if numeric_cols else None
        if default_target is None:
            st.warning("Khong tim thay cot numeric de tinh preview sample.")
            return

        mod = load_train_module()
        if mod is None or not hasattr(mod, "build_time_series_samples"):
            st.warning("Khong load duoc helper build_time_series_samples.")
            return

        col_map = {c.lower(): c for c in df_sel.columns}
        ts_col = col_map.get("ts_utc") or col_map.get("time") or col_map.get("timestamp")
        exclude_cols = [c for c in [ts_col, "location_key"] if c]
        feature_cols = [c for c in df_sel.columns if c not in exclude_cols]
        if default_target not in feature_cols:
            feature_cols.append(default_target)

        x_seq, loc_ids, y, y_ts, _, _ = mod.build_time_series_samples(
            df_sel,
            default_target,
            window,
            horizon,
            sample_stride=sample_stride,
            feature_cols=feature_cols,
            include_target_history=True,
        )
        if hasattr(mod, "split_data_by_timeline"):
            train, val, test = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)
        else:
            train, val, test = split_data_by_timeline(x_seq, loc_ids, y, y_ts)

        st.markdown(
            f"**Preview samples**: total={len(y):,} | train={len(train.y):,} | "
            f"val={len(val.y):,} | test={len(test.y):,} | stride={sample_stride}"
        )
    except Exception as exc:
        st.warning(f"Khong the tinh preview samples: {exc}")


def render_train_config() -> dict:
    """Render model/training config. These values are used for both predict and retrain."""
    df = st.session_state.get("df", pd.DataFrame())
    all_cols = df.columns.tolist()
    blocked_lower = {"ts_utc", "time", "timestamp", "location_key"}
    feature_options = [
        c
        for c in all_cols
        if c.lower() not in blocked_lower
        and not c.lower().startswith("unnamed:")
        and c not in {"y_true", "y_pred", "abs_error"}
    ]

    st.subheader("Cau hinh model")
    conf1, conf2, conf3 = st.columns(3)

    with conf1:
        target_col = st.selectbox(
            "Target column",
            options=feature_options,
            index=feature_options.index("aqi") if "aqi" in feature_options else 0,
        )
        feature_cols = st.multiselect("Input feature columns", options=feature_options, default=list(feature_options))
        loss_name = st.selectbox("Loss", options=["huber", "mse"], index=0)

    with conf2:
        window_size = st.number_input("Window size (timesteps)", min_value=1, max_value=168, value=72, step=1)
        horizon = st.number_input("Horizon (timesteps)", min_value=1, max_value=168, value=12, step=1)
        sample_stride = st.number_input("Sliding step", min_value=1, max_value=168, value=4, step=1)
        st.session_state["train_window_size"] = int(window_size)
        st.session_state["train_horizon"] = int(horizon)
        st.session_state["train_sample_stride"] = int(sample_stride)
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=50, step=1)
        early_stop_patience = st.number_input("Early stop patience", min_value=0, max_value=50, value=5, step=1)
        batch_size = st.number_input("Batch size", min_value=8, max_value=8192, value=128, step=8)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=3e-4, format="%.6f")
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-4, format="%.6f")

    with conf3:
        d_model = st.number_input("d_model", min_value=16, max_value=512, value=64, step=16)
        n_layers = st.number_input("n_layers", min_value=1, max_value=8, value=2, step=1)
        grad_accum_steps = st.number_input("Gradient accumulation", min_value=1, max_value=64, value=1, step=1)
        max_grad_norm = st.number_input("Max grad norm", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        use_gpu = st.checkbox("Dung GPU neu co", value=True)

    import torch

    if use_gpu and not torch.cuda.is_available():
        st.warning("PyTorch hien khong nhan CUDA. Predict/train se chay bang CPU.")

    return {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "loss_name": loss_name,
        "window_size": int(window_size),
        "horizon": int(horizon),
        "sample_stride": int(sample_stride),
        "epochs": int(epochs),
        "early_stop_patience": int(early_stop_patience),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "grad_accum_steps": int(grad_accum_steps),
        "max_grad_norm": float(max_grad_norm),
        "seed": int(seed),
        "use_gpu": bool(use_gpu),
    }


def render_model_results(
    summary: dict,
    hist_df: pd.DataFrame,
) -> None:
    loaded_checkpoint = str(summary.get("mode", "")).startswith("loaded")
    if loaded_checkpoint:
        st.success("Da chay inference moi bang checkpoint tren Streamlit")
    else:
        st.success("Train/Test hoan tat")

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Test MAE", _fmt_metric(summary.get("test_mae")))
    met2.metric("Test RMSE", _fmt_metric(summary.get("test_rmse")))
    met3.metric("Test R2", _fmt_metric(summary.get("test_r2")))
    met4.metric("Locations done", f"{int(summary['future_locations']):,}")

    st.write("### Thong ke")
    stats = {
        "mode": summary.get("mode", "train"),
        "n_rows_used": summary["n_rows_used"],
        "split_train": summary["split_train"],
        "split_val": summary["split_val"],
        "split_test": summary["split_test"],
        "future_rows": summary["future_rows"],
        "future_locations": summary["future_locations"],
        "val_mae": summary.get("val_mae"),
        "val_rmse": summary.get("val_rmse"),
        "val_r2": summary.get("val_r2"),
        "test_mae": summary.get("test_mae"),
        "test_rmse": summary.get("test_rmse"),
        "test_r2": summary.get("test_r2"),
        "val_mae_norm": summary.get("val_mae_norm"),
        "val_rmse_norm": summary.get("val_rmse_norm"),
        "train_only_sec": round(float(summary.get("train_only_sec", 0.0)), 2),
        "eval_sec": round(float(summary.get("eval_sec", 0.0)), 2),
        "forecast_sec": round(float(summary.get("forecast_sec", 0.0)), 2),
        "run_sec": round(float(summary.get("run_sec", 0.0)), 2),
    }
    st.write(stats)

    if loaded_checkpoint:
        st.write("### Lich su train cua checkpoint")
        if hist_df is not None and not hist_df.empty:
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("Khong tim thay metrics_history.csv canh checkpoint.")
    else:
        st.write("### Lich su train")
        st.dataframe(hist_df, use_container_width=True)


def render_forecast_download(future_df: pd.DataFrame, summary: dict, key_prefix: str = "forecast") -> None:
    st.write("### Ket qua du doan tiep theo")
    st.dataframe(future_df.head(300), use_container_width=True)

    download_name = os.path.basename(summary.get("future_pred_path", "future_predictions.csv"))
    st.download_button(
        "Download file du doan",
        data=future_df.to_csv(index=False).encode("utf-8"),
        file_name=download_name,
        mime="text/csv",
        key=f"{key_prefix}_download_predictions",
    )
    st.info("File nay duoc tao tu lan predict/train vua chay tren Streamlit, khong phai doc lai future_*.csv co san.")
    st.code(f"run_dir: {os.path.dirname(summary['future_pred_path'])}\nFiles: {download_name}")
