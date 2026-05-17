"""
ui_components.py
----------------
Các component UI tái sử dụng: sidebar, metrics, bảng so sánh, download, v.v.
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
import streamlit as st

from Utils import load_train_module, split_data_by_timeline

# Thư mục lưu tạm file upload (nằm cạnh app/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UPLOAD_TEMP_DIR = _PROJECT_ROOT / "runs" / "uploaded"


def _fmt_metric(value, fallback=np.nan) -> str:
    try:
        if value is None:
            value = fallback
        return f"{float(value):.4f}"
    except Exception:
        return f"{float(fallback):.4f}"


# ---------------------------------------------------------------------------
# Sidebar & dataset loading
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str, object | None]:
    """Render sidebar nguồn dữ liệu, trả về (source, data_path, uploaded_file)."""
    with st.sidebar:
        st.header("📂 Nguồn dữ liệu")

        # ── Trạng thái dataset hiện tại ─────────────────────────────────────
        _active_path = st.session_state.get("data_path", None)
        _active_df   = st.session_state.get("df", None)
        if _active_df is not None and _active_path:
            st.success(
                f"✅ Dataset đang dùng:\n"
                f"`{Path(_active_path).name}`\n"
                f"{_active_df.shape[0]:,} rows · {_active_df.shape[1]} cols"
            )
        elif _active_df is not None:
            st.info(
                f"📄 Dataset đang dùng: *(file upload tạm)*\n"
                f"{_active_df.shape[0]:,} rows · {_active_df.shape[1]} cols"
            )
        else:
            st.warning("⚠️ Chưa load dataset. Hãy chọn nguồn bên dưới và nhấn **Load**.")

        st.divider()

        # ── Chọn nguồn ──────────────────────────────────────────────────────
        source = st.radio(
            "Nguồn dataset",
            ["📁 Đường dẫn trong workspace", "⬆️ Upload file CSV"],
            index=0,
            help="Chọn cách cung cấp file CSV đầu vào cho Mamba.",
        )
        use_upload = source.startswith("⬆️")

        data_path = ""
        uploaded  = None

        if not use_upload:
            data_path = st.text_input(
                "Path CSV (tương đối hoặc tuyệt đối)",
                value=st.session_state.get("_sidebar_path_input", str(_PROJECT_ROOT / "dataset" / "air_quality.csv")),
                help="Ví dụ: dataset/2025.csv  hoặc  D:/data/myfile.csv",
                key="_sidebar_path_input",
            )
            st.caption(f"📌 Thư mục gốc: `{_PROJECT_ROOT}`")
        else:
            uploaded = st.file_uploader(
                "Chọn file CSV để upload",
                type=["csv"],
                help="File sẽ được lưu tạm vào thư mục `runs/uploaded/`.",
            )
            if uploaded is not None:
                st.caption(f"📎 File đã chọn: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

        # ── Nút Load / Reset ─────────────────────────────────────────────────
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            load_clicked = st.button("⬇️ Load", use_container_width=True, type="primary")
        with btn_col2:
            reset_clicked = st.button("🗑️ Reset", use_container_width=True)

        if reset_clicked:
            for key in ["df", "data_path", "_upload_saved_path"]:
                st.session_state.pop(key, None)
            st.rerun()

        if load_clicked:
            _load_dataset(use_upload, data_path, uploaded)

    return source, data_path, uploaded


def _load_dataset(use_upload: bool, data_path: str, uploaded) -> None:
    """Load dataset vào session_state['df'] và lưu path tuyệt đối vào session_state['data_path']."""
    try:
        if use_upload:
            # ── Trường hợp upload file ──────────────────────────────────────
            if uploaded is None:
                st.sidebar.error("❌ Bạn chưa chọn file CSV để upload.")
                return

            # Lưu file tạm ra disk để TFT subprocess đọc được đường dẫn thực
            _UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            saved_path = _UPLOAD_TEMP_DIR / uploaded.name
            with open(saved_path, "wb") as f:
                f.write(uploaded.getbuffer())

            df = pd.read_csv(saved_path)
            st.session_state["df"]              = df
            st.session_state["data_path"]       = str(saved_path.resolve())
            st.session_state["_upload_saved_path"] = str(saved_path.resolve())

            st.sidebar.success(
                f"✅ Upload thành công: **{uploaded.name}**\n"
                f"{df.shape[0]:,} rows · {df.shape[1]} cols"
            )

        else:
            # ── Trường hợp nhập đường dẫn ───────────────────────────────────
            if not data_path or not data_path.strip():
                st.sidebar.error("❌ Đường dẫn CSV không được để trống.")
                return

            raw_path = Path(data_path.strip())
            abs_path = raw_path if raw_path.is_absolute() else (_PROJECT_ROOT / raw_path)
            abs_path = abs_path.resolve()
            if not abs_path.exists():
                st.sidebar.error(
                    f"❌ Không tìm thấy file:\n`{abs_path}`\n\n"
                    "Hãy kiểm tra lại đường dẫn hoặc dùng đường dẫn tuyệt đối."
                )
                return

            df = pd.read_csv(abs_path)
            st.session_state["df"]        = df
            st.session_state["data_path"] = str(abs_path)

            st.sidebar.success(
                f"✅ Load thành công: **{Path(abs_path).name}**\n"
                f"{df.shape[0]:,} rows · {df.shape[1]} cols"
            )

    except Exception as e:
        st.sidebar.error(f"❌ Load dataset lỗi: {e}")


# ---------------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------------

def render_data_preview(df: pd.DataFrame) -> None:
    """Hiển thị preview và thông tin cơ bản của dataset."""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Preview dữ liệu")
        st.dataframe(df.head(20), use_container_width=True)
    with col2:
        st.subheader("Thông tin")
        st.write(f"Rows: {len(df):,}")
        st.write(f"Columns: {df.shape[1]}")


# ---------------------------------------------------------------------------
# Location selector + sample preview
# ---------------------------------------------------------------------------

def render_location_selector(df: pd.DataFrame) -> list[str]:
    """Render location multiselect + sample count preview. Trả về selected_locations."""
    locations = sorted(df["location_key"].dropna().astype(str).unique().tolist()) if "location_key" in df.columns else []
    st.subheader("Chọn địa điểm để train + forecast")
    selected_locations = st.multiselect(
        "Chọn địa điểm để train + forecast",
        options=locations,
        default=locations[: min(3, len(locations))],
        help="Có thể chọn 1 hoặc nhiều địa điểm. Mô hình sẽ train chung theo nhiều tỉnh.",
    )

    preview_window = int(st.session_state.get("train_window_size", 96))
    preview_horizon = int(st.session_state.get("train_horizon", 24))
    preview_stride = int(st.session_state.get("train_sample_stride", 1))

    if selected_locations:
        _render_sample_count_preview(
            df, selected_locations, int(preview_window), int(preview_horizon), int(preview_stride)
        )

    return selected_locations


def _render_sample_count_preview(
    df: pd.DataFrame, selected_locations: list[str], window: int, horizon: int, sample_stride: int
) -> None:
    """Hiển thị số lượng sample train/val/test theo preview window/horizon."""
    try:
        df_sel = df.loc[
            df["location_key"].astype(str).isin([str(x) for x in selected_locations])
        ].copy()
        default_target = (
            "aqi"
            if "aqi" in df_sel.columns
            else next(
                (c for c in df_sel.select_dtypes(include=["number"]).columns if c != "_loc_id"),
                None,
            )
        )
        if default_target is None:
            st.warning("Không tìm thấy cột số để preview sample counts.")
            return

        mod = load_train_module()
        if mod is None or not hasattr(mod, "build_time_series_samples"):
            st.warning("Không thể load helper 'build_time_series_samples' để preview samples.")
            return

        col_map = {c.lower(): c for c in df_sel.columns}
        ts_col = col_map.get("ts_utc") or col_map.get("time") or col_map.get("timestamp")
        exclude_cols = [c for c in [ts_col, col_map.get("location_key") or "location_key"] if c]
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
            f"**Preview ({len(selected_locations)} locations)**: "
            f"total samples={len(y):,} | sliding step={sample_stride}"
        )
        st.write(f"Train: {len(train.y):,}  |  Val: {len(val.y):,}  |  Test: {len(test.y):,}")
    except Exception as e:
        st.warning(f"Không thể tính preview samples: {e}")


# ---------------------------------------------------------------------------
# Train config form
# ---------------------------------------------------------------------------

def render_train_config() -> dict:
    """Render toàn bộ form cấu hình train. Trả về dict config."""
    df = st.session_state.get("df", pd.DataFrame())
    all_cols = df.columns.tolist()
    reserved_cols = {"y_true", "y_pred", "abs_error"}
    blocked_lower = {"ts_utc", "time", "timestamp", "location_key"}
    feature_options = [
        c for c in all_cols
        if c not in reserved_cols
        and c.lower() not in blocked_lower
        and not c.lower().startswith("unnamed:")
    ]

    st.subheader("Cấu hình train")
    conf1, conf2, conf3 = st.columns(3)

    with conf1:
        target_col = st.selectbox(
            "Target column (biến cần dự đoán)",
            options=feature_options,
            index=feature_options.index("aqi") if "aqi" in feature_options else 0,
        )
        default_features = list(feature_options)
        feature_cols = st.multiselect(
            "Input feature columns",
            options=feature_options,
            default=default_features,
        )
        loss_name = st.selectbox("Loss", options=["huber", "mse"], index=0)

    with conf2:
        window_size = st.number_input("Window size (timesteps)", min_value=1, max_value=168, value=96, step=1)
        horizon = st.number_input("Horizon (timesteps)", min_value=1, max_value=168, value=24, step=1)
        sample_stride = st.number_input("Sliding step", min_value=1, max_value=168, value=1, step=1)
        st.session_state["train_window_size"] = int(window_size)
        st.session_state["train_horizon"] = int(horizon)
        st.session_state["train_sample_stride"] = int(sample_stride)
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=50, step=1)
        early_stop_patience = st.number_input(
            "Early stop patience", min_value=0, max_value=50, value=5, step=1
        )
        batch_size = st.number_input("Batch size", min_value=8, max_value=8192, value=128, step=8)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=3e-4, format="%.6f")
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-4, format="%.6f")

    with conf3:
        d_model = st.number_input("d_model", min_value=16, max_value=512, value=64, step=16)
        n_layers = st.number_input("n_layers", min_value=1, max_value=8, value=2, step=1)
        grad_accum_steps = st.number_input("Gradient accumulation", min_value=1, max_value=64, value=1, step=1)
        max_grad_norm = st.number_input("Max grad norm", min_value=0.0, max_value=100.0, value=1.0, step=0.5)

    run1, run2, run3 = st.columns(3)
    with run1:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
    with run2:
        st.markdown(" ")
    with run3:
        use_gpu = st.checkbox("Dùng GPU (nếu có)", value=True)

    import torch
    if use_gpu and not torch.cuda.is_available():
        st.warning(
            "Bạn đang bật GPU nhưng PyTorch hiện không nhận CUDA (torch+cpu). "
            "Train sẽ chạy bằng CPU nên thời gian mỗi epoch sẽ cao."
        )

    return dict(
        target_col=target_col,
        feature_cols=feature_cols,
        loss_name=loss_name,
        window_size=int(window_size),
        horizon=int(horizon),
        sample_stride=int(sample_stride),
        epochs=int(epochs),
        early_stop_patience=int(early_stop_patience),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        d_model=int(d_model),
        n_layers=int(n_layers),
        grad_accum_steps=int(grad_accum_steps),
        max_grad_norm=float(max_grad_norm),
        seed=int(seed),
        use_gpu=bool(use_gpu),
    )


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def render_mamba_results(summary: dict, hist_df: pd.DataFrame, future_df: pd.DataFrame, df: pd.DataFrame, selected_locations: list[str]) -> None:
    """Hiển thị kết quả sau khi train Mamba."""
    st.success("Train/Test hoàn tất")

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Val MAE (norm)", _fmt_metric(summary.get("val_mae_norm")))
    met2.metric("Val RMSE (norm)", _fmt_metric(summary.get("val_rmse_norm")))
    met3.metric("Val R2", f"{summary['val_r2']:.4f}")
    met4.metric("Locations done", f"{int(summary['future_locations']):,}")

    st.write("### Số dòng sau khi lọc theo địa điểm")
    train_counts = (
        df.loc[df["location_key"].astype(str).isin([str(x) for x in selected_locations]), "location_key"]
        .astype(str)
        .value_counts()
        .rename_axis("location_key")
        .reset_index(name="train_source_rows")
    )
    test_counts = pd.DataFrame(
        {
            "location_key": selected_locations,
            "test_source_rows": [int(summary["split_test"])] * len(selected_locations),
        }
    )
    used_counts = (
        future_df["location"].astype(str)
        .value_counts()
        .rename_axis("location_key")
        .reset_index(name="future_rows")
    )
    stats_df = (
        train_counts
        .merge(test_counts, on="location_key", how="outer")
        .merge(used_counts, on="location_key", how="outer")
        .fillna(0)
    )

    merged_stats = stats_df.merge(
        pd.DataFrame(
            [
                {
                    "n_rows_used": summary["n_rows_used"],
                    "split_train": summary["split_train"],
                    "split_val": summary["split_val"],
                    "split_test": summary["split_test"],
                }
            ]
        ),
        how="cross",
    )
    st.dataframe(merged_stats, use_container_width=True)

    st.write("### Thống kê split")
    st.write(
        {k: round(float(v), 2) if isinstance(v, float) else v
         for k, v in {
             "split_train": summary["split_train"],
             "split_val": summary["split_val"],
             "split_test": summary["split_test"],
             "n_rows_used": summary["n_rows_used"],
             "future_rows": summary["future_rows"],
             "future_locations": summary["future_locations"],
             "train_only_sec": summary.get("train_only_sec", np.nan),
             "eval_sec": summary.get("eval_sec", np.nan),
             "forecast_sec": summary.get("forecast_sec", np.nan),
             "io_sec": summary.get("io_sec", np.nan),
             "run_sec": summary["run_sec"],
         }.items()}
    )

    st.write("### Lịch sử train Mamba")
    st.dataframe(hist_df, use_container_width=True)


def render_forecast_download(future_df: pd.DataFrame, summary: dict) -> None:
    """Hiển thị bảng dự báo 24h và nút download."""
    st.write("### Dự báo 24 giờ tiếp theo (từng địa điểm)")
    st.dataframe(future_df.head(300), use_container_width=True)
    st.download_button(
        "Download file tổng (mọi location)",
        data=future_df.to_csv(index=False).encode("utf-8"),
        file_name="future_24h_predictions.csv",
        mime="text/csv",
    )
    st.info("Mamba xuất 1 file tổng trong run_dir: future_24h_predictions.csv (gồm time, location, predicted).")
    st.code(
        f"run_dir: {os.path.dirname(summary['future_pred_path'])}\n"
        "Files: future_24h_predictions.csv"
    )
