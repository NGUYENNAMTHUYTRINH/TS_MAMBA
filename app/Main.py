"""
main.py
-------
Entry point cua Streamlit app.
Chay bang: streamlit run app/Main.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for path in [str(APP_DIR), str(PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from Pipeline_mamba import predict_with_saved_model, train_pipeline
from Pipeline_lstm import predict_lstm_with_saved_model, train_lstm_pipeline
from Ui_components import (
    render_data_preview,
    render_forecast_download,
    render_location_selector,
    render_mamba_results,
    render_sidebar,
    render_train_config,
)


def _selected_feature_cols(train_cfg: dict) -> list[str]:
    feature_cols = list(train_cfg["feature_cols"])
    if train_cfg["target_col"] not in feature_cols:
        feature_cols.append(train_cfg["target_col"])
    return feature_cols


def _default_checkpoint_path(project_root: Path, model_type: str) -> str:
    """Return the newest trained checkpoint path for the Streamlit input."""
    if model_type == "LSTM":
        patterns = [
            "runs/lstm/*/best_lstm.pt",
            "runs/*/best_lstm.pt",
            "runs/*/lstm/best_lstm.pt",
            "LSTM-Time-Series-Forecasting/outputs/*/best_lstm.pt",
            "LSTM-Time-Series-Forecasting/outputs/best_lstm.pt",
        ]
        fallback = project_root / "LSTM-Time-Series-Forecasting" / "outputs" / "best_lstm.pt"
    else:
        patterns = [
            "runs/mamba/*/best_mamba_aqi.pt",
            "runs/mamba/*/best_mamba.pt",
            "runs/*/best_mamba_aqi.pt",
            "runs/*/mamba/best_mamba.pt",
            "outputs/best_mamba_aqi.pt",
        ]
        fallback = project_root / "outputs" / "best_mamba_aqi.pt"

    candidates = []
    for pattern in patterns:
        candidates.extend(project_root.glob(pattern))

    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return str(fallback)

    latest = max(existing, key=lambda path: path.stat().st_mtime)
    return str(latest)


def _model_run_dir(project_root: Path, model_name: str, timestamp: str) -> str:
    return str(project_root / "runs" / model_name.lower() / timestamp)


def _render_comparison(results: list[tuple[str, dict, object, object]]) -> None:
    rows = []
    for name, summary, _, _ in results:
        rows.append({
            "model": name,
            "test_mae": summary.get("test_mae"),
            "test_rmse": summary.get("test_rmse"),
            "test_r2": summary.get("test_r2"),
            "val_mae_norm": summary.get("val_mae_norm"),
            "val_rmse_norm": summary.get("val_rmse_norm"),
            "val_r2": summary.get("val_r2"),
            "future_rows": summary.get("future_rows"),
        })
    st.write("### So sanh model")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _metric_value(summary: dict, key: str, default: float) -> float:
    try:
        value = summary.get(key)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _render_best_model(results: list[tuple[str, dict, object, object]]) -> None:
    if not results:
        return

    best = min(
        results,
        key=lambda item: (
            _metric_value(item[1], "test_rmse", float("inf")),
            -_metric_value(item[1], "test_r2", float("-inf")),
            _metric_value(item[1], "test_mae", float("inf")),
        ),
    )
    name, summary, _, _ = best

    st.write("### Model tot nhat")
    st.success(f"{name} dang tot nhat theo Test RMSE thap nhat.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", name)
    c2.metric("Test RMSE", f"{_metric_value(summary, 'test_rmse', float('nan')):.4f}")
    c3.metric("Test MAE", f"{_metric_value(summary, 'test_mae', float('nan')):.4f}")
    c4.metric("Test R2", f"{_metric_value(summary, 'test_r2', float('nan')):.4f}")


def main() -> None:
    st.set_page_config(page_title="AQI Mamba", layout="wide")
    st.title("AQI Forecasting: Mamba")
    st.caption("Du doan tren Streamlit bang checkpoint da train, hoac train lai model khi can.")

    render_sidebar()

    if "df" not in st.session_state:
        st.info("Vui long load dataset tu sidebar de bat dau.")
        return

    df = st.session_state["df"]
    selected_locations = render_location_selector(df)
    train_cfg = render_train_config()
    render_data_preview(df)

    results: list[tuple[str, dict, object, object]] = []

    project_root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.subheader("Model")
    model_type = st.selectbox("Loai model", ["Mamba", "LSTM", "Mamba + LSTM"], index=0)
    checkpoint_paths: dict[str, str] = {}
    if model_type == "Mamba + LSTM":
        cp1, cp2 = st.columns(2)
        with cp1:
            checkpoint_paths["Mamba"] = st.text_input(
                "Checkpoint Mamba da train",
                value=_default_checkpoint_path(project_root, "Mamba"),
                key="checkpoint_path_mamba_compare",
                help="Streamlit se load checkpoint va chay inference moi.",
            )
        with cp2:
            checkpoint_paths["LSTM"] = st.text_input(
                "Checkpoint LSTM da train",
                value=_default_checkpoint_path(project_root, "LSTM"),
                key="checkpoint_path_lstm_compare",
                help="Streamlit se load checkpoint va chay inference moi.",
            )
    else:
        checkpoint_paths[model_type] = st.text_input(
            "Checkpoint da train",
            value=_default_checkpoint_path(project_root, model_type),
            key=f"checkpoint_path_{model_type.lower()}",
            help="Streamlit se load checkpoint va chay inference moi, khong doc file future_*.csv co san.",
        )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        predict_clicked = st.button("Du doan bang checkpoint", use_container_width=True, type="primary")
    with action_col2:
        train_clicked = st.button("Train lai model", use_container_width=True)

    if predict_clicked or train_clicked:
        if not selected_locations:
            st.warning("Vui long chon it nhat mot dia diem.")
            st.stop()

        feature_cols = _selected_feature_cols(train_cfg)
        if not feature_cols:
            st.warning("Vui long chon it nhat mot cot feature.")
            st.stop()

        status_placeholder = st.empty()

        try:
            if predict_clicked:
                status_placeholder.info("Dang load checkpoint va chay inference moi tren Streamlit...")
                models_to_run = ["Mamba", "LSTM"] if model_type == "Mamba + LSTM" else [model_type]
                for name in models_to_run:
                    if name == "LSTM":
                        summary, hist_df, future_df = predict_lstm_with_saved_model(
                            df=df,
                            selected_locations=selected_locations,
                            target_col=train_cfg["target_col"],
                            feature_cols=feature_cols,
                            checkpoint_path=checkpoint_paths["LSTM"],
                            batch_size=train_cfg["batch_size"],
                            use_gpu=train_cfg["use_gpu"],
                            run_dir=_model_run_dir(project_root, "lstm", timestamp),
                        )
                    else:
                        summary, hist_df, future_df = predict_with_saved_model(
                            df=df,
                            forecast_base_df=None,
                            selected_locations=selected_locations,
                            target_col=train_cfg["target_col"],
                            feature_cols=feature_cols,
                            window_size=train_cfg["window_size"],
                            horizon=train_cfg["horizon"],
                            sample_stride=train_cfg["sample_stride"],
                            loss_name=train_cfg["loss_name"],
                            batch_size=train_cfg["batch_size"],
                            use_gpu=train_cfg["use_gpu"],
                            checkpoint_path=checkpoint_paths["Mamba"],
                            run_dir=_model_run_dir(project_root, "mamba", timestamp),
                        )
                    results.append((name, summary, hist_df, future_df))
                status_placeholder.success("Da du doan xong bang checkpoint.")

            else:
                status_placeholder.info(f"Dang huan luyen {model_type}...")
                models_to_run = ["Mamba", "LSTM"] if model_type == "Mamba + LSTM" else [model_type]
                for name in models_to_run:
                    if name == "LSTM":
                        summary, hist_df, future_df = train_lstm_pipeline(
                            df=df,
                            selected_locations=selected_locations,
                            target_col=train_cfg["target_col"],
                            feature_cols=feature_cols,
                            window_size=train_cfg["window_size"],
                            horizon=train_cfg["horizon"],
                            epochs=train_cfg["epochs"],
                            batch_size=train_cfg["batch_size"],
                            lr=train_cfg["lr"],
                            hidden_size=train_cfg["d_model"],
                            num_layers=train_cfg["n_layers"],
                            loss_name=train_cfg["loss_name"],
                            seed=train_cfg["seed"],
                            use_gpu=train_cfg["use_gpu"],
                            early_stop_patience=train_cfg["early_stop_patience"],
                            run_dir=_model_run_dir(project_root, "lstm", timestamp),
                        )
                    else:
                        summary, hist_df, future_df = train_pipeline(
                            df=df,
                            forecast_base_df=None,
                            selected_locations=selected_locations,
                            target_col=train_cfg["target_col"],
                            feature_cols=feature_cols,
                            window_size=train_cfg["window_size"],
                            horizon=train_cfg["horizon"],
                            sample_stride=train_cfg["sample_stride"],
                            epochs=train_cfg["epochs"],
                            batch_size=train_cfg["batch_size"],
                            lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"],
                            d_model=train_cfg["d_model"],
                            n_layers=train_cfg["n_layers"],
                            loss_name=train_cfg["loss_name"],
                            seed=train_cfg["seed"],
                            num_workers=4,
                            use_gpu=train_cfg["use_gpu"],
                            log_interval=50,
                            grad_accum_steps=train_cfg["grad_accum_steps"],
                            max_grad_norm=train_cfg["max_grad_norm"],
                            early_stop_patience=train_cfg["early_stop_patience"],
                            early_stop_min_delta=0.0,
                            run_dir=_model_run_dir(project_root, "mamba", timestamp),
                        )
                    results.append((name, summary, hist_df, future_df))
                status_placeholder.success(f"Hoan thanh huan luyen {model_type}.")

        except Exception as e:
            st.error(f"Loi pipeline: {e}")
            import traceback

            traceback.print_exc()
            return

    if results:
        if len(results) > 1:
            _render_best_model(results)
            _render_comparison(results)
        for name, summary, hist_df, future_df in results:
            st.divider()
            st.write(f"### {name} result")
            render_mamba_results(summary, hist_df)
            if future_df is not None:
                render_forecast_download(future_df, summary, key_prefix=name.lower())


if __name__ == "__main__":
    main()
