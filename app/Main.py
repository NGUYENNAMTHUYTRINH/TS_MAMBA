"""
main.py
-------
Entry point của Streamlit app.
Chạy bằng:  streamlit run app/Main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Đảm bảo nhận diện đúng các module trong thư mục app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import streamlit as st

from Pipeline_mamba import train_pipeline

from Ui_components import (
    render_data_preview,
    render_forecast_download,
    render_location_selector,
    render_mamba_results,
    render_sidebar,
    render_train_config,
)


def main() -> None:
    st.set_page_config(page_title="AQI Mamba", layout="wide")
    st.title("🚀 AQI Forecasting: Mamba")
    st.caption("Huấn luyện và dự báo chất lượng không khí với Mamba.")

    # --- 1. Sidebar & Load Dataset ---
    render_sidebar()

    if "df" not in st.session_state:
        st.info("👈 Vui lòng load dataset từ sidebar để bắt đầu.")
        return

    df = st.session_state["df"]

    # --- 2. Cấu hình chung và chọn địa điểm ---
    selected_locations = render_location_selector(df)
    
    train_cfg = render_train_config()

    # Preview dữ liệu (đã bọc try-except bên trong component)
    render_data_preview(df)

    # --- 3. Khởi tạo các biến chứa kết quả (Tránh lỗi NameError) ---
    summary, hist_df, future_df = None, None, None

    # Tạo đường dẫn lưu kết quả dựa trên thời gian
    project_root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir = str(project_root / "runs" / timestamp)

    # --- 4. Vòng lặp huấn luyện khi nhấn nút ---
    if st.button("🔥 Run Training", use_container_width=True):
        if not selected_locations:
            st.warning("⚠️ Vui lòng chọn ít nhất một địa điểm.")
            st.stop()

        feature_cols = list(train_cfg["feature_cols"])
        if train_cfg["target_col"] not in feature_cols:
            feature_cols.append(train_cfg["target_col"])

        if not feature_cols:
            st.warning("⚠️ Vui lòng chọn ít nhất một cột feature.")
            st.stop()

        status_placeholder = st.empty()

        try:
            status_placeholder.info("⏳ Đang huấn luyện Mamba...")
            mamba_run_dir = os.path.join(base_run_dir, "mamba")
            summary, hist_df, future_df = train_pipeline(
                df=df,
                forecast_base_df=None,
                selected_locations=selected_locations,
                target_col=train_cfg["target_col"],
                feature_cols=feature_cols,
                window_size=train_cfg["window_size"],
                horizon=train_cfg["horizon"],
                epochs=train_cfg["epochs"],
                batch_size=train_cfg["batch_size"],
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
                d_model=train_cfg["d_model"],
                n_layers=train_cfg["n_layers"],
                loss_name=train_cfg["loss_name"],
                seed=train_cfg["seed"],
                num_workers=train_cfg["num_workers"],
                use_gpu=train_cfg["use_gpu"],
                log_interval=50,
                grad_accum_steps=train_cfg["grad_accum_steps"],
                max_grad_norm=train_cfg["max_grad_norm"],
                early_stop_patience=train_cfg["early_stop_patience"],
                early_stop_min_delta=train_cfg["early_stop_min_delta"],
                run_dir=mamba_run_dir,
            )

            status_placeholder.success("✅ Hoàn thành huấn luyện Mamba!")

        except Exception as e:
            st.error(f"❌ Quá trình huấn luyện gặp lỗi: {e}")
            import traceback
            traceback.print_exc()
            return

    # --- 5. Hiển thị kết quả (Render Results) ---
    # Chỉ hiển thị nếu biến kết quả không phải None (nghĩa là đã được chạy thành công)
    
    # Kết quả Mamba
    if summary is not None:
        render_mamba_results(summary, hist_df, future_df, df, selected_locations)
        # Nút tải dự báo cho Mamba (nếu có)
        if future_df is not None:
            render_forecast_download(future_df, summary)

    # --- 6. Kết thúc render ---


if __name__ == "__main__":
    main()