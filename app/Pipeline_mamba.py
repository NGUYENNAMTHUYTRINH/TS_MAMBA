"""
pipeline_mamba.py
-----------------
Mamba model (TabularMambaRegressor), training loop, evaluate, và train_pipeline.
"""

from __future__ import annotations

import os
import sys
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
for path in [APP_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.metrics import compute_metrics, denormalize
from Utils import (
    build_future_24h_frame,
    format_time_utc_strings,
    load_train_module,
    normalize_locations,
)


# ---------------------------------------------------------------------------
# Evaluate helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, device, y_mean: float, y_std: float) -> dict:
    """Evaluate Mamba model trên một DataLoader."""
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    preds_norm, targets_norm = [], []

    for xb, loc_ids, yb in loader:
        xb, loc_ids, yb = xb.to(device), loc_ids.to(device), yb.to(device)
        out = model(xb, loc_ids)
        loss = criterion(out, yb)
        total_loss += loss.item() * yb.size(0)
        preds_norm.append(out.detach().cpu().numpy())
        targets_norm.append(yb.detach().cpu().numpy())
        preds.append(out.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())

    preds_arr = denormalize(np.concatenate(preds, axis=0), y_mean, y_std)
    targets_arr = denormalize(np.concatenate(targets, axis=0), y_mean, y_std)

    preds_norm_arr = np.concatenate(preds_norm, axis=0)
    targets_norm_arr = np.concatenate(targets_norm, axis=0)

    metrics = compute_metrics(targets_arr, preds_arr)
    norm_metrics = compute_metrics(targets_norm_arr, preds_norm_arr)

    metrics.update({
        "loss": total_loss / len(loader.dataset),
        "mae_norm": norm_metrics["mae"],
        "rmse_norm": norm_metrics["rmse"],
        "preds": preds_arr,
        "targets": targets_arr,
    })
    return metrics


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def train_pipeline(
    df: pd.DataFrame,
    forecast_base_df: pd.DataFrame | None,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    window_size: int,
    horizon: int,
    sample_stride: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    d_model: int,
    n_layers: int,
    loss_name: str,
    seed: int,
    num_workers: int,
    use_gpu: bool,
    log_interval: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    run_dir: str | None = None,
    forecast_file_name: str = "future_mamba_predictions.csv",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Train Mamba sequence model và trả về (summary, history_df, future_df)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("Cần chọn ít nhất 1 location để train Mamba.")

    work_df = df.copy()
    if "location_key" not in work_df.columns or ("Time" not in work_df.columns and "ts_utc" not in work_df.columns):
        raise ValueError("Dataset train Mamba cần có cột 'location_key' và 'Time' (hoặc 'ts_utc').")

    work_df = work_df.loc[
        work_df["location_key"].astype(str).isin([str(x) for x in selected_locations])
    ].copy()
    if work_df.empty:
        raise ValueError("Không có dữ liệu train cho các location đã chọn.")

    # --- Load helper từ scripts/train_mamba_aqi.py ---
    mod = load_train_module()
    if mod is None or not hasattr(mod, "build_time_series_samples"):
        raise RuntimeError(
            "Không load được module mamba/scripts/train_mamba_aqi.py để chạy Mamba sequence."
        )

    x_seq, loc_ids, y, y_ts, num_locations, ts_feature_cols = mod.build_time_series_samples(
        df=work_df,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
        sample_stride=sample_stride,
        feature_cols=feature_cols,
        include_target_history=True,
    )

    train_split, val_split, test_split = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)

    # --- Normalize ---
    x_mean = train_split.x_seq.mean(axis=(0, 1), keepdims=True)
    x_std = train_split.x_seq.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    for s in [train_split, val_split, test_split]:
        s.x_seq = (s.x_seq - x_mean) / x_std

    y_mean = float(train_split.y.mean())
    y_std = float(train_split.y.std())
    if y_std < 1e-6:
        y_std = 1.0
    for s in [train_split, val_split, test_split]:
        s.y = (s.y - y_mean) / y_std

    train_ds = mod.AQIDataset(train_split)
    val_ds = mod.AQIDataset(val_split)
    test_ds = mod.AQIDataset(test_split)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    pin_memory = device.type == "cuda"
    amp_enabled = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(cpu_threads)

    loader_kwargs: dict = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    raw_model = mod.TimeSeriesMambaRegressor(
        num_features=train_split.x_seq.shape[-1],
        num_locations=num_locations,
        d_model=d_model,
        n_layers=n_layers,
        horizon=horizon,
    ).to(device)
    model = raw_model

    compile_enabled = False
    try:
        import triton  # noqa: F401
        triton_available = True
    except Exception:
        triton_available = False

    if device.type == "cuda" and triton_available and hasattr(torch, "compile"):
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            compile_enabled = True
        except Exception:
            compile_enabled = False

    criterion = nn.HuberLoss(delta=1.0) if loss_name == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict] = []
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_steps = epochs * len(train_loader)
    global_step = 0
    prog = st.progress(0)
    log_box = st.empty()
    log_lines: list[str] = []

    start_all = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, loc_batch, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=pin_memory)
            loc_batch = loc_batch.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)

            amp_context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()
            with amp_context:
                out = model(xb, loc_batch)
                loss = criterion(out, yb)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_for_backward = loss / grad_accum_steps
            if amp_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if amp_enabled:
                    scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * yb.size(0)
            global_step += 1

            if total_steps > 0 and (step % 20 == 0 or step == len(train_loader)):
                prog.progress(min(global_step / total_steps, 1.0))

            if log_interval > 0 and (step % log_interval == 0 or step == len(train_loader)):
                avg_loss = running_loss / max(step * yb.size(0), 1)
                line = (
                    f"Epoch {epoch}/{epochs} | step {step}/{len(train_loader)} | "
                    f"batch_loss={loss.item():.6f} | running_avg={avg_loss:.6f}"
                )
                log_lines.append(line)
                log_box.code("\n".join(log_lines[-20:]))

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = mod.evaluate(model, val_loader, criterion, device, amp_enabled, y_mean, y_std)
        epoch_sec = time.time() - epoch_start

        epoch_line = (
            f"Epoch {epoch}/{epochs} done | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | mae={val_metrics.get('mae_norm', float('nan')):.4f} | "
            f"rmse={val_metrics.get('rmse_norm', float('nan')):.4f} | val_r2={val_metrics['r2']:.4f} | "
            f"sec={epoch_sec:.1f}"
        )
        log_lines.append(epoch_line)
        log_box.code("\n".join(log_lines[-20:]))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "mae": val_metrics.get("mae_norm"),
                "rmse": val_metrics.get("rmse_norm"),
                "val_r2": val_metrics["r2"],
                "train_sec": epoch_sec,
            }
        )

        if val_metrics["loss"] < best_val_loss - early_stop_min_delta:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            stopped_early = True
            stop_line = (
                f"Early stopping at epoch {epoch}/{epochs} | "
                f"best_val_loss={best_val_loss:.6f}"
            )
            log_lines.append(stop_line)
            log_box.code("\n".join(log_lines[-20:]))
            break

    if best_state is None:
        raise RuntimeError("Không có checkpoint hợp lệ trong quá trình train.")

    raw_model.load_state_dict(best_state)
    raw_model.to(device)
    eval_start = time.time()
    val_metrics = mod.evaluate(model, val_loader, criterion, device, amp_enabled, y_mean, y_std)
    test_metrics = mod.evaluate(model, test_loader, criterion, device, amp_enabled, y_mean, y_std)
    eval_sec = time.time() - eval_start

    # --- Build loc_to_id mapping ---
    cleaned = work_df.copy()
    ts_col = "Time" if "Time" in cleaned.columns else "ts_utc"
    cleaned["_ts"] = pd.to_datetime(cleaned[ts_col], utc=True, errors="coerce")
    cleaned = cleaned.dropna(subset=["_ts", "location_key", target_col]).copy()
    cleaned["_loc_id"] = cleaned["location_key"].astype("category").cat.codes.astype(np.int64)
    loc_to_id = (
        cleaned.assign(_loc_key_str=cleaned["location_key"].astype(str))
        .drop_duplicates(subset=["_loc_key_str"])
        .set_index("_loc_key_str")["_loc_id"]
        .to_dict()
    )

    # --- Forecast 24h ---
    base_df = forecast_base_df if forecast_base_df is not None else cleaned.copy()
    if not isinstance(base_df, pd.DataFrame) or base_df.empty:
        raise ValueError("Không có dữ liệu test làm mốc để dự báo 24h tiếp theo.")

    base_df = base_df.loc[
        base_df["location_key"].astype(str).isin(list(loc_to_id.keys()))
    ].copy()
    if base_df.empty:
        raise ValueError("Test CSV không có location trùng với dữ liệu train đã chọn.")

    base_col_map = {c.lower(): c for c in base_df.columns}
    base_ts_col = base_col_map.get("ts_utc") or base_col_map.get("time") or base_col_map.get("timestamp") or "_ts"
    if base_ts_col not in base_df.columns:
        raise ValueError("Cần có cột timestamp ('ts_utc', 'Time', hoặc 'timestamp') để dự báo 24h tiếp theo.")
    if base_ts_col != "ts_utc":
        base_df["ts_utc"] = base_df[base_ts_col]

    for col in ts_feature_cols:
        if col not in base_df.columns:
            base_df[col] = np.nan
        base_df[col] = pd.to_numeric(base_df[col], errors="coerce")
        fill_val = base_df[col].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        base_df[col] = base_df[col].fillna(fill_val)

    future_df = build_future_24h_frame(base_df, feature_cols=ts_feature_cols, target_col=target_col)
    for col in ts_feature_cols:
        future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
        fill_val = base_df[col].median() if col in base_df.columns else 0.0
        if pd.isna(fill_val):
            fill_val = 0.0
        future_df[col] = future_df[col].fillna(fill_val)

    # --- Batched inference ---
    forecast_start = time.time()
    preds_rows: list[dict] = []
    model.eval()
    infer_x, infer_loc, infer_meta = [], [], []
    x_mean_2d = x_mean.squeeze(0)
    x_std_2d = x_std.squeeze(0)

    with torch.inference_mode():
        for loc in sorted(future_df["location_key"].astype(str).unique().tolist()):
            if loc not in loc_to_id:
                continue
            loc_hist = (
                base_df.loc[base_df["location_key"].astype(str) == loc]
                .copy()
                .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
                .dropna(subset=["ts_utc"])
                .sort_values("ts_utc")
            )
            if len(loc_hist) < window_size:
                continue

            rolling_window = loc_hist[ts_feature_cols].tail(window_size).to_numpy(dtype=np.float32)
            loc_future = (
                future_df.loc[future_df["location_key"].astype(str) == loc]
                .copy()
                .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
                .sort_values("ts_utc")
            )

            for _, row in loc_future.iterrows():
                x_norm = (rolling_window - x_mean_2d) / x_std_2d
                infer_x.append(x_norm.astype(np.float32, copy=False))
                infer_loc.append(int(loc_to_id[loc]))
                infer_meta.append((row["ts_utc"], loc))
                next_feats = row[ts_feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
                rolling_window = np.concatenate([rolling_window[1:], next_feats], axis=0)

        if infer_x:
            x_all = torch.from_numpy(np.stack(infer_x, axis=0)).to(device, non_blocking=pin_memory)
            loc_all = torch.tensor(infer_loc, dtype=torch.long, device=device)
            amp_context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()
            with amp_context:
                pred_norm_all = model(x_all, loc_all).detach().float().cpu().numpy()

            pred_all = pred_norm_all * y_std + y_mean
            for (ts_val, loc_val), pred_val in zip(infer_meta, pred_all):
                pred_scalar = float(np.asarray(pred_val, dtype=np.float32).reshape(-1)[0])
                preds_rows.append({"time": ts_val, "location": loc_val, "predicted": pred_scalar})

    forecast_sec = time.time() - forecast_start

    if not preds_rows:
        raise RuntimeError("Không tạo được dự báo 24h cho Mamba sequence.")

    future_out = (
        pd.DataFrame(preds_rows)
        .assign(time=lambda d: format_time_utc_strings(d["time"]))
        [["time", "location", "predicted"]]
        .sort_values(["location", "time"])
        .reset_index(drop=True)
    )

    # --- Lưu artifacts ---
    if run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "streamlit_runs", run_id)
    else:
        out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "best_mamba_aqi.pt")
    metrics_path = os.path.join(out_dir, "metrics_history.csv")
    future_pred_path = os.path.join(out_dir, forecast_file_name)

    io_start = time.time()
    torch.save(raw_model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    future_out.to_csv(future_pred_path, index=False)
    io_sec = time.time() - io_start

    summary = {
        "device": str(device),
        "torch_compile": bool(compile_enabled),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(num_workers > 0),
        "grad_accum_steps": int(grad_accum_steps),
        "n_rows_used": len(y),
        "split_train": len(train_split.y),
        "split_val": len(val_split.y),
        "split_test": len(test_split.y),
        "feature_count_after_encode": train_split.x_seq.shape[-1],
        "sample_stride": int(sample_stride),
        "encoded_features": ts_feature_cols,
        "val_loss": val_metrics["loss"],
        "val_r2": val_metrics["r2"],
        "val_mae": val_metrics.get("mae"),
        "val_rmse": val_metrics.get("rmse"),
        "val_mae_norm": val_metrics.get("mae_norm"),
        "val_rmse_norm": val_metrics.get("rmse_norm"),
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mae_norm": test_metrics.get("mae_norm"),
        "test_rmse_norm": test_metrics.get("rmse_norm"),
        "model_path": model_path,
        "metrics_path": metrics_path,
        "future_pred_path": future_pred_path,
        "future_rows": len(future_out),
        "future_locations": int(future_out["location"].nunique()),
        "per_location_files": [],
        "epochs_ran": len(history),
        "stopped_early": stopped_early,
        "best_val_loss": float(best_val_loss),
        "train_only_sec": float(pd.DataFrame(history)["train_sec"].sum()) if history else 0.0,
        "eval_sec": float(eval_sec),
        "forecast_sec": float(forecast_sec),
        "io_sec": float(io_sec),
        "run_sec": time.time() - start_all,
    }

    return summary, pd.DataFrame(history), future_out


def predict_with_saved_model(
    df: pd.DataFrame,
    forecast_base_df: pd.DataFrame | None,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    window_size: int,
    horizon: int,
    sample_stride: int,
    loss_name: str,
    batch_size: int,
    use_gpu: bool,
    checkpoint_path: str,
    run_dir: str | None = None,
    forecast_file_name: str = "future_mamba_predictions.csv",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load checkpoint da train va forecast, khong train lai model."""
    start_all = time.time()
    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("Can chon it nhat 1 location de du doan Mamba.")

    ckpt_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    inferred_d_model = int(state["input_proj.weight"].shape[0])
    inferred_horizon = int(state["head.2.weight"].shape[0])
    inferred_layers = [
        int(key.split(".")[1])
        for key in state
        if key.startswith("layers.") and key.split(".")[1].isdigit()
    ]
    inferred_n_layers = max(inferred_layers) + 1 if inferred_layers else 2
    inferred_num_locations = int(state["location_emb.weight"].shape[0])
    loc_embed_dim = int(state["location_emb.weight"].shape[1])
    inferred_num_features = int(state["input_proj.weight"].shape[1]) - loc_embed_dim

    d_model = inferred_d_model
    horizon = inferred_horizon
    n_layers = inferred_n_layers

    work_df = df.copy()
    if "location_key" not in work_df.columns or ("Time" not in work_df.columns and "ts_utc" not in work_df.columns):
        raise ValueError("Dataset can co cot 'location_key' va 'Time' (hoac 'ts_utc').")

    work_df = work_df.loc[
        work_df["location_key"].astype(str).isin([str(x) for x in selected_locations])
    ].copy()
    if work_df.empty:
        raise ValueError("Khong co du lieu cho cac location da chon.")

    mod = load_train_module()
    if mod is None or not hasattr(mod, "build_time_series_samples"):
        raise RuntimeError("Khong load duoc helper build_time_series_samples tu mamba/train_mamba_aqi.py.")

    x_seq, loc_ids, y, y_ts, num_locations, ts_feature_cols = mod.build_time_series_samples(
        df=work_df,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
        sample_stride=sample_stride,
        feature_cols=feature_cols,
        include_target_history=True,
    )

    train_split, val_split, test_split = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)

    if train_split.x_seq.shape[-1] != inferred_num_features:
        raise ValueError(
            "So luong feature trong UI/dataset khong khop checkpoint: "
            f"dataset co {train_split.x_seq.shape[-1]}, checkpoint can {inferred_num_features}. "
            "Hay chon dung cac input feature columns nhu luc train."
        )
    if num_locations != inferred_num_locations:
        raise ValueError(
            "So luong location khong khop checkpoint: "
            f"dataset dang chon {num_locations}, checkpoint can {inferred_num_locations}. "
            "Hay chon dung location da dung khi train."
        )

    x_mean = train_split.x_seq.mean(axis=(0, 1), keepdims=True)
    x_std = train_split.x_seq.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    for s in [train_split, val_split, test_split]:
        s.x_seq = (s.x_seq - x_mean) / x_std

    y_mean = float(train_split.y.mean())
    y_std = float(train_split.y.std())
    if y_std < 1e-6:
        y_std = 1.0
    for s in [train_split, val_split, test_split]:
        s.y = (s.y - y_mean) / y_std

    val_ds = mod.AQIDataset(val_split)
    test_ds = mod.AQIDataset(test_split)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    pin_memory = device.type == "cuda"
    amp_enabled = device.type == "cuda"

    loader_kwargs: dict = {"batch_size": batch_size, "num_workers": 0, "pin_memory": pin_memory}
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = mod.TimeSeriesMambaRegressor(
        num_features=train_split.x_seq.shape[-1],
        num_locations=num_locations,
        d_model=d_model,
        n_layers=n_layers,
        horizon=horizon,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    criterion = nn.HuberLoss(delta=1.0) if loss_name == "huber" else nn.MSELoss()
    eval_start = time.time()
    val_metrics = evaluate(model, val_loader, criterion, device, y_mean, y_std)
    test_metrics = evaluate(model, test_loader, criterion, device, y_mean, y_std)
    eval_sec = time.time() - eval_start

    cleaned = work_df.copy()
    ts_col = "Time" if "Time" in cleaned.columns else "ts_utc"
    cleaned["_ts"] = pd.to_datetime(cleaned[ts_col], utc=True, errors="coerce")
    cleaned = cleaned.dropna(subset=["_ts", "location_key", target_col]).copy()
    cleaned["_loc_id"] = cleaned["location_key"].astype("category").cat.codes.astype(np.int64)
    loc_to_id = (
        cleaned.assign(_loc_key_str=cleaned["location_key"].astype(str))
        .drop_duplicates(subset=["_loc_key_str"])
        .set_index("_loc_key_str")["_loc_id"]
        .to_dict()
    )

    base_df = forecast_base_df if forecast_base_df is not None else cleaned.copy()
    if not isinstance(base_df, pd.DataFrame) or base_df.empty:
        raise ValueError("Khong co du lieu moc de du bao.")

    base_df = base_df.loc[
        base_df["location_key"].astype(str).isin(list(loc_to_id.keys()))
    ].copy()
    if base_df.empty:
        raise ValueError("Du lieu forecast khong co location trung voi model.")

    base_col_map = {c.lower(): c for c in base_df.columns}
    base_ts_col = base_col_map.get("ts_utc") or base_col_map.get("time") or base_col_map.get("timestamp") or "_ts"
    if base_ts_col not in base_df.columns:
        raise ValueError("Can co cot timestamp ('ts_utc', 'Time', hoac 'timestamp') de du bao.")
    if base_ts_col != "ts_utc":
        base_df["ts_utc"] = base_df[base_ts_col]

    for col in ts_feature_cols:
        if col not in base_df.columns:
            base_df[col] = np.nan
        base_df[col] = pd.to_numeric(base_df[col], errors="coerce")
        fill_val = base_df[col].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        base_df[col] = base_df[col].fillna(fill_val)

    future_df = build_future_24h_frame(
        base_df,
        feature_cols=ts_feature_cols,
        target_col=target_col,
        hours=horizon,
    )
    for col in ts_feature_cols:
        future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
        fill_val = base_df[col].median() if col in base_df.columns else 0.0
        if pd.isna(fill_val):
            fill_val = 0.0
        future_df[col] = future_df[col].fillna(fill_val)

    forecast_start = time.time()
    preds_rows: list[dict] = []
    infer_x, infer_loc, infer_meta = [], [], []
    x_mean_2d = x_mean.squeeze(0)
    x_std_2d = x_std.squeeze(0)

    with torch.inference_mode():
        for loc in sorted(future_df["location_key"].astype(str).unique().tolist()):
            if loc not in loc_to_id:
                continue
            loc_hist = (
                base_df.loc[base_df["location_key"].astype(str) == loc]
                .copy()
                .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
                .dropna(subset=["ts_utc"])
                .sort_values("ts_utc")
            )
            if len(loc_hist) < window_size:
                continue

            rolling_window = loc_hist[ts_feature_cols].tail(window_size).to_numpy(dtype=np.float32)
            loc_future = (
                future_df.loc[future_df["location_key"].astype(str) == loc]
                .copy()
                .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
                .sort_values("ts_utc")
            )

            for _, row in loc_future.iterrows():
                x_norm = (rolling_window - x_mean_2d) / x_std_2d
                infer_x.append(x_norm.astype(np.float32, copy=False))
                infer_loc.append(int(loc_to_id[loc]))
                infer_meta.append((row["ts_utc"], loc))
                next_feats = row[ts_feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
                rolling_window = np.concatenate([rolling_window[1:], next_feats], axis=0)

        if infer_x:
            x_all = torch.from_numpy(np.stack(infer_x, axis=0)).to(device, non_blocking=pin_memory)
            loc_all = torch.tensor(infer_loc, dtype=torch.long, device=device)
            amp_context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()
            with amp_context:
                pred_norm_all = model(x_all, loc_all).detach().float().cpu().numpy()

            pred_all = pred_norm_all * y_std + y_mean
            for (ts_val, loc_val), pred_val in zip(infer_meta, pred_all):
                pred_scalar = float(np.asarray(pred_val, dtype=np.float32).reshape(-1)[0])
                preds_rows.append({"time": ts_val, "location": loc_val, "predicted": pred_scalar})

    forecast_sec = time.time() - forecast_start

    if not preds_rows:
        raise RuntimeError("Khong tao duoc du bao tu checkpoint da train.")

    future_out = (
        pd.DataFrame(preds_rows)
        .assign(time=lambda d: format_time_utc_strings(d["time"]))
        [["time", "location", "predicted"]]
        .sort_values(["location", "time"])
        .reset_index(drop=True)
    )

    if run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "streamlit_saved_model_runs", run_id)
    else:
        out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "loaded_model_metrics.csv")
    future_pred_path = os.path.join(out_dir, forecast_file_name)
    io_start = time.time()
    hist_df = pd.read_csv(os.path.join(os.path.dirname(ckpt_path), "metrics_history.csv")) if os.path.exists(os.path.join(os.path.dirname(ckpt_path), "metrics_history.csv")) else pd.DataFrame()
    pd.DataFrame([{"split": "val", **val_metrics}, {"split": "test", **test_metrics}]).drop(
        columns=["preds", "targets"], errors="ignore"
    ).to_csv(metrics_path, index=False)
    future_out.to_csv(future_pred_path, index=False)
    io_sec = time.time() - io_start

    summary = {
        "mode": "loaded_checkpoint",
        "device": str(device),
        "torch_compile": False,
        "num_workers": 0,
        "pin_memory": bool(pin_memory),
        "persistent_workers": False,
        "grad_accum_steps": 0,
        "n_rows_used": len(y),
        "split_train": len(train_split.y),
        "split_val": len(val_split.y),
        "split_test": len(test_split.y),
        "feature_count_after_encode": train_split.x_seq.shape[-1],
        "sample_stride": int(sample_stride),
        "encoded_features": ts_feature_cols,
        "val_loss": val_metrics["loss"],
        "val_r2": val_metrics["r2"],
        "val_mae": val_metrics.get("mae"),
        "val_rmse": val_metrics.get("rmse"),
        "val_mae_norm": val_metrics.get("mae_norm"),
        "val_rmse_norm": val_metrics.get("rmse_norm"),
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mae_norm": test_metrics.get("mae_norm"),
        "test_rmse_norm": test_metrics.get("rmse_norm"),
        "model_path": ckpt_path,
        "metrics_path": metrics_path,
        "future_pred_path": future_pred_path,
        "future_rows": len(future_out),
        "future_locations": int(future_out["location"].nunique()),
        "per_location_files": [],
        "epochs_ran": 0,
        "stopped_early": False,
        "best_val_loss": float(val_metrics["loss"]),
        "train_only_sec": 0.0,
        "eval_sec": float(eval_sec),
        "forecast_sec": float(forecast_sec),
        "io_sec": float(io_sec),
        "run_sec": time.time() - start_all,
    }

    return summary, hist_df, future_out
