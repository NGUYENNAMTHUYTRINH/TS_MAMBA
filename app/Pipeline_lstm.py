"""
pipeline_lstm.py
----------------
Streamlit pipeline for the LSTM model using the same input/output contract as
Pipeline_mamba.py.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from core.metrics import compute_metrics

LSTM_ROOT = APP_ROOT / "LSTM-Time-Series-Forecasting"
if str(LSTM_ROOT) not in sys.path:
    sys.path.insert(0, str(LSTM_ROOT))

from src.model import ImprovedLSTMForecaster


def _timestamp_col(df: pd.DataFrame) -> str:
    col_map = {c.lower(): c for c in df.columns}
    ts_col = col_map.get("time") or col_map.get("ts_utc") or col_map.get("timestamp")
    if ts_col is None:
        raise ValueError("Dataset can co cot Time, ts_utc hoac timestamp.")
    return ts_col


def _normalize_locations(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def _make_windows(values: np.ndarray, target_idx: int, lookback: int, horizon: int):
    xs, ys = [], []
    max_start = len(values) - lookback - horizon + 1
    for start in range(max_start):
        end = start + lookback
        xs.append(values[start:end])
        ys.append(values[end:end + horizon, target_idx])
    if not xs:
        return np.empty((0, lookback, values.shape[1]), dtype=np.float32), np.empty((0, horizon), dtype=np.float32)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _prepare_lstm_data(
    df: pd.DataFrame,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    lookback: int,
    horizon: int,
):
    selected_locations = _normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("Can chon it nhat 1 location.")
    if "location_key" not in df.columns:
        raise ValueError("Dataset can co cot location_key.")

    ts_col = _timestamp_col(df)
    work = df.copy()
    work["_ts"] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    work = work.loc[work["location_key"].astype(str).isin(selected_locations)].copy()
    work = work.dropna(subset=["_ts", "location_key", target_col]).sort_values(["location_key", "_ts"])
    if work.empty:
        raise ValueError("Khong co du lieu cho location da chon.")

    feat_cols = [c for c in feature_cols if c in work.columns and c != "_loc_id"]
    if target_col not in feat_cols:
        feat_cols.append(target_col)
    if target_col not in feat_cols:
        raise ValueError(f"Target {target_col} khong nam trong feature columns.")
    target_idx = feat_cols.index(target_col)

    for col in feat_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    all_locs = sorted(work["location_key"].dropna().astype(str).unique().tolist())
    loc_to_id = {loc: idx for idx, loc in enumerate(all_locs)}
    scalers: dict[str, StandardScaler] = {}
    x_parts, y_parts, loc_parts = [], [], []

    for loc in all_locs:
        grp = work.loc[work["location_key"].astype(str) == loc].copy()
        grp = grp.dropna(subset=feat_cols).sort_values("_ts")
        if len(grp) < lookback + horizon:
            continue

        scaler = StandardScaler()
        scaled = scaler.fit_transform(grp[feat_cols].to_numpy(dtype=np.float32))
        x_loc, y_loc = _make_windows(scaled, target_idx, lookback, horizon)
        if len(x_loc) == 0:
            continue
        scalers[loc] = scaler
        x_parts.append(x_loc)
        y_parts.append(y_loc)
        loc_parts.append(np.full(len(x_loc), loc_to_id[loc], dtype=np.int64))

    if not x_parts:
        raise ValueError("Khong tao duoc window nao cho LSTM.")

    x_all = np.concatenate(x_parts).astype(np.float32)
    y_all = np.concatenate(y_parts).astype(np.float32)
    loc_all = np.concatenate(loc_parts).astype(np.int64)

    n = len(y_all)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Khong du sample de chia train/val/test.")

    return {
        "work": work,
        "x": x_all,
        "y": y_all,
        "loc_ids": loc_all,
        "train_idx": np.arange(0, train_end),
        "val_idx": np.arange(train_end, val_end),
        "test_idx": np.arange(val_end, n),
        "feat_cols": feat_cols,
        "target_idx": target_idx,
        "loc_to_id": loc_to_id,
        "scalers": scalers,
    }


@torch.no_grad()
def _inverse_scaled_target(
    values: np.ndarray,
    loc_ids: np.ndarray,
    id_to_loc: dict[int, str],
    scalers: dict[str, StandardScaler],
    target_idx: int,
    num_features: int,
) -> np.ndarray:
    actual_parts = []
    for row, loc_id in zip(values, loc_ids):
        loc = id_to_loc[int(loc_id)]
        scaler = scalers[loc]
        dummy = np.zeros((row.shape[0], num_features), dtype=np.float32)
        dummy[:, target_idx] = row
        actual_parts.append(scaler.inverse_transform(dummy)[:, target_idx])
    return np.concatenate(actual_parts).reshape(-1)


@torch.no_grad()
def _evaluate(
    model,
    loader,
    criterion,
    device,
    id_to_loc: dict[int, str],
    scalers: dict[str, StandardScaler],
    target_idx: int,
    num_features: int,
) -> dict:
    model.eval()
    total = 0.0
    preds, targets, locs = [], [], []
    for xb, yb, lb in loader:
        xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
        out = model(xb, lb)
        loss = criterion(out, yb)
        total += loss.item() * xb.size(0)
        preds.append(out.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())
        locs.append(lb.detach().cpu().numpy())

    pred_norm_2d = np.concatenate(preds)
    true_norm_2d = np.concatenate(targets)
    loc_ids = np.concatenate(locs)
    pred_norm = pred_norm_2d.reshape(-1)
    true_norm = true_norm_2d.reshape(-1)
    pred = _inverse_scaled_target(pred_norm_2d, loc_ids, id_to_loc, scalers, target_idx, num_features)
    true = _inverse_scaled_target(true_norm_2d, loc_ids, id_to_loc, scalers, target_idx, num_features)
    metrics = compute_metrics(true, pred)
    norm_metrics = compute_metrics(true_norm, pred_norm)
    metrics.update({
        "loss": total / len(loader.dataset),
        "mae_norm": norm_metrics["mae"],
        "rmse_norm": norm_metrics["rmse"],
    })
    return metrics


def _prepare_lstm_eval_data(
    df: pd.DataFrame,
    selected_locations: list[str],
    target_col: str,
    feat_cols: list[str],
    target_idx: int,
    lookback: int,
    horizon: int,
    loc_to_id: dict[str, int],
    scalers: dict[str, StandardScaler],
):
    if "location_key" not in df.columns:
        raise ValueError("Dataset can co cot location_key.")
    ts_col = _timestamp_col(df)
    selected = set(_normalize_locations(selected_locations))
    work = df.copy()
    work["_ts"] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    work = work.dropna(subset=["_ts", "location_key", target_col]).sort_values(["location_key", "_ts"])
    if selected:
        work = work.loc[work["location_key"].astype(str).isin(selected)].copy()
    for col in feat_cols:
        if col not in work.columns:
            raise ValueError(f"Dataset thieu cot feature cua LSTM checkpoint: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce")

    x_parts, y_parts, loc_parts = [], [], []
    for loc in sorted(loc_to_id):
        if loc not in scalers:
            continue
        grp = (
            work.loc[work["location_key"].astype(str) == loc]
            .copy()
            .dropna(subset=feat_cols)
            .sort_values("_ts")
        )
        if len(grp) < lookback + horizon:
            continue
        scaled = scalers[loc].transform(grp[feat_cols].to_numpy(dtype=np.float32))
        x_loc, y_loc = _make_windows(scaled, target_idx, lookback, horizon)
        if len(x_loc) == 0:
            continue
        x_parts.append(x_loc)
        y_parts.append(y_loc)
        loc_parts.append(np.full(len(x_loc), loc_to_id[loc], dtype=np.int64))

    if not x_parts:
        raise ValueError("Khong tao duoc window nao de tinh metric LSTM.")

    x_all = np.concatenate(x_parts).astype(np.float32)
    y_all = np.concatenate(y_parts).astype(np.float32)
    loc_all = np.concatenate(loc_parts).astype(np.int64)
    n = len(y_all)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    return {
        "x": x_all,
        "y": y_all,
        "loc_ids": loc_all,
        "train_idx": np.arange(0, train_end),
        "val_idx": np.arange(train_end, val_end),
        "test_idx": np.arange(val_end, n),
    }


def _forecast_from_model(
    model,
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    target_idx: int,
    loc_to_id: dict[str, int],
    scalers: dict[str, StandardScaler],
    lookback: int,
    horizon: int,
    device,
) -> pd.DataFrame:
    ts_col = _timestamp_col(df)
    work = df.copy()
    work["_ts"] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    rows = []
    model.eval()

    with torch.inference_mode():
        for loc in sorted(loc_to_id):
            grp = (
                work.loc[work["location_key"].astype(str) == loc]
                .copy()
                .dropna(subset=["_ts"])
                .sort_values("_ts")
            )
            if loc not in scalers or len(grp) < lookback:
                continue
            for col in feat_cols:
                grp[col] = pd.to_numeric(grp[col], errors="coerce")
            grp = grp.dropna(subset=feat_cols)
            if len(grp) < lookback:
                continue

            scaler = scalers[loc]
            window_raw = grp[feat_cols].tail(lookback).to_numpy(dtype=np.float32)
            window_scaled = scaler.transform(window_raw).astype(np.float32)
            xb = torch.from_numpy(window_scaled[None, :, :]).to(device)
            lb = torch.tensor([loc_to_id[loc]], dtype=torch.long, device=device)
            pred_scaled = model(xb, lb).detach().cpu().numpy()[0]

            dummy = np.zeros((horizon, len(feat_cols)), dtype=np.float32)
            dummy[:, target_idx] = pred_scaled
            pred = scaler.inverse_transform(dummy)[:, target_idx]
            last_ts = grp["_ts"].max()
            for step, value in enumerate(pred, start=1):
                rows.append({
                    "time": last_ts + pd.Timedelta(hours=step),
                    "location": loc,
                    "predicted": float(value),
                })

    if not rows:
        raise RuntimeError("Khong tao duoc du bao LSTM.")
    return (
        pd.DataFrame(rows)
        .assign(time=lambda d: pd.to_datetime(d["time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
        [["time", "location", "predicted"]]
        .sort_values(["location", "time"])
        .reset_index(drop=True)
    )


def train_lstm_pipeline(
    df: pd.DataFrame,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    window_size: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    num_layers: int,
    loss_name: str,
    seed: int,
    use_gpu: bool,
    early_stop_patience: int,
    run_dir: str | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_all = time.time()
    data = _prepare_lstm_data(df, selected_locations, target_col, feature_cols, window_size, horizon)
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    def loader(idx, shuffle=False):
        ds = TensorDataset(
            torch.from_numpy(data["x"][idx]).float(),
            torch.from_numpy(data["y"][idx]).float(),
            torch.from_numpy(data["loc_ids"][idx]).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda")

    train_loader = loader(data["train_idx"], shuffle=True)
    val_loader = loader(data["val_idx"])
    test_loader = loader(data["test_idx"])
    id_to_loc = {idx: loc for loc, idx in data["loc_to_id"].items()}

    model = ImprovedLSTMForecaster(
        input_size=len(data["feat_cols"]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon=horizon,
        num_locations=len(data["loc_to_id"]),
        embed_dim=16,
    ).to(device)

    criterion = nn.MSELoss() if loss_name == "mse" else nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total = 0.0
        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, lb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * xb.size(0)

        train_loss = total / len(train_loader.dataset)
        val_metrics = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            id_to_loc,
            data["scalers"],
            data["target_idx"],
            len(data["feat_cols"]),
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "mae": val_metrics["mae"],
            "rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "train_sec": time.time() - epoch_start,
        })

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("Khong co checkpoint LSTM hop le.")

    model.load_state_dict(best_state)
    model.to(device)
    val_metrics = _evaluate(
        model,
        val_loader,
        criterion,
        device,
        id_to_loc,
        data["scalers"],
        data["target_idx"],
        len(data["feat_cols"]),
    )
    test_metrics = _evaluate(
        model,
        test_loader,
        criterion,
        device,
        id_to_loc,
        data["scalers"],
        data["target_idx"],
        len(data["feat_cols"]),
    )
    future_out = _forecast_from_model(
        model, data["work"], data["feat_cols"], target_col, data["target_idx"],
        data["loc_to_id"], data["scalers"], window_size, horizon, device
    )

    out_dir = run_dir or os.path.join("runs", "lstm", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "best_lstm.pt")
    scaler_path = os.path.join(out_dir, "scalers.pkl")
    history_path = os.path.join(out_dir, "training_history.csv")
    future_path = os.path.join(out_dir, "future_lstm_predictions.csv")

    torch.save({
        "model_state": best_state,
        "horizon": horizon,
        "lookback": window_size,
        "feat_cols": data["feat_cols"],
        "target_idx": data["target_idx"],
        "num_locations": len(data["loc_to_id"]),
        "loc_to_id": data["loc_to_id"],
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "embed_dim": 16,
    }, model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "scalers": data["scalers"],
            "loc_to_id": data["loc_to_id"],
            "feat_cols": data["feat_cols"],
            "target_idx": data["target_idx"],
        }, f)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(history_path, index=False)
    future_out.to_csv(future_path, index=False)

    summary = {
        "mode": "train_lstm",
        "device": str(device),
        "n_rows_used": len(data["y"]),
        "split_train": len(data["train_idx"]),
        "split_val": len(data["val_idx"]),
        "split_test": len(data["test_idx"]),
        "feature_count_after_encode": len(data["feat_cols"]),
        "sample_stride": 1,
        "encoded_features": data["feat_cols"],
        "val_loss": val_metrics["loss"],
        "val_r2": val_metrics["r2"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_mae_norm": val_metrics["mae_norm"],
        "val_rmse_norm": val_metrics["rmse_norm"],
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mae_norm": test_metrics["mae_norm"],
        "test_rmse_norm": test_metrics["rmse_norm"],
        "model_path": model_path,
        "metrics_path": history_path,
        "future_pred_path": future_path,
        "future_rows": len(future_out),
        "future_locations": int(future_out["location"].nunique()),
        "epochs_ran": len(hist_df),
        "train_only_sec": float(hist_df["train_sec"].sum()) if not hist_df.empty else 0.0,
        "eval_sec": 0.0,
        "forecast_sec": 0.0,
        "run_sec": time.time() - start_all,
    }
    return summary, hist_df, future_out


def predict_lstm_with_saved_model(
    df: pd.DataFrame,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    checkpoint_path: str,
    batch_size: int,
    use_gpu: bool,
    run_dir: str | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    start_all = time.time()
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Khong tim thay checkpoint LSTM: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    lookback = int(ckpt["lookback"])
    horizon = int(ckpt["horizon"])
    feat_cols = list(ckpt["feat_cols"])
    target_idx = int(ckpt["target_idx"])
    loc_to_id = {str(k): int(v) for k, v in ckpt.get("loc_to_id", {}).items()}
    if not loc_to_id:
        selected = _normalize_locations(selected_locations)
        loc_to_id = {loc: idx for idx, loc in enumerate(sorted(selected))}

    scaler_path = ckpt_path.parent / "scalers.pkl"
    if not scaler_path.exists():
        single_scaler = ckpt_path.parent / "scaler.pkl"
        scaler_path = single_scaler if single_scaler.exists() else scaler_path
    if not scaler_path.exists():
        raise FileNotFoundError(f"Khong tim thay scaler LSTM trong {ckpt_path.parent}")

    with open(scaler_path, "rb") as f:
        bundle = pickle.load(f)
    if "scalers" in bundle:
        scalers = {str(k): v for k, v in bundle["scalers"].items()}
        if not loc_to_id:
            loc_to_id = {str(k): int(v) for k, v in bundle.get("loc_to_id", bundle.get("loc2idx", {})).items()}
    else:
        scaler = bundle["scaler"]
        scalers = {loc: scaler for loc in loc_to_id}

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = ImprovedLSTMForecaster(
        input_size=len(feat_cols),
        hidden_size=int(ckpt["hidden_size"]),
        num_layers=int(ckpt["num_layers"]),
        horizon=horizon,
        num_locations=int(ckpt["num_locations"]),
        embed_dim=int(ckpt.get("embed_dim", 16)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    selected = set(_normalize_locations(selected_locations))
    if selected:
        loc_to_id = {loc: idx for loc, idx in loc_to_id.items() if loc in selected}

    eval_start = time.time()
    eval_data = _prepare_lstm_eval_data(
        df=df,
        selected_locations=selected_locations,
        target_col=target_col,
        feat_cols=feat_cols,
        target_idx=target_idx,
        lookback=lookback,
        horizon=horizon,
        loc_to_id=loc_to_id,
        scalers=scalers,
    )
    id_to_loc = {idx: loc for loc, idx in loc_to_id.items()}
    criterion = nn.MSELoss()

    def loader(idx):
        ds = TensorDataset(
            torch.from_numpy(eval_data["x"][idx]).float(),
            torch.from_numpy(eval_data["y"][idx]).float(),
            torch.from_numpy(eval_data["loc_ids"][idx]).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    val_metrics = _evaluate(
        model,
        loader(eval_data["val_idx"]),
        criterion,
        device,
        id_to_loc,
        scalers,
        target_idx,
        len(feat_cols),
    )
    test_metrics = _evaluate(
        model,
        loader(eval_data["test_idx"]),
        criterion,
        device,
        id_to_loc,
        scalers,
        target_idx,
        len(feat_cols),
    )
    eval_sec = time.time() - eval_start

    forecast_start = time.time()
    future_out = _forecast_from_model(
        model, df, feat_cols, target_col, target_idx, loc_to_id,
        scalers, lookback, horizon, device
    )
    forecast_sec = time.time() - forecast_start

    out_dir = run_dir or str(ckpt_path.parent)
    os.makedirs(out_dir, exist_ok=True)
    future_path = os.path.join(out_dir, "future_lstm_predictions.csv")
    metrics_path = os.path.join(out_dir, "loaded_lstm_metrics.csv")
    future_out.to_csv(future_path, index=False)
    pd.DataFrame([{"split": "val", **val_metrics}, {"split": "test", **test_metrics}]).to_csv(metrics_path, index=False)

    hist_path = ckpt_path.parent / "training_history.csv"
    hist_df = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    summary = {
        "mode": "loaded_lstm",
        "device": str(device),
        "n_rows_used": len(eval_data["y"]),
        "split_train": len(eval_data["train_idx"]),
        "split_val": len(eval_data["val_idx"]),
        "split_test": len(eval_data["test_idx"]),
        "feature_count_after_encode": len(feat_cols),
        "sample_stride": 1,
        "encoded_features": feat_cols,
        "val_loss": val_metrics["loss"],
        "val_r2": val_metrics["r2"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_mae_norm": val_metrics["mae_norm"],
        "val_rmse_norm": val_metrics["rmse_norm"],
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mae_norm": test_metrics["mae_norm"],
        "test_rmse_norm": test_metrics["rmse_norm"],
        "model_path": str(ckpt_path),
        "metrics_path": metrics_path,
        "future_pred_path": future_path,
        "future_rows": len(future_out),
        "future_locations": int(future_out["location"].nunique()),
        "epochs_ran": 0,
        "train_only_sec": 0.0,
        "eval_sec": float(eval_sec),
        "forecast_sec": float(forecast_sec),
        "run_sec": time.time() - start_all,
    }
    return summary, hist_df, future_out
