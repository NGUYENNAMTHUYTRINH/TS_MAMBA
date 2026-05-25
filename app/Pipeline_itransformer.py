"""
pipeline_itransformer.py
------------------------
Streamlit/YAML wrapper for the iTransformer project.

The original iTransformer code has its own CLI, checkpoint layout, and
one-location CSV requirement. This wrapper converts the shared AQI dataset into
that format, runs the iTransformer script, then exposes the same output contract
as the Mamba/LSTM pipelines.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

APP_ROOT = Path(__file__).resolve().parent.parent
ITRANSFORMER_ROOT = APP_ROOT / "itransformer"
for path in [APP_ROOT, ITRANSFORMER_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from core.metrics import compute_metrics
from Utils import get_timestamp_col, normalize_locations, write_json


AIR_QUALITY_FEATURE_COLS = [
    "pm25", "pm10", "no2", "o3", "so2", "co",
    "aod", "dust", "uv_index", "co2", "aqi",
]


def _select_single_location(df: pd.DataFrame, selected_locations: list[str]) -> str:
    if "location_key" not in df.columns:
        raise ValueError("Dataset can co cot location_key.")
    selected = normalize_locations(selected_locations)
    if not selected:
        locations = sorted(df["location_key"].dropna().astype(str).unique().tolist())
        if not locations:
            raise ValueError("Khong tim thay location_key hop le.")
        return locations[0]
    if len(selected) != 1:
        raise ValueError("iTransformer hien ho tro 1 location moi lan chay. Hay chon 1 dia diem.")
    return selected[0]


def _prepare_itransformer_csv(
    df: pd.DataFrame,
    selected_locations: list[str],
    target_col: str,
    run_dir: Path,
) -> tuple[Path, str, int]:
    location = _select_single_location(df, selected_locations)
    ts_col = get_timestamp_col(df)
    work = df.copy()
    work = work.loc[work["location_key"].astype(str) == str(location)].copy()
    if work.empty:
        raise ValueError(f"Khong co du lieu cho location: {location}")

    if target_col not in work.columns:
        raise ValueError(f"Khong tim thay target column: {target_col}")

    missing = [c for c in AIR_QUALITY_FEATURE_COLS if c not in work.columns]
    if missing:
        raise ValueError(
            "iTransformer can du 11 cot feature AQI mac dinh: "
            + ", ".join(AIR_QUALITY_FEATURE_COLS)
            + f". Dang thieu: {', '.join(missing)}"
        )

    out = work[["location_key", ts_col] + AIR_QUALITY_FEATURE_COLS].copy()
    out = out.rename(columns={ts_col: "ts_utc"})
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts_utc", "location_key", target_col]).sort_values("ts_utc")
    for col in AIR_QUALITY_FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        fill_val = out[col].median()
        out[col] = out[col].fillna(0.0 if pd.isna(fill_val) else fill_val)

    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "air_quality.csv"
    out.to_csv(csv_path, index=False)
    return csv_path, location, len(out)


def _setting_name(config: dict[str, Any]) -> str:
    return "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
        config["model_id"],
        config["model"],
        config["data"],
        config["features"],
        config["seq_len"],
        config["label_len"],
        config["pred_len"],
        config["d_model"],
        config["n_heads"],
        config["e_layers"],
        config["d_layers"],
        config["d_ff"],
        config["factor"],
        config["embed"],
        config["distil"],
        config["des"],
        0,
    )


def _build_cmd(
    config: dict[str, Any],
    is_training: int,
    root_path: Path,
    checkpoints: Path,
    skip_test: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "run.py",
        "--is_training", str(is_training),
        "--model_id", config["model_id"],
        "--model", config["model"],
        "--data", config["data"],
        "--root_path", str(root_path),
        "--data_path", config["data_path"],
        "--features", config["features"],
        "--target", config["target"],
        "--freq", config["freq"],
        "--checkpoints", str(checkpoints),
        "--seq_len", str(config["seq_len"]),
        "--label_len", str(config["label_len"]),
        "--pred_len", str(config["pred_len"]),
        "--enc_in", str(config["enc_in"]),
        "--dec_in", str(config["dec_in"]),
        "--c_out", str(config["c_out"]),
        "--d_model", str(config["d_model"]),
        "--n_heads", str(config["n_heads"]),
        "--e_layers", str(config["e_layers"]),
        "--d_layers", str(config["d_layers"]),
        "--d_ff", str(config["d_ff"]),
        "--factor", str(config["factor"]),
        "--dropout", str(config["dropout"]),
        "--embed", config["embed"],
        "--activation", config["activation"],
        "--num_workers", str(config["num_workers"]),
        "--train_epochs", str(config["train_epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--patience", str(config["patience"]),
        "--learning_rate", str(config["learning_rate"]),
        "--loss", config["loss"],
        "--des", config["des"],
        "--gpu", str(config["gpu"]),
    ]
    if bool(config.get("use_gpu", True)):
        cmd.extend(["--use_gpu", "True"])
    else:
        cmd.extend(["--use_gpu", "False"])
    if bool(config.get("use_amp", False)):
        cmd.append("--use_amp")
    if bool(config.get("inverse", True)):
        cmd.append("--inverse")
    if skip_test:
        cmd.append("--skip_test")
    return cmd


def _run_itransformer(cmd: list[str]) -> None:
    if not (ITRANSFORMER_ROOT / "run.py").exists():
        raise FileNotFoundError(f"Khong tim thay {ITRANSFORMER_ROOT / 'run.py'}")
    subprocess.run(cmd, cwd=ITRANSFORMER_ROOT, check=True)


def _cleanup_legacy_outputs(setting: str) -> None:
    for path in [
        ITRANSFORMER_ROOT / "results" / setting,
        ITRANSFORMER_ROOT / "test_results" / setting,
        ITRANSFORMER_ROOT / "checkpoints" / setting,
    ]:
        if path.exists():
            shutil.rmtree(path)

    legacy_log = ITRANSFORMER_ROOT / "result_long_term_forecast.txt"
    if legacy_log.exists():
        legacy_log.unlink()


def _empty_metrics() -> dict[str, float]:
    return {"mae": np.nan, "mse": np.nan, "rmse": np.nan, "r2": np.nan}


def _str_to_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Gia tri boolean khong hop le: {value}")


def _best_val_metrics_from_history(hist_df: pd.DataFrame) -> dict[str, float]:
    if hist_df is None or hist_df.empty or "val_loss" not in hist_df.columns:
        return {}
    best_idx = pd.to_numeric(hist_df["val_loss"], errors="coerce").idxmin()
    row = hist_df.loc[best_idx]
    return {
        "best_epoch": int(row.get("epoch", 0)),
        "loss": float(row.get("val_loss", np.nan)),
        "mae": float(row.get("mae", np.nan)),
        "rmse": float(row.get("rmse", np.nan)),
        "mae_norm": float(row.get("val_mae_norm", row.get("mae", np.nan))),
        "rmse_norm": float(row.get("val_rmse_norm", row.get("rmse", np.nan))),
        "r2": float(row.get("val_r2", np.nan)),
    }


def _args_from_config(config: dict[str, Any], root_path: Path, use_gpu: bool) -> SimpleNamespace:
    args = dict(config)
    if args.get("data") == "air_quality":
        air_quality_dim = len(AIR_QUALITY_FEATURE_COLS)
        features = args.get("features", "MS")
        if features == "S":
            args["enc_in"] = 1
            args["dec_in"] = 1
            args["c_out"] = 1
        elif features == "MS":
            args["enc_in"] = air_quality_dim
            args["dec_in"] = 1
            args["c_out"] = 1
        elif features == "M":
            args["enc_in"] = air_quality_dim
            args["dec_in"] = air_quality_dim
            args["c_out"] = air_quality_dim
    args.update({
        "root_path": str(root_path),
        "data_path": "air_quality.csv",
        "checkpoints": str(root_path / "checkpoints"),
        "use_gpu": bool(use_gpu and torch.cuda.is_available()),
        "use_multi_gpu": False,
        "devices": "0",
        "device_ids": [0],
        "output_attention": False,
        "do_predict": False,
        "skip_test": False,
        "moving_avg": 25,
        "exp_name": "MTSF",
        "channel_independence": False,
        "class_strategy": "projection",
        "use_norm": 1,
        "target_root_path": str(root_path),
        "target_data_path": "air_quality.csv",
        "efficient_training": False,
        "partial_start_index": 0,
        "lradj": "type1",
        "itr": 1,
    })
    if not args["use_gpu"]:
        args["gpu"] = 0
    return SimpleNamespace(**args)


def _direct_predict_itransformer(
    *,
    df: pd.DataFrame,
    csv_path: Path,
    config: dict[str, Any],
    state: dict,
    target_col: str,
    location: str,
    use_gpu: bool,
) -> tuple[dict[str, float], pd.DataFrame, int, float, float]:
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast

    args = _args_from_config(config, csv_path.parent, use_gpu)
    exp = Exp_Long_Term_Forecast(args)
    exp.model.load_state_dict(state)
    exp.model.eval()

    test_data, test_loader = exp._get_data(flag="test")
    preds, trues = [], []
    preds_norm, trues_norm = [], []
    eval_start = time.time()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(exp.device)
            outputs_np = outputs.detach().cpu().numpy()
            batch_y_np = batch_y.detach().cpu().numpy()
            preds_norm.append(outputs_np.copy())
            trues_norm.append(batch_y_np.copy())
            if test_data.scale and args.inverse:
                shape = outputs_np.shape
                outputs_np = test_data.inverse_transform(outputs_np.squeeze(0)).reshape(shape)
                batch_y_np = test_data.inverse_transform(batch_y_np.squeeze(0)).reshape(shape)
            preds.append(outputs_np)
            trues.append(batch_y_np)

    eval_sec = time.time() - eval_start
    metrics = _empty_metrics()
    if preds and trues:
        metrics = compute_metrics(
            np.array(trues).reshape(-1, args.pred_len, np.array(trues).shape[-1]),
            np.array(preds).reshape(-1, args.pred_len, np.array(preds).shape[-1]),
        )
    if preds_norm and trues_norm:
        norm_metrics = compute_metrics(
            np.array(trues_norm).reshape(-1, args.pred_len, np.array(trues_norm).shape[-1]),
            np.array(preds_norm).reshape(-1, args.pred_len, np.array(preds_norm).shape[-1]),
        )
        metrics["mae_norm"] = norm_metrics["mae"]
        metrics["rmse_norm"] = norm_metrics["rmse"]

    forecast_start = time.time()
    (
        future_x,
        future_label_y,
        future_x_mark,
        future_y_mark,
        future_dates,
    ) = test_data.get_future_forecast_sample()

    future_x = torch.tensor(future_x).float().unsqueeze(0).to(exp.device)
    future_label_y = torch.tensor(future_label_y).float().unsqueeze(0).to(exp.device)
    future_x_mark = torch.tensor(future_x_mark).float().unsqueeze(0).to(exp.device)
    future_y_mark = torch.tensor(future_y_mark).float().unsqueeze(0).to(exp.device)
    future_dec_inp = torch.zeros(
        (1, args.pred_len, future_label_y.shape[-1]),
        dtype=future_label_y.dtype,
        device=exp.device,
    )
    future_dec_inp = torch.cat([future_label_y, future_dec_inp], dim=1)
    with torch.no_grad():
        future_outputs = exp.model(future_x, future_x_mark, future_dec_inp, future_y_mark)
    f_dim = -1 if args.features == "MS" else 0
    future_preds = future_outputs[:, -args.pred_len:, f_dim:].detach().cpu().numpy()
    if test_data.scale and args.inverse:
        shape = future_preds.shape
        future_preds = test_data.inverse_transform(future_preds.squeeze(0)).reshape(shape)
    future_preds = future_preds[0]
    pred_values = future_preds[:, 0] if future_preds.ndim == 2 else future_preds.reshape(-1)
    future_out = pd.DataFrame({
        "time": pd.to_datetime(future_dates, utc=True).strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "location": str(location),
        "predicted": pred_values.astype(float),
    })
    forecast_sec = time.time() - forecast_start
    return metrics, future_out, len(test_data), eval_sec, forecast_sec


def _history_from_log(run_dir: Path) -> pd.DataFrame:
    for name in ["metrics_history.csv", "training_history.csv"]:
        history_path = run_dir / name
        if history_path.exists():
            return pd.read_csv(history_path)
    return pd.DataFrame()


def _save_bundle(
    run_dir: Path,
    checkpoint_path: Path,
    config: dict[str, Any],
    setting: str,
    location: str,
) -> Path:
    model_path = run_dir / "best_itransformer.pt"
    state = torch.load(checkpoint_path, map_location="cpu")
    torch.save(
        {
            "model_state": state,
            "config": config,
            "setting": setting,
            "location": location,
        },
        model_path,
    )
    return model_path


def _summary(
    *,
    mode: str,
    device: str,
    rows_used: int,
    metrics: dict[str, float],
    model_path: Path,
    metrics_path: Path,
    future_path: Path | None,
    future_out: pd.DataFrame | None,
    run_sec: float,
    train_sec: float,
    eval_sec: float,
    forecast_sec: float,
    encoded_features: list[str],
    val_metrics: dict[str, float] | None = None,
) -> dict:
    val_metrics = val_metrics or {}
    return {
        "mode": mode,
        "device": device,
        "n_rows_used": rows_used,
        "split_train": int(rows_used * 0.7),
        "split_val": int(rows_used * 0.1),
        "split_test": rows_used - int(rows_used * 0.8),
        "feature_count_after_encode": len(encoded_features),
        "sample_stride": 1,
        "encoded_features": encoded_features,
        "val_loss": val_metrics.get("loss", metrics.get("best_val_loss", metrics.get("mse"))),
        "val_r2": val_metrics.get("r2", metrics.get("r2")),
        "val_mae": val_metrics.get("mae", metrics.get("mae")),
        "val_rmse": val_metrics.get("rmse", metrics.get("rmse")),
        "val_mae_norm": val_metrics.get("mae_norm", val_metrics.get("mae", metrics.get("mae"))),
        "val_rmse_norm": val_metrics.get("rmse_norm", val_metrics.get("rmse", metrics.get("rmse"))),
        "test_loss": metrics.get("loss", metrics.get("mse")),
        "test_mae": metrics.get("mae"),
        "test_rmse": metrics.get("rmse"),
        "test_r2": metrics.get("r2"),
        "test_mae_norm": metrics.get("mae_norm"),
        "test_rmse_norm": metrics.get("rmse_norm"),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "future_pred_path": str(future_path) if future_path is not None else "",
        "future_rows": len(future_out) if future_out is not None else 0,
        "future_locations": int(future_out["location"].nunique()) if future_out is not None else 0,
        "epochs_ran": 0,
        "train_only_sec": float(train_sec),
        "eval_sec": float(eval_sec),
        "forecast_sec": float(forecast_sec),
        "run_sec": float(run_sec),
    }


def _config(
    *,
    target_col: str,
    window_size: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_layers: int,
    loss_name: str,
    use_gpu: bool,
    num_workers: int,
    early_stop_patience: int,
    label_len: int,
    n_heads: int,
    d_layers: int,
    d_ff: int,
    factor: int,
    dropout: float,
    embed: str,
    activation: str,
    des: str,
    gpu: int,
    features: str,
    model_id: str,
    model: str,
    freq: str,
    enc_in: int,
    dec_in: int,
    c_out: int,
    use_amp: bool,
    inverse: bool,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "model": model,
        "data": "air_quality",
        "data_path": "air_quality.csv",
        "features": features,
        "target": target_col,
        "freq": freq,
        "seq_len": int(window_size),
        "label_len": int(label_len),
        "pred_len": int(horizon),
        "enc_in": int(enc_in),
        "dec_in": int(dec_in),
        "c_out": int(c_out),
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "e_layers": int(n_layers),
        "d_layers": int(d_layers),
        "d_ff": int(d_ff),
        "factor": int(factor),
        "distil": True,
        "dropout": float(dropout),
        "embed": embed,
        "activation": activation,
        "num_workers": int(num_workers),
        "train_epochs": int(epochs),
        "batch_size": int(batch_size),
        "patience": int(early_stop_patience),
        "learning_rate": float(lr),
        "loss": "Huber" if loss_name == "huber" else "MSE",
        "des": des,
        "gpu": int(gpu),
        "use_gpu": bool(use_gpu),
        "use_amp": bool(use_amp),
        "inverse": bool(inverse),
    }


def train_itransformer_pipeline(
    df: pd.DataFrame,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    window_size: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_layers: int,
    loss_name: str,
    seed: int,
    use_gpu: bool,
    early_stop_patience: int,
    num_workers: int,
    label_len: int,
    n_heads: int,
    d_layers: int,
    d_ff: int,
    factor: int,
    dropout: float,
    embed: str,
    activation: str,
    des: str,
    gpu: int,
    features: str,
    model_id: str,
    model: str,
    freq: str,
    enc_in: int,
    dec_in: int,
    c_out: int,
    use_amp: bool,
    inverse: bool,
    run_dir: str | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    del feature_cols, seed
    start = time.time()
    out_dir = Path(run_dir or APP_ROOT / "runs" / "itransformer" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path, location, rows_used = _prepare_itransformer_csv(df, selected_locations, target_col, out_dir)
    config = _config(
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        d_model=d_model,
        n_layers=n_layers,
        loss_name=loss_name,
        use_gpu=use_gpu,
        num_workers=num_workers,
        early_stop_patience=early_stop_patience,
        label_len=label_len,
        n_heads=n_heads,
        d_layers=d_layers,
        d_ff=d_ff,
        factor=factor,
        dropout=dropout,
        embed=embed,
        activation=activation,
        des=des,
        gpu=gpu,
        features=features,
        model_id=model_id,
        model=model,
        freq=freq,
        enc_in=enc_in,
        dec_in=dec_in,
        c_out=c_out,
        use_amp=use_amp,
        inverse=inverse,
    )
    setting = _setting_name(config)
    checkpoints = out_dir / "checkpoints"

    train_start = time.time()
    _run_itransformer(_build_cmd(config, 1, csv_path.parent, checkpoints, skip_test=True))
    train_sec = time.time() - train_start

    checkpoint_path = checkpoints / setting / "checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Khong tim thay checkpoint iTransformer: {checkpoint_path}")
    model_path = _save_bundle(out_dir, checkpoint_path, config, setting, location)
    _cleanup_legacy_outputs(setting)
    shutil.rmtree(checkpoints, ignore_errors=True)
    shutil.rmtree(out_dir / "data", ignore_errors=True)

    hist_df = _history_from_log(out_dir)
    val_metrics = _best_val_metrics_from_history(hist_df)
    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, {
        "best_epoch": val_metrics.get("best_epoch"),
        "val_loss": val_metrics.get("loss"),
        "val_mae": val_metrics.get("mae"),
        "val_rmse": val_metrics.get("rmse"),
        "val_mae_norm": val_metrics.get("mae_norm"),
        "val_rmse_norm": val_metrics.get("rmse_norm"),
        "val_r2": val_metrics.get("r2"),
    })

    summary = _summary(
        mode="train_itransformer_only",
        device="cuda" if use_gpu and torch.cuda.is_available() else "cpu",
        rows_used=rows_used,
        metrics=val_metrics,
        val_metrics=val_metrics,
        model_path=model_path,
        metrics_path=metrics_path,
        future_path=None,
        future_out=None,
        run_sec=time.time() - start,
        train_sec=train_sec,
        eval_sec=0.0,
        forecast_sec=0.0,
        encoded_features=AIR_QUALITY_FEATURE_COLS,
    )
    return summary, hist_df, None


def predict_itransformer_with_saved_model(
    df: pd.DataFrame,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    checkpoint_path: str,
    batch_size: int,
    use_gpu: bool,
    run_dir: str | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    del feature_cols, batch_size
    start = time.time()
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Khong tim thay checkpoint iTransformer: {ckpt_path}")

    bundle = torch.load(ckpt_path, map_location="cpu")
    if "config" not in bundle or "model_state" not in bundle:
        raise ValueError("Checkpoint iTransformer phai la best_itransformer.pt tao boi pipeline nay.")
    config = dict(bundle["config"])
    config["target"] = target_col
    config["use_gpu"] = bool(use_gpu)
    setting = str(bundle["setting"])
    location = _select_single_location(df, selected_locations or [bundle.get("location")])

    out_dir = Path(run_dir or ckpt_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path, location, rows_used = _prepare_itransformer_csv(df, [location], target_col, out_dir)
    metrics, future_out, rows_used, eval_sec, forecast_sec = _direct_predict_itransformer(
        df=df,
        csv_path=csv_path,
        config=config,
        state=bundle["model_state"],
        target_col=target_col,
        location=location,
        use_gpu=use_gpu,
    )
    _cleanup_legacy_outputs(setting)
    shutil.rmtree(out_dir / "data", ignore_errors=True)

    future_path = out_dir / "future_itransformer_predictions.csv"
    metrics_path = out_dir / "loaded_itransformer_metrics.json"
    future_out.to_csv(future_path, index=False)
    write_json(metrics_path, metrics)
    hist_df = _history_from_log(ckpt_path.parent)

    summary = _summary(
        mode="loaded_itransformer",
        device="cuda" if use_gpu and torch.cuda.is_available() else "cpu",
        rows_used=rows_used,
        metrics=metrics,
        val_metrics=_best_val_metrics_from_history(hist_df),
        model_path=ckpt_path,
        metrics_path=metrics_path,
        future_path=future_path,
        future_out=future_out,
        run_sec=time.time() - start,
        train_sec=0.0,
        eval_sec=eval_sec,
        forecast_sec=forecast_sec,
        encoded_features=AIR_QUALITY_FEATURE_COLS,
    )
    return summary, hist_df, future_out


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train iTransformer AQI with shared project layout.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--locations", default="")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--loss", required=True, choices=["huber", "mse"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", required=True, choices=["cuda", "cpu"])
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--n-layers", type=int, required=True, help="Encoder layers/e_layers.")
    parser.add_argument("--label-len", type=int, required=True)
    parser.add_argument("--n-heads", type=int, required=True)
    parser.add_argument("--d-layers", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--embed", required=True)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--des", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--features", required=True, choices=["M", "S", "MS"])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--freq", required=True)
    parser.add_argument("--enc-in", type=int, required=True)
    parser.add_argument("--dec-in", type=int, required=True)
    parser.add_argument("--c-out", type=int, required=True)
    parser.add_argument("--use-amp", type=_str_to_bool, required=True)
    parser.add_argument("--inverse", type=_str_to_bool, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = APP_ROOT / data_path
    df = pd.read_csv(data_path)
    locations = normalize_locations(args.locations)

    summary, _, _ = train_itransformer_pipeline(
        df=df,
        selected_locations=locations,
        target_col=args.target_col,
        feature_cols=AIR_QUALITY_FEATURE_COLS,
        window_size=args.window_size,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        loss_name=args.loss,
        seed=args.seed,
        use_gpu=args.device == "cuda",
        early_stop_patience=args.patience,
        num_workers=args.num_workers,
        label_len=args.label_len,
        n_heads=args.n_heads,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        factor=args.factor,
        dropout=args.dropout,
        embed=args.embed,
        activation=args.activation,
        des=args.des,
        gpu=args.gpu,
        features=args.features,
        model_id=args.model_id,
        model=args.model,
        freq=args.freq,
        enc_in=args.enc_in,
        dec_in=args.dec_in,
        c_out=args.c_out,
        use_amp=args.use_amp,
        inverse=args.inverse,
        run_dir=args.out_dir,
    )
    print("iTransformer complete.")
    print(f"Model  : {summary['model_path']}")
    print(f"Metrics: {summary['metrics_path']}")
    print(f"History: {Path(summary['metrics_path']).with_name('metrics_history.csv')}")
    print("Skip prediction. Streamlit can load this checkpoint later for inference.")


if __name__ == "__main__":
    _main()
