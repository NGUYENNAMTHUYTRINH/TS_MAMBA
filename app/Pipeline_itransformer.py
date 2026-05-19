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
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

APP_ROOT = Path(__file__).resolve().parent.parent
ITRANSFORMER_ROOT = APP_ROOT / "itransformer"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

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


def _results_dir(setting: str) -> Path:
    return ITRANSFORMER_ROOT / "results" / setting


def _copy_result_files(setting: str, run_dir: Path) -> None:
    src = _results_dir(setting)
    if not src.exists():
        return
    dst = run_dir / "itransformer_results"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _empty_metrics() -> dict[str, float]:
    return {"mae": np.nan, "mse": np.nan, "rmse": np.nan, "r2": np.nan}


def _metrics_from_history(hist_df: pd.DataFrame) -> dict[str, float]:
    if hist_df is None or hist_df.empty or "val_loss" not in hist_df.columns:
        return {}
    best_idx = pd.to_numeric(hist_df["val_loss"], errors="coerce").idxmin()
    row = hist_df.loc[best_idx]
    val_loss = float(row.get("val_loss", np.nan))
    return {
        "mode": "train_loss_only",
        "best_epoch": int(row.get("epoch", 0)),
        "best_val_loss": val_loss,
        "mse": val_loss,
        "mae": float(row.get("mae", np.nan)),
        "rmse": float(row.get("rmse", np.nan)),
        "r2": float(row.get("val_r2", np.nan)),
    }


def _compute_metrics_from_results(setting: str) -> dict[str, float]:
    result_dir = _results_dir(setting)
    pred_path = result_dir / "pred.npy"
    true_path = result_dir / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        return _empty_metrics()
    preds = np.load(pred_path)
    trues = np.load(true_path)
    return compute_metrics(trues, preds)


def _load_future(setting: str, target_col: str, location: str) -> pd.DataFrame:
    future_path = _results_dir(setting) / "future_forecast.csv"
    if not future_path.exists():
        raise RuntimeError("Khong tim thay future_forecast.csv cua iTransformer.")
    future = pd.read_csv(future_path)
    pred_col = f"forecast_{target_col}"
    if pred_col not in future.columns:
        pred_cols = [c for c in future.columns if c.startswith("forecast_")]
        if not pred_cols:
            raise RuntimeError("future_forecast.csv khong co cot forecast_*.")
        pred_col = pred_cols[0]
    return (
        future.assign(
            time=lambda d: pd.to_datetime(d["forecast_time"], utc=True, errors="coerce")
            .dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            location=str(location),
            predicted=lambda d: pd.to_numeric(d[pred_col], errors="coerce"),
        )[["time", "location", "predicted"]]
        .dropna(subset=["time", "predicted"])
        .reset_index(drop=True)
    )


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
    write_json(
        run_dir / "itransformer_config.json",
        {
            "config": config,
            "setting": setting,
            "location": location,
            "checkpoint_path": str(checkpoint_path),
        },
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
) -> dict:
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
        "val_loss": metrics.get("best_val_loss", metrics.get("mse")),
        "val_r2": metrics.get("r2"),
        "val_mae": metrics.get("mae"),
        "val_rmse": metrics.get("rmse"),
        "val_mae_norm": metrics.get("mae"),
        "val_rmse_norm": metrics.get("rmse"),
        "test_loss": metrics.get("mse"),
        "test_mae": metrics.get("mae"),
        "test_rmse": metrics.get("rmse"),
        "test_r2": metrics.get("r2"),
        "test_mae_norm": np.nan,
        "test_rmse_norm": np.nan,
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
    label_len: int | None = None,
    n_heads: int = 8,
    d_layers: int = 1,
    d_ff: int | None = None,
    factor: int = 1,
    dropout: float = 0.1,
    embed: str = "timeF",
    activation: str = "gelu",
    des: str = "Exp",
    gpu: int = 0,
    features: str = "MS",
    model_id: str = "air_quality_AQI",
    model: str = "Transformer",
    freq: str = "h",
    enc_in: int = 11,
    dec_in: int = 11,
    c_out: int = 1,
    use_amp: bool = False,
    inverse: bool = True,
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
        "label_len": int(label_len or min(48, window_size)),
        "pred_len": int(horizon),
        "enc_in": int(enc_in),
        "dec_in": int(dec_in),
        "c_out": int(c_out),
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "e_layers": int(n_layers),
        "d_layers": int(d_layers),
        "d_ff": int(d_ff or max(d_model, 128)),
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
        "loss": "MSE" if loss_name == "mse" else "MSE",
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
    run_dir: str | None = None,
    label_len: int | None = None,
    n_heads: int = 8,
    d_layers: int = 1,
    d_ff: int | None = None,
    factor: int = 1,
    dropout: float = 0.1,
    embed: str = "timeF",
    activation: str = "gelu",
    des: str = "Exp",
    gpu: int = 0,
    features: str = "MS",
    model_id: str = "air_quality_AQI",
    model: str = "Transformer",
    freq: str = "h",
    enc_in: int = 11,
    dec_in: int = 11,
    c_out: int = 1,
    use_amp: bool = False,
    inverse: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    del feature_cols, seed
    start = time.time()
    out_dir = Path(run_dir or APP_ROOT / "runs" / "itransformer" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_ctx = tempfile.TemporaryDirectory(prefix="itransformer_train_")
    tmp_root = Path(tmp_ctx.name)
    csv_path, location, rows_used = _prepare_itransformer_csv(df, selected_locations, target_col, tmp_root)
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
        num_workers=0,
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
    checkpoints = tmp_root / "checkpoints"

    try:
        train_start = time.time()
        _run_itransformer(_build_cmd(config, 1, csv_path.parent, checkpoints, skip_test=True))
        train_sec = time.time() - train_start

        checkpoint_path = checkpoints / setting / "checkpoint.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Khong tim thay checkpoint iTransformer: {checkpoint_path}")
        model_path = _save_bundle(out_dir, checkpoint_path, config, setting, location)
        tmp_history_path = tmp_root / "metrics_history.csv"
        if tmp_history_path.exists():
            shutil.copy2(tmp_history_path, out_dir / "metrics_history.csv")
    finally:
        tmp_ctx.cleanup()

    hist_df = _history_from_log(out_dir)
    metrics = _metrics_from_history(hist_df)
    future_out = None
    future_path = None
    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, metrics)

    summary = _summary(
        mode="train_itransformer_only",
        device="cuda" if use_gpu and torch.cuda.is_available() else "cpu",
        rows_used=rows_used,
        metrics=metrics,
        model_path=model_path,
        metrics_path=metrics_path,
        future_path=future_path,
        future_out=future_out,
        run_sec=time.time() - start,
        train_sec=train_sec,
        eval_sec=0.0,
        forecast_sec=0.0,
        encoded_features=AIR_QUALITY_FEATURE_COLS,
    )
    return summary, hist_df, future_out


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

    local_checkpoint_dir = ITRANSFORMER_ROOT / "checkpoints" / setting
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle["model_state"], local_checkpoint_dir / "checkpoint.pth")

    with tempfile.TemporaryDirectory(prefix="itransformer_predict_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        csv_path, location, rows_used = _prepare_itransformer_csv(df, [location], target_col, tmp_root)
        eval_start = time.time()
        _run_itransformer(_build_cmd(config, 0, csv_path.parent, ITRANSFORMER_ROOT / "checkpoints"))
        eval_sec = time.time() - eval_start
    _copy_result_files(setting, out_dir)

    metrics = _compute_metrics_from_results(setting)
    forecast_start = time.time()
    future_out = _load_future(setting, target_col, location)
    forecast_sec = time.time() - forecast_start

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
    parser.add_argument("--data-path", default="dataset/air_quality.csv")
    parser.add_argument("--target-col", default="aqi")
    parser.add_argument("--locations", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", default="huber", choices=["huber", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3, help="Encoder layers/e_layers.")
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", default="timeF")
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--des", default="Exp")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--features", default="MS", choices=["M", "S", "MS"])
    parser.add_argument("--model-id", default="air_quality_AQI")
    parser.add_argument("--model", default="Transformer")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--enc-in", type=int, default=11)
    parser.add_argument("--dec-in", type=int, default=11)
    parser.add_argument("--c-out", type=int, default=1)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-inverse", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
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
        run_dir=args.out_dir,
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
        inverse=not args.no_inverse,
    )
    print("iTransformer complete.")
    print(f"Model  : {summary['model_path']}")
    print(f"Metrics: {summary['metrics_path']}")
    print(f"History: {Path(summary['metrics_path']).with_name('metrics_history.csv')}")
    print("Skip prediction. Streamlit can load this checkpoint later for inference.")


if __name__ == "__main__":
    _main()
