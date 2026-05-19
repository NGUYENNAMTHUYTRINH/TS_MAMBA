from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "mamba").is_dir() and (path / "LSTM-Time-Series-Forecasting").is_dir():
            return path
    raise RuntimeError("Khong tim thay project root co thu muc mamba va LSTM-Time-Series-Forecasting.")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(SCRIPT_DIR)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _add_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_mamba_cmd(cfg: dict[str, Any], run_name: str) -> list[str]:
    common = cfg.get("common", {})
    mamba = cfg.get("mamba", {})
    output_root = cfg.get("run", {}).get("output_root", "runs")
    out_dir = PROJECT_ROOT / output_root / "mamba" / run_name
    locations = _as_list(common.get("locations"))

    cmd = [
        sys.executable,
        "mamba/train_mamba_aqi.py",
        "--data-path", str(common.get("data_path", "dataset/air_quality.csv")),
        "--target-col", str(common.get("target_col", "aqi")),
        "--epochs", str(common.get("epochs", 20)),
        "--window-size", str(common.get("window_size", 72)),
        "--horizon", str(common.get("horizon", 12)),
        "--sample-stride", str(mamba.get("sample_stride", 4)),
        "--batch-size", str(common.get("batch_size", 32)),
        "--lr", str(common.get("lr", 0.001)),
        "--loss", str(common.get("loss", "huber")),
        "--seed", str(common.get("seed", 42)),
        "--num-workers", str(common.get("num_workers", 0)),
        "--device", str(mamba.get("device", "cuda")),
        "--weight-decay", str(mamba.get("weight_decay", 0.0001)),
        "--d-model", str(common.get("d_model", 64)),
        "--n-layers", str(common.get("n_layers", 2)),
        "--grad-accum-steps", str(mamba.get("grad_accum_steps", 1)),
        "--max-grad-norm", str(mamba.get("max_grad_norm", 1.0)),
        "--patience", str(mamba.get("patience", 5)),
        "--min-delta", str(mamba.get("min_delta", 0.0)),
        "--out-dir", str(out_dir),
    ]
    if locations:
        cmd.extend(["--locations", ",".join(locations)])
    if _bool(mamba.get("amp", True)):
        cmd.append("--amp")
    if _bool(mamba.get("forecast_24h", False)):
        cmd.append("--forecast-24h")
        _add_optional(cmd, "--forecast-base", mamba.get("forecast_base"))
        _add_optional(cmd, "--forecast-out", mamba.get("forecast_out"))
    return cmd


def build_lstm_cmd(cfg: dict[str, Any], run_name: str) -> list[str]:
    common = cfg.get("common", {})
    lstm = cfg.get("lstm", {})
    output_root = cfg.get("run", {}).get("output_root", "runs")
    locations = _as_list(common.get("locations"))

    cmd = [
        sys.executable,
        "LSTM-Time-Series-Forecasting/run.py",
        "--input", str(common.get("data_path", "dataset/air_quality.csv")),
        "--outdir", str(Path(output_root) / "lstm"),
        "--run_name", run_name,
        "--value_column", str(common.get("target_col", "aqi")),
        "--epochs", str(common.get("epochs", 20)),
        "--batch-size", str(common.get("batch_size", 32)),
        "--lookback", str(common.get("window_size", 72)),
        "--horizon", str(common.get("horizon", 12)),
        "--hidden_size", str(common.get("d_model", 64)),
        "--num_layers", str(common.get("n_layers", 2)),
        "--embed_dim", str(lstm.get("embed_dim", 16)),
        "--lr", str(common.get("lr", 0.001)),
        "--loss", str(common.get("loss", "huber")),
        "--seed", str(common.get("seed", 42)),
        "--num_workers", str(common.get("num_workers", 0)),
        "--early_stop_patience", str(lstm.get("early_stop_patience", 3)),
        "--overfit_threshold", str(lstm.get("overfit_threshold", 0.05)),
    ]
    if len(locations) == 1:
        cmd.extend(["--location", locations[0]])
    elif locations:
        cmd.append("--locations")
        cmd.extend(locations)
    if _bool(lstm.get("predict", False)):
        cmd.append("--predict")
        _add_optional(cmd, "--output_csv", lstm.get("output_csv"))
    return cmd


def build_itransformer_cmd(cfg: dict[str, Any], run_name: str) -> list[str]:
    common = cfg.get("common", {})
    itransformer = cfg.get("itransformer", {})
    output_root = cfg.get("run", {}).get("output_root", "runs")
    out_dir = PROJECT_ROOT / output_root / "itransformer" / run_name
    locations = _as_list(common.get("locations"))

    cmd = [
        sys.executable,
        "app/Pipeline_itransformer.py",
        "--data-path", str(common.get("data_path", "dataset/air_quality.csv")),
        "--target-col", str(common.get("target_col", "aqi")),
        "--epochs", str(common.get("epochs", 20)),
        "--window-size", str(common.get("window_size", 72)),
        "--horizon", str(common.get("horizon", 12)),
        "--batch-size", str(common.get("batch_size", 32)),
        "--lr", str(common.get("lr", 0.001)),
        "--loss", str(common.get("loss", "huber")),
        "--seed", str(common.get("seed", 42)),
        "--device", str(itransformer.get("device", "cuda")),
        "--model-id", str(itransformer.get("model_id", "air_quality_AQI")),
        "--model", str(itransformer.get("model", "Transformer")),
        "--features", str(itransformer.get("features", "MS")),
        "--freq", str(itransformer.get("freq", "h")),
        "--label-len", str(itransformer.get("label_len", 48)),
        "--n-heads", str(itransformer.get("n_heads", 8)),
        "--d-model", str(common.get("d_model", 128)),
        "--n-layers", str(common.get("n_layers", 3)),
        "--d-layers", str(itransformer.get("d_layers", 1)),
        "--d-ff", str(itransformer.get("d_ff", 128)),
        "--factor", str(itransformer.get("factor", 1)),
        "--dropout", str(itransformer.get("dropout", 0.1)),
        "--embed", str(itransformer.get("embed", "timeF")),
        "--activation", str(itransformer.get("activation", "gelu")),
        "--des", str(itransformer.get("des", "Exp")),
        "--gpu", str(itransformer.get("gpu", 0)),
        "--enc-in", str(itransformer.get("enc_in", 11)),
        "--dec-in", str(itransformer.get("dec_in", 11)),
        "--c-out", str(itransformer.get("c_out", 1)),
        "--patience", str(itransformer.get("patience", 3)),
        "--out-dir", str(out_dir),
    ]
    if _bool(itransformer.get("use_amp", False)):
        cmd.append("--use-amp")
    if not _bool(itransformer.get("inverse", True)):
        cmd.append("--no-inverse")
    if locations:
        cmd.extend(["--locations", ",".join(locations)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mamba, LSTM, and iTransformer from one YAML config.")
    parser.add_argument("--config", default="train_models.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        local_config = SCRIPT_DIR / config_path
        config_path = local_config if local_config.exists() else PROJECT_ROOT / config_path
    cfg = _load_config(config_path)

    run_cfg = cfg.get("run", {})
    run_name = run_cfg.get("name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    models = [m.lower() for m in _as_list(run_cfg.get("models", ["mamba", "lstm"]))]

    print(f"Config : {config_path}")
    print(f"Run id : {run_name}")
    print(f"Models : {', '.join(models)}")

    if "mamba" in models and _bool(cfg.get("mamba", {}).get("enabled", True)):
        _run(build_mamba_cmd(cfg, run_name))
    if "lstm" in models and _bool(cfg.get("lstm", {}).get("enabled", True)):
        _run(build_lstm_cmd(cfg, run_name))
    if "itransformer" in models and _bool(cfg.get("itransformer", {}).get("enabled", True)):
        _run(build_itransformer_cmd(cfg, run_name))

    print("\nDone.")
    print(f"Mamba: {PROJECT_ROOT / run_cfg.get('output_root', 'runs') / 'mamba' / run_name}")
    print(f"LSTM : {PROJECT_ROOT / run_cfg.get('output_root', 'runs') / 'lstm' / run_name}")
    print(f"iTransformer: {PROJECT_ROOT / run_cfg.get('output_root', 'runs') / 'itransformer' / run_name}")


if __name__ == "__main__":
    main()
