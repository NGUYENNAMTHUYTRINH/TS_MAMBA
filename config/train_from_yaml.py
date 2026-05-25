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


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, dict):
        raise KeyError(f"Thieu section '{name}' trong file YAML.")
    return value


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise KeyError(f"Thieu key '{section_name}.{key}' trong file YAML.")
    return section[key]


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_mamba_cmd(cfg: dict[str, Any], run_name: str) -> list[str]:
    run_cfg = _section(cfg, "run")
    data = _section(cfg, "data")
    mamba = _section(cfg, "mamba")
    output_root = _required(run_cfg, "output_root", "run")
    out_dir = PROJECT_ROOT / output_root / "mamba" / run_name
    locations = _as_list(_required(data, "locations", "data"))

    cmd = [
        sys.executable,
        "mamba/train_mamba_aqi.py",
        "--data-path", str(_required(data, "data_path", "data")),
        "--target-col", str(_required(data, "target_col", "data")),
        "--epochs", str(_required(mamba, "epochs", "mamba")),
        "--window-size", str(_required(mamba, "window_size", "mamba")),
        "--horizon", str(_required(mamba, "horizon", "mamba")),
        "--sample-stride", str(_required(mamba, "sample_stride", "mamba")),
        "--batch-size", str(_required(mamba, "batch_size", "mamba")),
        "--lr", str(_required(mamba, "lr", "mamba")),
        "--loss", str(_required(mamba, "loss", "mamba")),
        "--seed", str(_required(mamba, "seed", "mamba")),
        "--num-workers", str(_required(mamba, "num_workers", "mamba")),
        "--device", str(_required(mamba, "device", "mamba")),
        "--weight-decay", str(_required(mamba, "weight_decay", "mamba")),
        "--d-model", str(_required(mamba, "d_model", "mamba")),
        "--n-layers", str(_required(mamba, "n_layers", "mamba")),
        "--grad-accum-steps", str(_required(mamba, "grad_accum_steps", "mamba")),
        "--max-grad-norm", str(_required(mamba, "max_grad_norm", "mamba")),
        "--patience", str(_required(mamba, "patience", "mamba")),
        "--min-delta", str(_required(mamba, "min_delta", "mamba")),
        "--out-dir", str(out_dir),
    ]
    if locations:
        cmd.extend(["--locations", ",".join(locations)])
    if _bool(_required(mamba, "amp", "mamba")):
        cmd.append("--amp")
    if _bool(_required(mamba, "forecast_24h", "mamba")):
        cmd.append("--forecast-24h")
        _add_optional(cmd, "--forecast-base", _required(mamba, "forecast_base", "mamba"))
        _add_optional(cmd, "--forecast-out", _required(mamba, "forecast_out", "mamba"))
    return cmd


def build_lstm_cmd(cfg: dict[str, Any], run_name: str) -> list[str]:
    run_cfg = _section(cfg, "run")
    data = _section(cfg, "data")
    lstm = _section(cfg, "lstm")
    output_root = _required(run_cfg, "output_root", "run")
    locations = _as_list(_required(data, "locations", "data"))

    cmd = [
        sys.executable,
        "LSTM-Time-Series-Forecasting/run.py",
        "--input", str(_required(data, "data_path", "data")),
        "--outdir", str(Path(output_root) / "lstm"),
        "--run_name", run_name,
        "--value_column", str(_required(data, "target_col", "data")),
        "--epochs", str(_required(lstm, "epochs", "lstm")),
        "--batch-size", str(_required(lstm, "batch_size", "lstm")),
        "--lookback", str(_required(lstm, "window_size", "lstm")),
        "--horizon", str(_required(lstm, "horizon", "lstm")),
        "--hidden_size", str(_required(lstm, "d_model", "lstm")),
        "--num_layers", str(_required(lstm, "n_layers", "lstm")),
        "--embed_dim", str(_required(lstm, "embed_dim", "lstm")),
        "--lr", str(_required(lstm, "lr", "lstm")),
        "--loss", str(_required(lstm, "loss", "lstm")),
        "--seed", str(_required(lstm, "seed", "lstm")),
        "--num_workers", str(_required(lstm, "num_workers", "lstm")),
        "--early_stop_patience", str(_required(lstm, "early_stop_patience", "lstm")),
        "--overfit_threshold", str(_required(lstm, "overfit_threshold", "lstm")),
    ]
    if len(locations) == 1:
        cmd.extend(["--location", locations[0]])
    elif locations:
        cmd.append("--locations")
        cmd.extend(locations)
    if _bool(_required(lstm, "predict", "lstm")):
        cmd.append("--predict")
        _add_optional(cmd, "--output_csv", _required(lstm, "output_csv", "lstm"))
    return cmd


def build_itransformer_cmd(cfg: dict[str, Any], run_name: str) -> list[str]:
    run_cfg = _section(cfg, "run")
    data = _section(cfg, "data")
    itransformer = _section(cfg, "itransformer")
    output_root = _required(run_cfg, "output_root", "run")
    out_dir = PROJECT_ROOT / output_root / "itransformer" / run_name
    locations = _as_list(_required(data, "locations", "data"))

    cmd = [
        sys.executable,
        "app/Pipeline_itransformer.py",
        "--data-path", str(_required(data, "data_path", "data")),
        "--target-col", str(_required(data, "target_col", "data")),
        "--epochs", str(_required(itransformer, "epochs", "itransformer")),
        "--window-size", str(_required(itransformer, "window_size", "itransformer")),
        "--horizon", str(_required(itransformer, "horizon", "itransformer")),
        "--batch-size", str(_required(itransformer, "batch_size", "itransformer")),
        "--lr", str(_required(itransformer, "lr", "itransformer")),
        "--loss", str(_required(itransformer, "loss", "itransformer")),
        "--seed", str(_required(itransformer, "seed", "itransformer")),
        "--num-workers", str(_required(itransformer, "num_workers", "itransformer")),
        "--device", str(_required(itransformer, "device", "itransformer")),
        "--model-id", str(_required(itransformer, "model_id", "itransformer")),
        "--model", str(_required(itransformer, "model", "itransformer")),
        "--features", str(_required(itransformer, "features", "itransformer")),
        "--freq", str(_required(itransformer, "freq", "itransformer")),
        "--label-len", str(_required(itransformer, "label_len", "itransformer")),
        "--n-heads", str(_required(itransformer, "n_heads", "itransformer")),
        "--d-model", str(_required(itransformer, "d_model", "itransformer")),
        "--n-layers", str(_required(itransformer, "n_layers", "itransformer")),
        "--d-layers", str(_required(itransformer, "d_layers", "itransformer")),
        "--d-ff", str(_required(itransformer, "d_ff", "itransformer")),
        "--factor", str(_required(itransformer, "factor", "itransformer")),
        "--dropout", str(_required(itransformer, "dropout", "itransformer")),
        "--embed", str(_required(itransformer, "embed", "itransformer")),
        "--activation", str(_required(itransformer, "activation", "itransformer")),
        "--des", str(_required(itransformer, "des", "itransformer")),
        "--gpu", str(_required(itransformer, "gpu", "itransformer")),
        "--enc-in", str(_required(itransformer, "enc_in", "itransformer")),
        "--dec-in", str(_required(itransformer, "dec_in", "itransformer")),
        "--c-out", str(_required(itransformer, "c_out", "itransformer")),
        "--patience", str(_required(itransformer, "patience", "itransformer")),
        "--use-amp", str(_required(itransformer, "use_amp", "itransformer")),
        "--inverse", str(_required(itransformer, "inverse", "itransformer")),
        "--out-dir", str(out_dir),
    ]
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

    run_cfg = _section(cfg, "run")
    run_name = _required(run_cfg, "name", "run") or datetime.now().strftime("%Y%m%d_%H%M%S")
    models = [m.lower() for m in _as_list(_required(run_cfg, "models", "run"))]
    output_root = _required(run_cfg, "output_root", "run")

    print(f"Config : {config_path}")
    print(f"Run id : {run_name}")
    print(f"Models : {', '.join(models)}")

    if "mamba" in models and _bool(_required(_section(cfg, "mamba"), "enabled", "mamba")):
        _run(build_mamba_cmd(cfg, run_name))
    if "lstm" in models and _bool(_required(_section(cfg, "lstm"), "enabled", "lstm")):
        _run(build_lstm_cmd(cfg, run_name))
    if "itransformer" in models and _bool(_required(_section(cfg, "itransformer"), "enabled", "itransformer")):
        _run(build_itransformer_cmd(cfg, run_name))

    print("\nDone.")
    print(f"Mamba: {PROJECT_ROOT / output_root / 'mamba' / run_name}")
    print(f"LSTM : {PROJECT_ROOT / output_root / 'lstm' / run_name}")
    print(f"iTransformer: {PROJECT_ROOT / output_root / 'itransformer' / run_name}")


if __name__ == "__main__":
    main()
