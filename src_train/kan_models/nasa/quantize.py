"""Apply QuantKAN PTQ to the NASA KAN checkpoint and export a quantized JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kan_models.nasa.train import (
    DEFAULT_CONFIG_PATH,
    build_model,
    evaluate_split,
    extract_export_payload,
    load_config,
    load_dataset,
    predict_batched,
    save_predictions,
)
from kan_models.utils import (
    QUANTKAN_ROOT,
    detect_device,
    load_ptq_api,
    quantkan_commit,
    serialize_width,
    write_json,
)


@dataclass(frozen=True)
class QuantOutputPaths:
    checkpoint: Path
    export_json: Path
    metrics_json: Path
    predictions_csv: Path
    act_scales_json: Path
    act_params_json: Path
    weight_qlog_json: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply QuantKAN uniform PTQ to the NASA KAN checkpoint."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the NASA TOML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint path. Defaults to the fp32 training checkpoint from the config output dir.",
    )
    parser.add_argument("--w-bit", type=int, default=16, help="Weight bit-width.")
    parser.add_argument("--a-bit", type=int, default=16, help="Activation bit-width.")
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.999,
        help="Percentile used to collect activation scales.",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=8,
        help="Maximum calibration batches for activation scale collection.",
    )
    parser.add_argument(
        "--calib-batch-size",
        type=int,
        default=64,
        help="Calibration batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Target device for PTQ evaluation. Default: cpu.",
    )
    return parser.parse_args(argv)


def output_paths(output_dir: Path, export_filename: str, w_bit: int, a_bit: int) -> QuantOutputPaths:
    stem = Path(export_filename).stem
    if stem.endswith("_riscv_export"):
        stem = stem[: -len("_riscv_export")]
    label = f"{stem}_quantkan_uniform_w{w_bit}a{a_bit}_pc"
    return QuantOutputPaths(
        checkpoint=output_dir / f"{label}_checkpoint.pt",
        export_json=output_dir / f"{label}_export.json",
        metrics_json=output_dir / f"{label}_metrics.json",
        predictions_csv=output_dir / f"{label}_test_predictions.csv",
        act_scales_json=output_dir / f"{label}_act_scales.json",
        act_params_json=output_dir / f"{label}_act_params.json",
        weight_qlog_json=output_dir / f"{label}_weight_qlog.json",
    )


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Invalid model_state_dict in checkpoint: {path}")
        return state_dict
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint format: {path}")


def build_calibration_loader(features: torch.Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(features.detach().cpu())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def export_quantization_metadata(
    *,
    w_bit: int,
    a_bit: int,
    percentile: float,
    calib_batches: int,
    calib_batch_size: int,
    qlog: dict[str, Any],
    act_scales: dict[str, float],
) -> dict[str, Any]:
    return {
        "source": "quantkan_ptq_uniform",
        "quantkan_root": str(QUANTKAN_ROOT.relative_to(QUANTKAN_ROOT.parents[1])),
        "quantkan_commit": quantkan_commit(),
        "weight": {
            "bits": int(w_bit),
            "mode": "symmetric",
            "per_channel": True,
            "layer_types": ["KANLayer"],
            "qlog": qlog,
        },
        "activation": {
            "bits": int(a_bit),
            "signed": True,
            "percentile": float(percentile),
            "calib_batches": int(calib_batches),
            "calib_batch_size": int(calib_batch_size),
            "layer_types": ["KANLayer"],
            "scales": {key: float(value) for key, value in act_scales.items()},
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = load_config(args.config)
    dataset = load_dataset(experiment.data)
    device = detect_device(args.device)

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else (experiment.output.output_dir / experiment.output.model_filename).resolve()
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"NASA checkpoint not found: {checkpoint_path}")

    output_dir = experiment.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir, experiment.output.export_json_filename, args.w_bit, args.a_bit)

    model = build_model(
        input_dim=dataset.input_dim,
        model_config=experiment.model,
        seed=experiment.data.random_seed,
        device=device,
    )
    model.load_state_dict(load_checkpoint_state(checkpoint_path))
    model.to(device)
    model.eval()

    X_train = dataset.X_train.to(device)
    y_train = dataset.y_train.to(device)
    X_val = dataset.X_val.to(device)
    y_val = dataset.y_val.to(device)
    X_test = dataset.X_test.to(device)
    y_test = dataset.y_test.to(device)

    fp32_test_metrics = evaluate_split(model, X_test, y_test, experiment)

    ptq = load_ptq_api()
    qlog = ptq["quantize_weights_uniform"](
        model,
        w_bit=args.w_bit,
        per_channel=True,
        mode="symmetric",
        layer_types=("KANLayer",),
    )
    calib_loader = build_calibration_loader(X_train, args.calib_batch_size)
    act_scales = ptq["collect_input_activation_scales"](
        model,
        calib_loader,
        percentile=args.percentile,
        layer_types=("KANLayer",),
        max_batches=args.calib_batches,
        device=str(device),
        estimator="batch_max_percentile",
        abit=args.a_bit,
        signed=True,
    )
    ptq["wrap_input_quant"](
        model,
        scales=act_scales,
        abit=args.a_bit,
        layer_types=("KANLayer",),
        signed=True,
    )

    train_metrics = evaluate_split(model, X_train, y_train, experiment)
    val_metrics = evaluate_split(model, X_val, y_val, experiment)
    test_metrics = evaluate_split(model, X_test, y_test, experiment)
    test_predictions = (
        predict_batched(model, X_test, experiment.training.eval_batch_size)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        * dataset.target_scale
    )

    metrics = {
        "mode": "quantkan_ptq_uniform",
        "config_path": str(experiment.config_path),
        "source_checkpoint": str(checkpoint_path),
        "device": str(device),
        "input_dim": dataset.input_dim,
        "model_width": serialize_width(model.width),
        "target_scale": experiment.data.target_scale,
        "rul_tolerance": experiment.evaluation.rul_tolerance,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "reference_fp32_test": fp32_test_metrics,
        "quantization": {
            "weight_bits": int(args.w_bit),
            "activation_bits": int(args.a_bit),
            "percentile": float(args.percentile),
            "calib_batches": int(args.calib_batches),
            "calib_batch_size": int(args.calib_batch_size),
        },
    }

    ptq["unwrap_input_quant"](model)
    export_payload = extract_export_payload(experiment, dataset, model, metrics)
    export_payload["quantization"] = export_quantization_metadata(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        percentile=args.percentile,
        calib_batches=args.calib_batches,
        calib_batch_size=args.calib_batch_size,
        qlog=qlog,
        act_scales=act_scales,
    )

    write_json(paths.export_json, export_payload)
    write_json(paths.metrics_json, metrics)
    write_json(paths.weight_qlog_json, qlog)
    write_json(paths.act_scales_json, {key: float(value) for key, value in act_scales.items()})
    ptq["save_act_params"](
        str(paths.act_params_json),
        abit=args.a_bit,
        signed=True,
        percentile=args.percentile,
        estimator="batch_max_percentile",
        layer_types=("KANLayer",),
    )
    save_predictions(paths.predictions_csv, dataset.y_test, test_predictions, dataset.target_scale)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_width": serialize_width(model.width),
            "grid": experiment.model.grid,
            "k": experiment.model.k,
            "target_scale": experiment.data.target_scale,
            "feature_names": dataset.feature_names,
            "metrics": metrics,
            "export": export_payload,
            "quantization": export_payload["quantization"],
            "config": experiment.raw_config,
        },
        paths.checkpoint,
    )

    print("\nNASA QuantKAN PTQ risultati finali")
    print(f"Config: {experiment.config_path}")
    print(f"Checkpoint fp32: {checkpoint_path}")
    print(f"Export quantizzato: {paths.export_json}")
    print(f"Predictions quantizzate: {paths.predictions_csv}")
    print(
        "Test metrics: "
        + json.dumps(test_metrics, ensure_ascii=True)
    )
    print(
        "Reference fp32 test metrics: "
        + json.dumps(fp32_test_metrics, ensure_ascii=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
