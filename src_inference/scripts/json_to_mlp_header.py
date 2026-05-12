#!/usr/bin/env python3
"""Generate include/mlp_model.h from an MLP JSON export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = ROOT / "model" / "mlp" / "mlp_export.json"
HEADER_PATH = ROOT / "include" / "mlp_model.h"
ACTIVATIONS = {
    "identity": 0,
    "linear": 0,
    "none": 0,
    "tanh": 1,
    "relu": 2,
    "sigmoid": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate include/mlp_model.h from an MLP JSON export."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=str(DEFAULT_JSON_PATH),
        help="MLP JSON path. Default: model/mlp_export.json",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.is_dir():
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No MLP JSON export found in model directory: {path}")
        if len(json_files) > 1:
            names = ", ".join(item.name for item in json_files)
            raise ValueError(
                f"Multiple MLP JSON exports found in {path}: {names}. "
                "Pass the JSON path explicitly."
            )
        return json_files[0].resolve()
    return path


def require_key(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ValueError(f"{where}: missing required key '{key}'")
    return data[key]


def require_int(data: dict[str, Any], key: str, where: str) -> int:
    value = require_key(data, key, where)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where}: '{key}' must be a positive integer")
    return value


def optional_positive_int(
    data: dict[str, Any],
    keys: tuple[str, ...],
    where: str,
) -> int | None:
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{where}: '{key}' must be a positive integer")
        return value
    return None


def normalize_activation(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where}: activation must be a string")
    name = value.strip().lower()
    if name not in ACTIVATIONS:
        allowed = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"{where}: unsupported activation '{value}'. Allowed: {allowed}")
    return "identity" if name in {"linear", "none"} else name


def as_float_matrix(value: Any, rows: int, cols: int, where: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{where}: weights must have {rows} rows")
    matrix: list[list[float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(
                f"{where}: weights[{row_index}] must have {cols} values"
            )
        matrix.append([float(item) for item in row])
    return matrix


def as_float_vector(value: Any, size: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{where}: biases must have {size} values")
    return [float(item) for item in value]


def normalize_layers(data: dict[str, Any]) -> list[dict[str, Any]]:
    if "layers" in data:
        layers = require_key(data, "layers", "root")
        if not isinstance(layers, list) or not layers:
            raise ValueError("root: 'layers' must be a non-empty list")
        return layers

    weights = require_key(data, "weights", "root")
    biases = require_key(data, "biases", "root")
    activations = data.get("activations", data.get("activation"))

    if not isinstance(weights, list) or not weights:
        raise ValueError("root: 'weights' must be a non-empty list")
    if not isinstance(biases, list) or len(biases) != len(weights):
        raise ValueError("root: 'biases' must be a list with one entry per layer")

    if isinstance(activations, str):
        activation_list = [activations] * len(weights)
    elif isinstance(activations, list) and len(activations) == len(weights):
        activation_list = activations
    else:
        raise ValueError(
            "root: provide 'layers' or provide 'weights', 'biases', and "
            "'activation'/'activations'"
        )

    layers = []
    previous_dim = require_int(data, "input_dim", "root")
    for index, weight_matrix in enumerate(weights):
        if not isinstance(weight_matrix, list) or not weight_matrix:
            raise ValueError(f"layer {index}: weight matrix must be non-empty")
        output_dim = len(weight_matrix)
        layers.append(
            {
                "input_dim": previous_dim,
                "output_dim": output_dim,
                "weights": weight_matrix,
                "biases": biases[index],
                "activation": activation_list[index],
            }
        )
        previous_dim = output_dim
    return layers


def c_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return f"{text}f"


def format_float_array(values: list[float], indent: str) -> str:
    lines = []
    for index in range(0, len(values), 6):
        chunk = values[index : index + 6]
        lines.append(indent + ", ".join(c_float(value) for value in chunk))
    return ",\n".join(lines) if lines else indent + "0.0f"


def format_int_array(values: list[int], indent: str) -> str:
    return indent + ", ".join(str(int(value)) for value in values)


def main() -> None:
    args = parse_args()
    json_path = resolve_path(args.model)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    input_dim = require_int(data, "input_dim", "root")
    output_dim = require_int(data, "output_dim", "root")
    raw_layers = normalize_layers(data)

    layer_sizes = [input_dim]
    layer_activations: list[int] = []
    flat_weights: list[float] = []
    flat_biases: list[float] = []

    previous_dim = input_dim
    for layer_index, layer in enumerate(raw_layers):
        where = f"layer {layer_index}"
        layer_input_dim = optional_positive_int(
            layer,
            ("input_dim", "in_features"),
            where,
        )
        layer_output_dim = optional_positive_int(
            layer,
            ("output_dim", "out_features"),
            where,
        )

        weights_value = require_key(layer, "weights", where)
        if layer_input_dim is None:
            layer_input_dim = len(weights_value[0]) if weights_value else None
        if layer_output_dim is None:
            layer_output_dim = len(weights_value) if weights_value else None
        if layer_input_dim is None or layer_output_dim is None:
            raise ValueError(
                f"{where}: missing layer dimensions; provide input_dim/output_dim "
                "or in_features/out_features"
            )
        if layer_input_dim != previous_dim:
            raise ValueError(
                f"{where}: input dimension {layer_input_dim} does not match previous "
                f"layer output dimension {previous_dim}"
            )
        weights = as_float_matrix(weights_value, layer_output_dim, layer_input_dim, where)
        bias_value = layer.get("biases", layer.get("bias"))
        if bias_value is None:
            raise ValueError(f"{where}: missing required key 'biases' or 'bias'")
        biases = as_float_vector(bias_value, layer_output_dim, where)

        activation_value = layer.get("activation")
        if activation_value is None and layer_index == len(raw_layers) - 1:
            activation_value = data.get("output_activation", "identity")
        if activation_value is None:
            activation_value = require_key(data, "activation", "root")
        activation_name = normalize_activation(activation_value, where)

        for row in weights:
            flat_weights.extend(row)
        flat_biases.extend(biases)
        layer_activations.append(ACTIVATIONS[activation_name])
        layer_sizes.append(layer_output_dim)
        previous_dim = layer_output_dim

    if layer_sizes[-1] != output_dim:
        raise ValueError(
            f"root: output_dim={output_dim} does not match final layer size={layer_sizes[-1]}"
        )

    total_weights = len(flat_weights)
    total_biases = len(flat_biases)
    max_layer_width = max(layer_sizes)

    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.write_text(
        f"""#ifndef MLP_MODEL_H
#define MLP_MODEL_H

/*
 * Auto-generated by scripts/json_to_mlp_header.py.
 * Source JSON: {json_path}
 * Do not edit this file by hand; regenerate it after changing the JSON export.
 *
 * MLP_WEIGHTS is flattened row-major as W[layer][out_neuron][in_neuron].
 * For each layer, advance by output_dim * input_dim weights.
 */

#define MLP_ACT_IDENTITY 0
#define MLP_ACT_TANH 1
#define MLP_ACT_RELU 2
#define MLP_ACT_SIGMOID 3

#define MLP_INPUT_DIM {input_dim}
#define MLP_OUTPUT_DIM {output_dim}
#define MLP_NUM_LAYERS {len(raw_layers)}
#define MLP_MAX_LAYER_WIDTH {max_layer_width}
#define MLP_TOTAL_WEIGHTS {total_weights}
#define MLP_TOTAL_BIASES {total_biases}

static const int MLP_LAYER_SIZES[MLP_NUM_LAYERS + 1] = {{
{format_int_array(layer_sizes, "    ")}
}};

static const int MLP_LAYER_ACTIVATIONS[MLP_NUM_LAYERS] = {{
{format_int_array(layer_activations, "    ")}
}};

static const float MLP_WEIGHTS[MLP_TOTAL_WEIGHTS] = {{
{format_float_array(flat_weights, "    ")}
}};

static const float MLP_BIASES[MLP_TOTAL_BIASES] = {{
{format_float_array(flat_biases, "    ")}
}};

#endif /* MLP_MODEL_H */
""",
        encoding="utf-8",
    )

    print("Generated include/mlp_model.h")
    print(f"source_json = {json_path}")
    print(f"model_type = {data.get('model_type', 'N/A')}")
    print(f"function = {data.get('function', 'N/A')}")
    print(f"layer_sizes = {layer_sizes}")
    print(f"input_dim = {input_dim}")
    print(f"output_dim = {output_dim}")
    print(f"num_layers = {len(raw_layers)}")
    print(f"total_weights = {total_weights}")
    print(f"total_biases = {total_biases}")


if __name__ == "__main__":
    main()
