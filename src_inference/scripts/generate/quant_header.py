#!/usr/bin/env python3
"""Generate include/kan_model_quant.h from a QuantKAN PTQ JSON export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HEADER_PATH = ROOT / "include" / "kan_model_quant.h"
MODEL_ALIASES = {
    "quant16": ROOT.parent / "artifacts" / "nasa_kan" / "nasa_kan_quantkan_uniform_w16a16_pc_export.json",
    "quant8": ROOT.parent / "artifacts" / "nasa_kan" / "nasa_kan_quantkan_uniform_w8a8_pc_export.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate include/kan_model_quant.h from a quantized KAN JSON export."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="quant16",
        help="Model alias or JSON path.",
    )
    parser.add_argument("--bits", type=int, default=16, help="Integer bit-width for quantized weights.")
    parser.add_argument(
        "--lut-size",
        type=int,
        default=0,
        help="Deprecated compatibility flag. Ignored because spline basis is evaluated at runtime.",
    )
    parser.add_argument("--output", type=Path, default=HEADER_PATH, help="Output header path.")
    return parser.parse_args()


def resolve_model_path(model: str) -> Path:
    if model in MODEL_ALIASES:
        candidate = MODEL_ALIASES[model]
    else:
        candidate = Path(model)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
            if not candidate.exists():
                project_candidate = ROOT.parent / model
                if project_candidate.exists():
                    candidate = project_candidate

    candidate = candidate.resolve()
    if candidate.is_dir():
        json_files = sorted(candidate.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON export found in model directory: {candidate}")
        if len(json_files) > 1:
            names = ", ".join(path.name for path in json_files)
            raise ValueError(
                f"Multiple JSON exports found in {candidate}: {names}. "
                "Pass the JSON path explicitly."
            )
        return json_files[0].resolve()
    return candidate


def require_key(data: dict, key: str):
    if key not in data:
        raise KeyError(f"Missing required key in JSON: {key}")
    return data[key]


def normalize_layers(data: dict) -> list[dict]:
    layers = data.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("The quantized export must contain a non-empty 'layers' list.")
    return layers


def c_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return f"{text}f"


def format_1d_int_array(values: list[int], indent: str, chunk_size: int = 16) -> str:
    lines = []
    for index in range(0, len(values), chunk_size):
        chunk = values[index : index + chunk_size]
        lines.append(indent + ", ".join(str(int(value)) for value in chunk))
    return ",\n".join(lines) if lines else indent + "0"


def format_1d_float_array(values: list[float], indent: str, chunk_size: int = 8) -> str:
    lines = []
    for index in range(0, len(values), chunk_size):
        chunk = values[index : index + chunk_size]
        lines.append(indent + ", ".join(c_float(value) for value in chunk))
    return ",\n".join(lines) if lines else indent + "0.0f"


def format_2d_float_array(values: list[list[float]], indent: str) -> str:
    rows = []
    inner_indent = indent + "    "
    for row in values:
        rows.append(
            indent + "{\n" + format_1d_float_array(row, inner_indent) + "\n" + indent + "}"
        )
    return ",\n".join(rows)


def is_uniform_knot_vector(knots: list[float], tolerance: float = 1e-5) -> bool:
    if len(knots) < 2:
        return False
    delta = float(knots[1]) - float(knots[0])
    if abs(delta) <= tolerance:
        return False
    for left, right in zip(knots, knots[1:]):
        if abs((float(right) - float(left)) - delta) > tolerance:
            return False
    return True


def all_close_to(values: list[float], expected: float, tolerance: float = 1e-7) -> bool:
    return all(abs(float(value) - expected) <= tolerance for value in values)


def quantize_symmetric(value: float, scale: float, qmax: int) -> int:
    if scale <= 0.0:
        return 0
    quantized = int(round(value / scale))
    return max(-qmax, min(qmax, quantized))


def build_dense_knots(
    layers: list[dict],
    num_layers: int,
    max_input_dim: int,
    num_knots: int,
) -> list[list[list[float]]]:
    dense_knots: list[list[list[float]]] = []
    for layer in layers:
        input_dim = int(layer["input_dim"])
        layer_knots = [[0.0] * num_knots for _ in range(max_input_dim)]
        raw_knots = layer["knots"]
        if len(raw_knots) != input_dim:
            raise ValueError(
                f"layer {layer['layer_index']}: expected {input_dim} knot vectors, found {len(raw_knots)}"
            )
        for input_index, knot_vector in enumerate(raw_knots):
            if len(knot_vector) != num_knots:
                raise ValueError(
                    f"layer {layer['layer_index']}: expected {num_knots} knots, found {len(knot_vector)}"
                )
            layer_knots[input_index] = [float(value) for value in knot_vector]
        dense_knots.append(layer_knots)

    if len(dense_knots) != num_layers:
        raise ValueError(f"expected {num_layers} layers, built {len(dense_knots)}")
    return dense_knots


def build_knot_metadata(
    dense_knots: list[list[list[float]]],
    layer_input_dims: list[int],
    max_input_dim: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], bool]:
    knot_bases: list[list[float]] = []
    knot_deltas: list[list[float]] = []
    knot_inv_deltas: list[list[float]] = []
    all_active_uniform = True

    for layer_index, layer_knots in enumerate(dense_knots):
        layer_bases: list[float] = []
        layer_deltas: list[float] = []
        layer_inv_deltas: list[float] = []
        input_dim = layer_input_dims[layer_index]

        for input_index in range(max_input_dim):
            knot_vector = layer_knots[input_index]
            active = input_index < input_dim
            uniform = active and is_uniform_knot_vector(knot_vector)
            delta = float(knot_vector[1]) - float(knot_vector[0]) if uniform else 0.0
            if active and not uniform:
                all_active_uniform = False

            layer_bases.append(float(knot_vector[0]) if uniform else 0.0)
            layer_deltas.append(delta)
            layer_inv_deltas.append(1.0 / delta if uniform and delta != 0.0 else 0.0)

        knot_bases.append(layer_bases)
        knot_deltas.append(layer_deltas)
        knot_inv_deltas.append(layer_inv_deltas)

    return knot_bases, knot_deltas, knot_inv_deltas, all_active_uniform


def main() -> None:
    args = parse_args()
    if args.bits < 2 or args.bits > 16:
        raise ValueError(f"--bits must be in [2, 16], got {args.bits}")

    json_path = resolve_model_path(args.model)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    width = [int(value) for value in require_key(data, "width")]
    input_dim = int(require_key(data, "input_dim"))
    output_dim = int(require_key(data, "output_dim"))
    degree = int(require_key(data, "degree"))
    num_control_points = int(require_key(data, "num_control_points"))
    num_knots = int(require_key(data, "num_knots"))
    num_intervals = int(data.get("num_intervals", num_knots - 1 - 2 * degree))
    target_scale = float(data.get("target_scale", 1.0))
    layers = normalize_layers(data)
    num_layers = len(layers)

    if degree != 3:
        raise ValueError(
            f"The current integer KAN runtime supports only cubic splines (degree 3), got {degree}."
        )

    max_input_dim = max(int(layer["input_dim"]) for layer in layers)
    max_output_dim = max(int(layer["output_dim"]) for layer in layers)
    max_layer_width = max(width)
    qmax = (1 << (args.bits - 1)) - 1

    layer_input_dims = [int(layer["input_dim"]) for layer in layers]
    layer_output_dims = [int(layer["output_dim"]) for layer in layers]
    num_edges = sum(layer_input_dims[index] * layer_output_dims[index] for index in range(num_layers))
    activation_scales = data.get("quantization", {}).get("activation", {}).get("scales", {})
    layer_input_amax = [float(activation_scales.get(f"act_fun.{index}", 0.0)) for index in range(num_layers)]

    dense_knots = build_dense_knots(
        layers,
        num_layers,
        max_input_dim,
        num_knots,
    )
    knot_bases, knot_deltas, knot_inv_deltas, all_active_knots_uniform = build_knot_metadata(
        dense_knots,
        layer_input_dims,
        max_input_dim,
    )
    if not all_active_knots_uniform:
        raise ValueError("The quantized runtime currently supports only uniform knots.")

    edge_offsets: list[int] = []
    edge_control_points: list[list[int]] = []
    edge_control_dequant: list[float] = []
    edge_scale_base_q: list[int] = []
    edge_scale_base_dequant: list[float] = []

    for layer_index, layer in enumerate(layers):
        input_size = int(layer["input_dim"])
        output_size = int(layer["output_dim"])
        raw_cp = layer["control_points"]
        raw_scale_sp = layer["scale_sp"]
        raw_scale_base = layer["scale_base"]
        raw_mask = layer["mask"]

        edge_offsets.append(len(edge_control_points))

        for input_index in range(input_size):
            for output_index in range(output_size):
                mask = float(raw_mask[input_index][output_index])
                cp_float = [
                    mask * float(raw_scale_sp[input_index][output_index]) * float(value)
                    for value in raw_cp[input_index][output_index]
                ]
                max_abs_cp = max((abs(value) for value in cp_float), default=0.0)
                cp_scale = max_abs_cp / float(qmax) if max_abs_cp > 0.0 else 0.0
                edge_control_points.append(
                    [quantize_symmetric(value, cp_scale, qmax) for value in cp_float]
                )
                edge_control_dequant.append(cp_scale)

                base_float = mask * float(raw_scale_base[input_index][output_index])
                base_scale = abs(base_float) / float(qmax) if base_float != 0.0 else 0.0
                edge_scale_base_q.append(quantize_symmetric(base_float, base_scale, qmax))
                edge_scale_base_dequant.append(base_scale)

    has_base_branch = not all_close_to(edge_scale_base_dequant, 0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"""#ifndef KAN_MODEL_QUANT_H
#define KAN_MODEL_QUANT_H

#include <stdint.h>

/*
 * Auto-generated by scripts/generate/quant_header.py.
 * Source JSON: {json_path}
 * Quant bits: {args.bits}
 * Do not edit this file by hand.
 */

#define KANQ_INPUT_DIM {input_dim}
#define KANQ_OUTPUT_DIM {output_dim}
#define KANQ_DEGREE {degree}
#define KANQ_NUM_CONTROL_POINTS {num_control_points}
#define KANQ_NUM_KNOTS {num_knots}
#define KANQ_NUM_INTERVALS {num_intervals}
#define KANQ_NUM_LAYERS {num_layers}
#define KANQ_MAX_LAYER_WIDTH {max_layer_width}
#define KANQ_MAX_INPUT_DIM {max_input_dim}
#define KANQ_MAX_OUTPUT_DIM {max_output_dim}
#define KANQ_NUM_EDGES {num_edges}
#define KANQ_BITS {args.bits}
#define KANQ_QMAX {qmax}
#define KANQ_BASE_BRANCH_ENABLED {1 if has_base_branch else 0}

static const float KANQ_TARGET_SCALE = {c_float(target_scale)};

static const int KANQ_WIDTHS[KANQ_NUM_LAYERS + 1] = {{
{format_1d_int_array(width, "    ")}
}};

static const int KANQ_LAYER_INPUT_DIMS[KANQ_NUM_LAYERS] = {{
{format_1d_int_array(layer_input_dims, "    ")}
}};

static const int KANQ_LAYER_OUTPUT_DIMS[KANQ_NUM_LAYERS] = {{
{format_1d_int_array(layer_output_dims, "    ")}
}};

static const int KANQ_LAYER_EDGE_OFFSETS[KANQ_NUM_LAYERS] = {{
{format_1d_int_array(edge_offsets, "    ")}
}};

static const float KANQ_LAYER_INPUT_AMAX[KANQ_NUM_LAYERS] = {{
{format_1d_float_array(layer_input_amax, "    ")}
}};

static const float KANQ_LAYER_KNOT_BASE[KANQ_NUM_LAYERS][KANQ_MAX_INPUT_DIM] = {{
{format_2d_float_array(knot_bases, "    ")}
}};

static const float KANQ_LAYER_KNOT_DELTA[KANQ_NUM_LAYERS][KANQ_MAX_INPUT_DIM] = {{
{format_2d_float_array(knot_deltas, "    ")}
}};

static const float KANQ_LAYER_KNOT_INV_DELTA[KANQ_NUM_LAYERS][KANQ_MAX_INPUT_DIM] = {{
{format_2d_float_array(knot_inv_deltas, "    ")}
}};

static const int16_t KANQ_EDGE_CONTROL_POINTS[KANQ_NUM_EDGES][KANQ_NUM_CONTROL_POINTS] = {{
{",\n".join("    {" + ", ".join(str(value) for value in row) + "}" for row in edge_control_points)}
}};

static const float KANQ_EDGE_CONTROL_DEQUANT[KANQ_NUM_EDGES] = {{
{format_1d_float_array(edge_control_dequant, "    ")}
}};

static const int16_t KANQ_EDGE_SCALE_BASE_Q[KANQ_NUM_EDGES] = {{
{format_1d_int_array(edge_scale_base_q, "    ")}
}};

static const float KANQ_EDGE_SCALE_BASE_DEQUANT[KANQ_NUM_EDGES] = {{
{format_1d_float_array(edge_scale_base_dequant, "    ")}
}};

#endif /* KAN_MODEL_QUANT_H */
""",
        encoding="utf-8",
    )

    print(f"Generated {args.output}")
    print(f"source_json = {json_path}")
    print(f"bits = {args.bits}")


if __name__ == "__main__":
    main()
