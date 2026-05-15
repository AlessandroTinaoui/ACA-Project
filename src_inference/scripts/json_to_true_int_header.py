#!/usr/bin/env python3
"""Generate include/kan_model_true_int.h for the real integer NASA path."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER_PATH = ROOT / "include" / "kan_model_true_int.h"
MODEL_ALIASES = {
    "nasa_true_int16": ROOT.parent / "artifacts" / "nasa_kan" / "nasa_kan_quantkan_uniform_w16a16_pc_export.json",
    "nasa_true_int8": ROOT.parent / "artifacts" / "nasa_kan" / "nasa_kan_quantkan_uniform_w8a8_pc_export.json",
}
REQUIANT_SHIFT = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate include/kan_model_true_int.h from a quantized NASA KAN export."
    )
    parser.add_argument("model", nargs="?", default="nasa_true_int16")
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--output", type=Path, default=HEADER_PATH)
    return parser.parse_args()


def resolve_model_path(model: str) -> Path:
    if model in MODEL_ALIASES:
        candidate = MODEL_ALIASES[model]
    else:
        candidate = Path(model)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
    candidate = candidate.resolve()
    if candidate.is_dir():
        json_files = sorted(candidate.glob("*.json"))
        if len(json_files) != 1:
            raise ValueError(f"Expected exactly one JSON file in {candidate}, found {len(json_files)}.")
        return json_files[0]
    return candidate


def require_key(data: dict, key: str):
    if key not in data:
        raise KeyError(f"Missing required key in JSON: {key}")
    return data[key]


def c_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return f"{text}f"


def format_1d_int_array(values: list[int], indent: str, chunk_size: int = 12) -> str:
    lines = []
    for index in range(0, len(values), chunk_size):
        chunk = values[index : index + chunk_size]
        lines.append(indent + ", ".join(str(int(value)) for value in chunk))
    return ",\n".join(lines) if lines else indent + "0"


def format_1d_i64_array(values: list[int], indent: str, chunk_size: int = 8) -> str:
    lines = []
    for index in range(0, len(values), chunk_size):
        chunk = values[index : index + chunk_size]
        lines.append(indent + ", ".join(f"{int(value)}LL" for value in chunk))
    return ",\n".join(lines) if lines else indent + "0LL"


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


def format_2d_int_array(values: list[list[int]], indent: str) -> str:
    rows = []
    inner_indent = indent + "    "
    for row in values:
        rows.append(
            indent + "{\n" + format_1d_int_array(row, inner_indent) + "\n" + indent + "}"
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


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def quantize_symmetric(value: float, scale: float, qmax: int) -> int:
    if scale <= 0.0:
        return 0
    quantized = int(round(value / scale))
    return max(-qmax, min(qmax, quantized))


def bspline_basis_values(x: float, knots: list[float], degree: int, num_control_points: int) -> list[float]:
    if x >= knots[-1]:
        x = math.nextafter(knots[-1], knots[0])

    num_intervals = len(knots) - 1
    prev = [0.0] * num_intervals
    for index in range(num_intervals):
        if knots[index] <= x < knots[index + 1]:
            prev[index] = 1.0

    for order in range(1, degree + 1):
        curr = [0.0] * num_intervals
        for index in range(num_intervals - order):
            left = 0.0
            left_den = knots[index + order] - knots[index]
            if left_den > 0.0:
                left = ((x - knots[index]) / left_den) * prev[index]

            right = 0.0
            right_den = knots[index + order + 1] - knots[index + 1]
            if right_den > 0.0:
                right = ((knots[index + order + 1] - x) / right_den) * prev[index + 1]
            curr[index] = left + right
        prev = curr

    return prev[:num_control_points]


def companion_predictions(export_json: Path) -> Path:
    return export_json.with_name(export_json.stem.replace("_export", "_test_predictions") + ".csv")


def load_final_output_amax(export_json: Path, target_scale: float) -> float:
    csv_path = companion_predictions(export_json)
    if not csv_path.exists():
        return 1.0

    max_abs = 0.0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "predicted_rul" not in (reader.fieldnames or []):
            return 1.0
        for row in reader:
            value = abs(float(row["predicted_rul"]) / target_scale)
            if value > max_abs:
                max_abs = value
    return max_abs if max_abs > 0.0 else 1.0


def all_close(values: list[float], reference: float, tolerance: float = 1e-7) -> bool:
    return all(abs(value - reference) <= tolerance for value in values)


def main() -> None:
    args = parse_args()
    if args.bits not in {8, 16}:
        raise ValueError("--bits must be 8 or 16")

    export_json = resolve_model_path(args.model)
    data = json.loads(export_json.read_text(encoding="utf-8"))
    layers = require_key(data, "layers")
    width = [int(value) for value in require_key(data, "width")]
    input_dim = int(require_key(data, "input_dim"))
    output_dim = int(require_key(data, "output_dim"))
    degree = int(require_key(data, "degree"))
    num_control_points = int(require_key(data, "num_control_points"))
    num_knots = int(require_key(data, "num_knots"))
    num_intervals = int(data.get("num_intervals", num_knots - 1 - 2 * degree))
    target_scale = float(data.get("target_scale", 1.0))
    num_layers = len(layers)
    max_input_dim = max(int(layer["input_dim"]) for layer in layers)
    max_output_dim = max(int(layer["output_dim"]) for layer in layers)
    max_layer_width = max(width)
    qmax = (1 << (args.bits - 1)) - 1
    basis_qmax = 32767
    layer_input_dims = [int(layer["input_dim"]) for layer in layers]
    layer_output_dims = [int(layer["output_dim"]) for layer in layers]
    num_edges = sum(layer_input_dims[index] * layer_output_dims[index] for index in range(num_layers))
    layer_input_amax = [float(data["quantization"]["activation"]["scales"][f"act_fun.{index}"]) for index in range(num_layers)]
    layer_input_step = [amax / float(qmax) for amax in layer_input_amax]
    final_output_amax = load_final_output_amax(export_json, target_scale)
    final_output_step = final_output_amax / float(qmax)
    output_steps = [layer_input_step[index + 1] if index + 1 < num_layers else final_output_step for index in range(num_layers)]

    common_knots: list[list[float]] = []
    layer_knot_base: list[float] = []
    layer_knot_inv_delta: list[float] = []
    for layer_index, layer in enumerate(layers):
        active_knots = [[float(value) for value in row] for row in layer["knots"][: layer_input_dims[layer_index]]]
        if not active_knots:
            raise ValueError(f"Layer {layer_index} has no active knot vectors.")
        if not all(is_uniform_knot_vector(knots) for knots in active_knots):
            raise ValueError(f"Layer {layer_index}: non-uniform knots are not supported by the true-int path.")
        reference = active_knots[0]
        for knots in active_knots[1:]:
            if any(abs(left - right) > 1e-6 for left, right in zip(reference, knots)):
                raise ValueError(
                    f"Layer {layer_index}: active inputs use different uniform knot grids; true-int path expects a shared grid."
                )
        common_knots.append(reference)
        layer_knot_base.append(reference[0])
        delta = reference[1] - reference[0]
        layer_knot_inv_delta.append(1.0 / delta if delta != 0.0 else 0.0)

    basis_lut_entries = 2 * qmax + 1
    basis_first_cp: list[int] = []
    basis_q15: list[int] = []
    base_lut_q: list[int] = []
    base_value_scale: list[float] = []
    for layer_index in range(num_layers):
        amax = layer_input_amax[layer_index]
        step = layer_input_step[layer_index]
        knots = common_knots[layer_index]
        max_abs_base = max(abs(silu(-amax)), abs(silu(amax)))
        if max_abs_base <= 0.0:
            max_abs_base = 1.0
        base_scale = max_abs_base / float(qmax)
        base_value_scale.append(base_scale)

        for code in range(-qmax, qmax + 1):
            x = float(code) * step
            basis = bspline_basis_values(x, knots, degree, num_control_points)
            active = [idx for idx, value in enumerate(basis) if abs(value) > 1e-10]
            first_cp = active[0] if active else 0
            basis_first_cp.append(first_cp)
            for local_index in range(4):
                cp_index = first_cp + local_index
                value = basis[cp_index] if cp_index < num_control_points else 0.0
                basis_q15.append(max(-basis_qmax, min(basis_qmax, int(round(value * basis_qmax)))))
            base_lut_q.append(quantize_symmetric(silu(x), base_scale, qmax))

    edge_offsets: list[int] = []
    edge_control_points_q: list[list[int]] = []
    edge_control_mul: list[int] = []
    edge_scale_base_q: list[int] = []
    edge_scale_base_mul: list[int] = []

    for layer_index, layer in enumerate(layers):
        input_size = int(layer["input_dim"])
        output_size = int(layer["output_dim"])
        edge_offsets.append(len(edge_control_points_q))

        for input_index in range(input_size):
            for output_index in range(output_size):
                mask = float(layer["mask"][input_index][output_index])
                cp_real = [
                    mask * float(layer["scale_sp"][input_index][output_index]) * float(value)
                    for value in layer["control_points"][input_index][output_index]
                ]
                cp_scale = max((abs(value) for value in cp_real), default=0.0) / float(qmax) if cp_real else 0.0
                edge_control_points_q.append(
                    [quantize_symmetric(value, cp_scale, qmax) for value in cp_real]
                )
                control_real_scale = cp_scale / float(basis_qmax)
                control_mul = int(round((control_real_scale / output_steps[layer_index]) * (1 << REQUIANT_SHIFT))) if output_steps[layer_index] > 0.0 else 0
                edge_control_mul.append(control_mul)

                base_weight_real = mask * float(layer["scale_base"][input_index][output_index])
                base_weight_scale = abs(base_weight_real) / float(qmax) if base_weight_real != 0.0 else 0.0
                edge_scale_base_q.append(quantize_symmetric(base_weight_real, base_weight_scale, qmax))
                base_real_scale = base_value_scale[layer_index] * base_weight_scale
                base_mul = int(round((base_real_scale / output_steps[layer_index]) * (1 << REQUIANT_SHIFT))) if output_steps[layer_index] > 0.0 else 0
                edge_scale_base_mul.append(base_mul)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"""#ifndef KAN_MODEL_TRUE_INT_H
#define KAN_MODEL_TRUE_INT_H

#include <stdint.h>

/*
 * Auto-generated by scripts/json_to_true_int_header.py.
 * Source JSON: {export_json}
 * Quant bits: {args.bits}
 * Do not edit this file by hand.
 */

#define KATI_INPUT_DIM {input_dim}
#define KATI_OUTPUT_DIM {output_dim}
#define KATI_DEGREE {degree}
#define KATI_NUM_CONTROL_POINTS {num_control_points}
#define KATI_NUM_KNOTS {num_knots}
#define KATI_NUM_INTERVALS {num_intervals}
#define KATI_NUM_LAYERS {num_layers}
#define KATI_MAX_LAYER_WIDTH {max_layer_width}
#define KATI_MAX_INPUT_DIM {max_input_dim}
#define KATI_MAX_OUTPUT_DIM {max_output_dim}
#define KATI_NUM_EDGES {num_edges}
#define KATI_BITS {args.bits}
#define KATI_QMAX {qmax}
#define KATI_NUM_LOCAL_BASIS 4
#define KATI_BASIS_LUT_ENTRIES {basis_lut_entries}
#define KATI_REQUANT_SHIFT {REQUIANT_SHIFT}
#define KATI_REQUANT_ROUND {(1 << (REQUIANT_SHIFT - 1))}
#define KATI_BASE_BRANCH_ENABLED {0 if all_close(edge_scale_base_mul, 0.0) else 1}

static const float KATI_TARGET_SCALE = {c_float(target_scale)};
static const float KATI_FINAL_OUTPUT_STEP = {c_float(final_output_step)};

static const int KATI_WIDTHS[KATI_NUM_LAYERS + 1] = {{
{format_1d_int_array(width, "    ")}
}};

static const int KATI_LAYER_INPUT_DIMS[KATI_NUM_LAYERS] = {{
{format_1d_int_array(layer_input_dims, "    ")}
}};

static const int KATI_LAYER_OUTPUT_DIMS[KATI_NUM_LAYERS] = {{
{format_1d_int_array(layer_output_dims, "    ")}
}};

static const int KATI_LAYER_EDGE_OFFSETS[KATI_NUM_LAYERS] = {{
{format_1d_int_array(edge_offsets, "    ")}
}};

static const float KATI_LAYER_INPUT_STEP[KATI_NUM_LAYERS] = {{
{format_1d_float_array(layer_input_step, "    ")}
}};

static const float KATI_LAYER_KNOT_BASE[KATI_NUM_LAYERS] = {{
{format_1d_float_array(layer_knot_base, "    ")}
}};

static const float KATI_LAYER_KNOT_INV_DELTA[KATI_NUM_LAYERS] = {{
{format_1d_float_array(layer_knot_inv_delta, "    ")}
}};

static const int16_t KATI_LAYER_BASIS_Q15[KATI_NUM_LAYERS * KATI_BASIS_LUT_ENTRIES * KATI_NUM_LOCAL_BASIS] = {{
{format_1d_int_array(basis_q15, "    ", chunk_size=24)}
}};

static const int16_t KATI_LAYER_BASIS_FIRST_CP[KATI_NUM_LAYERS * KATI_BASIS_LUT_ENTRIES] = {{
{format_1d_int_array(basis_first_cp, "    ", chunk_size=24)}
}};

static const int16_t KATI_LAYER_BASE_LUT_Q[KATI_NUM_LAYERS * KATI_BASIS_LUT_ENTRIES] = {{
{format_1d_int_array(base_lut_q, "    ", chunk_size=24)}
}};

static const int16_t KATI_EDGE_CONTROL_POINTS_Q[KATI_NUM_EDGES][KATI_NUM_CONTROL_POINTS] = {{
{format_2d_int_array(edge_control_points_q, "    ")}
}};

static const int64_t KATI_EDGE_CONTROL_MUL[KATI_NUM_EDGES] = {{
{format_1d_i64_array(edge_control_mul, "    ")}
}};

static const int16_t KATI_EDGE_SCALE_BASE_Q[KATI_NUM_EDGES] = {{
{format_1d_int_array(edge_scale_base_q, "    ")}
}};

static const int64_t KATI_EDGE_SCALE_BASE_MUL[KATI_NUM_EDGES] = {{
{format_1d_i64_array(edge_scale_base_mul, "    ")}
}};

#endif /* KAN_MODEL_TRUE_INT_H */
""",
        encoding="utf-8",
    )

    print(f"Generated {args.output}")
    print(f"source_json = {export_json}")
    print(f"bits = {args.bits}")


if __name__ == "__main__":
    main()
