#!/usr/bin/env python3
"""Generate include/kan_model_quant.h from a QuantKAN PTQ JSON export."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER_PATH = ROOT / "include" / "kan_model_quant.h"
MODEL_ALIASES = {
    "nasa_quant16": ROOT.parent / "artifacts" / "nasa_kan" / "nasa_kan_quantkan_uniform_w16a16_pc_export.json",
    "nasa_quant8": ROOT.parent / "artifacts" / "nasa_kan" / "nasa_kan_quantkan_uniform_w8a8_pc_export.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate include/kan_model_quant.h from a quantized KAN JSON export."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="nasa_quant16",
        help="Model alias or JSON path.",
    )
    parser.add_argument("--bits", type=int, default=16, help="Integer bit-width for the generated header.")
    parser.add_argument("--lut-size", type=int, default=4096, help="Per-support LUT size.")
    parser.add_argument("--output", type=Path, default=HEADER_PATH, help="Output header path.")
    return parser.parse_args()


def resolve_model_path(model: str) -> Path:
    if model in MODEL_ALIASES:
        candidate = MODEL_ALIASES[model]
    else:
        candidate = Path(model)
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()

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


def format_int_array(values: list[int], indent: str, chunk_size: int = 16) -> str:
    lines = []
    for index in range(0, len(values), chunk_size):
        chunk = values[index : index + chunk_size]
        lines.append(indent + ", ".join(str(int(value)) for value in chunk))
    return ",\n".join(lines) if lines else indent + "0"


def format_float_array(values: list[float], indent: str, chunk_size: int = 8) -> str:
    lines = []
    for index in range(0, len(values), chunk_size):
        chunk = values[index : index + chunk_size]
        lines.append(indent + ", ".join(c_float(value) for value in chunk))
    return ",\n".join(lines) if lines else indent + "0.0f"


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


def clamp_to_qmax(value: int, qmax: int) -> int:
    return max(-qmax, min(qmax, value))


def bspline_basis_values(
    x: float,
    knots: list[float],
    degree: int,
    num_control_points: int,
) -> list[float]:
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


def quantize_with_scale(value: float, scale: float, qmax: int) -> int:
    if scale <= 0.0:
        return 0
    return clamp_to_qmax(int(round(value / scale)), qmax)


def main() -> None:
    args = parse_args()
    if args.bits < 2 or args.bits > 16:
        raise ValueError(f"--bits must be in [2, 16], got {args.bits}")
    if args.lut_size <= 0:
        raise ValueError(f"--lut-size must be positive, got {args.lut_size}")

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
    max_input_dim = max(int(layer["input_dim"]) for layer in layers)
    max_output_dim = max(int(layer["output_dim"]) for layer in layers)
    max_layer_width = max(width)
    num_layer_slots = num_layers * max_input_dim
    num_edge_slots = num_layers * max_input_dim * max_output_dim
    qmax = (1 << (args.bits - 1)) - 1

    if degree != 3:
        raise ValueError(
            f"The current integer KAN runtime supports only cubic splines (degree 3), got {degree}."
        )

    activation_scales = data.get("quantization", {}).get("activation", {}).get("scales", {})
    layer_input_amax = [float(activation_scales.get(f"act_fun.{index}", 0.0)) for index in range(num_layers)]
    layer_input_dims = [int(layer["input_dim"]) for layer in layers]
    layer_output_dims = [int(layer["output_dim"]) for layer in layers]
    layer_edge_offsets: list[int] = []

    support_min = [0.0] * num_layer_slots
    support_max = [0.0] * num_layer_slots
    basis_first_cp = [0] * (num_layer_slots * args.lut_size)
    basis_pack = [0] * (num_layer_slots * args.lut_size * 4)
    base_lut = [0] * (num_layer_slots * args.lut_size)
    coeff4_pack = [0] * (num_edge_slots * num_control_points * 4)
    coeff_dequant = [0.0] * num_edge_slots
    base_weight = [0] * num_edge_slots
    base_dequant = [0.0] * num_edge_slots

    edge_counter = 0
    for layer_index, layer in enumerate(layers):
        input_size = int(layer["input_dim"])
        output_size = int(layer["output_dim"])
        layer_edge_offsets.append(edge_counter)

        knots_matrix = layer["knots"]
        control_points = layer["control_points"]
        scale_sp = layer["scale_sp"]
        scale_base = layer["scale_base"]
        mask = layer["mask"]

        for input_index in range(input_size):
            knots = [float(value) for value in knots_matrix[input_index]]
            if len(knots) != num_knots:
                raise ValueError(
                    f"Layer {layer_index} input {input_index}: expected {num_knots} knots, found {len(knots)}."
                )
            if not is_uniform_knot_vector(knots):
                raise ValueError(
                    f"Layer {layer_index} input {input_index}: non-uniform knots are not supported by the integer LUT path."
                )

            slot = layer_index * max_input_dim + input_index
            slot_support_min = float(knots[0])
            slot_support_max = float(knots[-1])
            support_min[slot] = slot_support_min
            support_max[slot] = slot_support_max

            max_abs_base = max(abs(silu(slot_support_min)), abs(silu(slot_support_max)))
            base_value_scale = max_abs_base / float(qmax) if max_abs_base > 0.0 else 0.0

            for lut_index in range(args.lut_size):
                if args.lut_size == 1:
                    x = 0.5 * (slot_support_min + slot_support_max)
                else:
                    x = slot_support_min + (slot_support_max - slot_support_min) * (
                        float(lut_index) / float(args.lut_size - 1)
                    )

                basis_values = bspline_basis_values(x, knots, degree, num_control_points)
                non_zero = [idx for idx, value in enumerate(basis_values) if abs(value) > 1e-10]
                first_cp = non_zero[0] if non_zero else 0
                slot_lut_index = slot * args.lut_size + lut_index
                basis_first_cp[slot_lut_index] = first_cp

                for local_index in range(4):
                    coeff_index = first_cp + local_index
                    basis_value = basis_values[coeff_index] if coeff_index < num_control_points else 0.0
                    basis_pack[slot_lut_index * 4 + local_index] = clamp_to_qmax(
                        int(round(basis_value * float(qmax))),
                        qmax,
                    )

                base_lut[slot_lut_index] = quantize_with_scale(silu(x), base_value_scale, qmax)

        for input_index in range(input_size):
            for output_index in range(output_size):
                slot = layer_index * max_input_dim + input_index
                edge_slot = (layer_index * max_input_dim + input_index) * max_output_dim + output_index
                edge_counter += 1

                edge_mask = float(mask[input_index][output_index])
                spline_weights = [
                    edge_mask * float(scale_sp[input_index][output_index]) * float(value)
                    for value in control_points[input_index][output_index]
                ]
                if len(spline_weights) != num_control_points:
                    raise ValueError(
                        f"Layer {layer_index} edge ({input_index}, {output_index}): "
                        f"expected {num_control_points} control points, found {len(spline_weights)}."
                    )
                max_abs_spline = max((abs(value) for value in spline_weights), default=0.0)
                spline_scale = max_abs_spline / float(qmax) if max_abs_spline > 0.0 else 0.0
                coeff_dequant[edge_slot] = spline_scale / float(qmax) if spline_scale > 0.0 else 0.0

                for first_cp in range(num_control_points):
                    base_index = ((edge_slot * num_control_points) + first_cp) * 4
                    for local_index in range(4):
                        coeff_index = first_cp + local_index
                        value = spline_weights[coeff_index] if coeff_index < num_control_points else 0.0
                        coeff4_pack[base_index + local_index] = quantize_with_scale(value, spline_scale, qmax)

                base_weight_value = edge_mask * float(scale_base[input_index][output_index])
                max_abs_base = abs(base_weight_value)
                base_weight_scale = max_abs_base / float(qmax) if max_abs_base > 0.0 else 0.0
                base_weight[edge_slot] = quantize_with_scale(base_weight_value, base_weight_scale, qmax)

                slot_base_scale = max(abs(silu(support_min[slot])), abs(silu(support_max[slot])))
                slot_base_scale = slot_base_scale / float(qmax) if slot_base_scale > 0.0 else 0.0
                base_dequant[edge_slot] = slot_base_scale * base_weight_scale

    header = f"""#ifndef KAN_MODEL_QUANT_H
#define KAN_MODEL_QUANT_H

#include <stdint.h>

/*
 * Auto-generated by scripts/json_to_quant_header.py.
 * Source JSON: {json_path}
 * Quant bits: {args.bits}
 * LUT size: {args.lut_size}
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
#define KANQ_NUM_EDGES {sum(layer_input_dims[index] * layer_output_dims[index] for index in range(num_layers))}
#define KANQ_NUM_LAYER_SLOTS {num_layer_slots}
#define KANQ_NUM_EDGE_SLOTS {num_edge_slots}
#define KANQ_NUM_LOCAL_BASIS 4
#define KANQ_BITS {args.bits}
#define KANQ_QMAX {qmax}
#define KANQ_LUT_SIZE {args.lut_size}
#define KANQ_BASE_BRANCH_ENABLED {1 if any(value != 0 for value in base_weight) else 0}

static const float KANQ_TARGET_SCALE = {c_float(target_scale)};

static const int KANQ_WIDTHS[KANQ_NUM_LAYERS + 1] = {{
{format_int_array(width, "    ")}
}};

static const int KANQ_LAYER_INPUT_DIMS[KANQ_NUM_LAYERS] = {{
{format_int_array(layer_input_dims, "    ")}
}};

static const int KANQ_LAYER_OUTPUT_DIMS[KANQ_NUM_LAYERS] = {{
{format_int_array(layer_output_dims, "    ")}
}};

static const int KANQ_LAYER_EDGE_OFFSETS[KANQ_NUM_LAYERS] = {{
{format_int_array(layer_edge_offsets, "    ")}
}};

static const float KANQ_LAYER_INPUT_AMAX[KANQ_NUM_LAYERS] = {{
{format_float_array(layer_input_amax, "    ")}
}};

static const float KANQ_LAYER_SUPPORT_MIN[KANQ_NUM_LAYER_SLOTS] = {{
{format_float_array(support_min, "    ")}
}};

static const float KANQ_LAYER_SUPPORT_MAX[KANQ_NUM_LAYER_SLOTS] = {{
{format_float_array(support_max, "    ")}
}};

static const uint8_t KANQ_LAYER_BASIS_FIRST_CP[KANQ_NUM_LAYER_SLOTS * KANQ_LUT_SIZE] = {{
{format_int_array(basis_first_cp, "    ", chunk_size=32)}
}};

static const int16_t KANQ_LAYER_BASIS_PACK[KANQ_NUM_LAYER_SLOTS * KANQ_LUT_SIZE * KANQ_NUM_LOCAL_BASIS] = {{
{format_int_array(basis_pack, "    ", chunk_size=24)}
}};

static const int16_t KANQ_LAYER_BASE_LUT[KANQ_NUM_LAYER_SLOTS * KANQ_LUT_SIZE] = {{
{format_int_array(base_lut, "    ", chunk_size=32)}
}};

static const int16_t KANQ_LAYER_COEFF4_PACK[KANQ_NUM_EDGE_SLOTS * KANQ_NUM_CONTROL_POINTS * KANQ_NUM_LOCAL_BASIS] = {{
{format_int_array(coeff4_pack, "    ", chunk_size=24)}
}};

static const float KANQ_LAYER_COEFF_DEQUANT[KANQ_NUM_EDGE_SLOTS] = {{
{format_float_array(coeff_dequant, "    ")}
}};

static const int16_t KANQ_LAYER_BASE_WEIGHT[KANQ_NUM_EDGE_SLOTS] = {{
{format_int_array(base_weight, "    ", chunk_size=24)}
}};

static const float KANQ_LAYER_BASE_DEQUANT[KANQ_NUM_EDGE_SLOTS] = {{
{format_float_array(base_dequant, "    ")}
}};

#endif /* KAN_MODEL_QUANT_H */
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header, encoding="utf-8")
    print(f"Generated {args.output}")
    print(f"source_json = {json_path}")
    print(f"bits = {args.bits}")
    print(f"lut_size = {args.lut_size}")


if __name__ == "__main__":
    main()
