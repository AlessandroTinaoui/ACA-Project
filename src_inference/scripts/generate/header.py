#!/usr/bin/env python3
"""Generate include/kan_model.h from a KAN JSON export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_PATH = ROOT.parent / "artifacts" / "nasa_kan" / "nasa_kan_riscv_export.json"
HEADER_PATH = ROOT / "include" / "kan_model.h"
MODEL_ALIASES = {
    "default": DEFAULT_JSON_PATH,
    "kan": DEFAULT_JSON_PATH,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate include/kan_model.h from a KAN JSON export."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="default",
        help=(
            "Model alias or JSON path. Supported aliases: "
            + ", ".join(sorted(MODEL_ALIASES))
        ),
    )
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


def c_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return f"{text}f"


def require_key(data: dict, key: str):
    if key not in data:
        raise KeyError(f"Missing required key in JSON: {key}")
    return data[key]


def c_int_array(values: list[int]) -> str:
    return ", ".join(str(int(value)) for value in values)


def format_1d_int_array(values: list[int], indent: str) -> str:
    lines = []
    for index in range(0, len(values), 8):
        chunk = values[index : index + 8]
        lines.append(indent + ", ".join(str(int(value)) for value in chunk))
    return ",\n".join(lines) if lines else indent + "0"


def format_2d_int_array(values: list[list[int]], indent: str) -> str:
    rows = []
    inner_indent = indent + "    "
    for row in values:
        rows.append(
            indent + "{\n" + format_1d_int_array(row, inner_indent) + "\n" + indent + "}"
        )
    return ",\n".join(rows)


def normalize_layers(data: dict) -> list[dict]:
    if "layers" in data:
        return data["layers"]

    return [
        {
            "layer_index": 0,
            "input_dim": int(require_key(data, "input_dim")),
            "output_dim": int(require_key(data, "output_dim")),
            "knots": [require_key(data, "knots")],
            "control_points": [[require_key(data, "control_points")]],
            "scale_base": [[float(require_key(data, "scale_base"))]],
            "scale_sp": [[float(require_key(data, "scale_sp"))]],
            "mask": [[1.0]],
        }
    ]


def zero_knots(num_knots: int) -> list[float]:
    return [0.0] * num_knots


def zero_control_points(num_control_points: int) -> list[float]:
    return [0.0] * num_control_points


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


def build_knot_metadata(
    dense_knots: list[list[list[float]]],
    layer_input_dims: list[int],
    max_input_dim: int,
) -> tuple[list[list[int]], list[list[float]], list[list[float]], list[list[float]], bool]:
    uniform_flags: list[list[int]] = []
    knot_bases: list[list[float]] = []
    knot_deltas: list[list[float]] = []
    knot_inv_deltas: list[list[float]] = []
    all_active_uniform = True

    for layer_index, layer_knots in enumerate(dense_knots):
        layer_flags: list[int] = []
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

            layer_flags.append(1 if uniform else 0)
            layer_bases.append(float(knot_vector[0]) if uniform else 0.0)
            layer_deltas.append(delta)
            layer_inv_deltas.append(1.0 / delta if uniform else 0.0)

        uniform_flags.append(layer_flags)
        knot_bases.append(layer_bases)
        knot_deltas.append(layer_deltas)
        knot_inv_deltas.append(layer_inv_deltas)

    return uniform_flags, knot_bases, knot_deltas, knot_inv_deltas, all_active_uniform


def format_1d_float_array(values: list[float], indent: str) -> str:
    lines = []
    for index in range(0, len(values), 4):
        chunk = values[index : index + 4]
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


def format_3d_float_array(values: list[list[list[float]]], indent: str) -> str:
    rows = []
    inner_indent = indent + "    "
    for matrix in values:
        rows.append(
            indent + "{\n" + format_2d_float_array(matrix, inner_indent) + "\n" + indent + "}"
        )
    return ",\n".join(rows)


def format_4d_float_array(
    values: list[list[list[list[float]]]],
    indent: str,
) -> str:
    rows = []
    inner_indent = indent + "    "
    for tensor in values:
        rows.append(
            indent + "{\n" + format_3d_float_array(tensor, inner_indent) + "\n" + indent + "}"
        )
    return ",\n".join(rows)


def build_dense_layers(
    layers: list[dict],
    num_layers: int,
    max_input_dim: int,
    max_output_dim: int,
    num_knots: int,
    num_control_points: int,
):
    dense_knots = []
    dense_control_points = []
    dense_scale_sp = []
    dense_scale_base = []
    dense_mask = []

    for layer in layers:
        input_dim = int(layer["input_dim"])
        output_dim = int(layer["output_dim"])

        layer_knots = [zero_knots(num_knots) for _ in range(max_input_dim)]
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

        layer_cp = [
            [zero_control_points(num_control_points) for _ in range(max_output_dim)]
            for _ in range(max_input_dim)
        ]
        layer_scale_sp = [[0.0 for _ in range(max_output_dim)] for _ in range(max_input_dim)]
        layer_scale_base = [[0.0 for _ in range(max_output_dim)] for _ in range(max_input_dim)]
        layer_mask = [[0.0 for _ in range(max_output_dim)] for _ in range(max_input_dim)]

        raw_cp = layer["control_points"]
        raw_scale_sp = layer["scale_sp"]
        raw_scale_base = layer["scale_base"]
        raw_mask = layer["mask"]

        if len(raw_cp) != input_dim:
            raise ValueError(
                f"layer {layer['layer_index']}: expected {input_dim} control-point rows, found {len(raw_cp)}"
            )

        for input_index in range(input_dim):
            if len(raw_cp[input_index]) != output_dim:
                raise ValueError(
                    f"layer {layer['layer_index']}: expected {output_dim} outputs for input {input_index}, found {len(raw_cp[input_index])}"
                )
            for output_index in range(output_dim):
                cp_vector = raw_cp[input_index][output_index]
                if len(cp_vector) != num_control_points:
                    raise ValueError(
                        f"layer {layer['layer_index']}: expected {num_control_points} control points, found {len(cp_vector)}"
                    )
                layer_cp[input_index][output_index] = [float(value) for value in cp_vector]
                layer_scale_sp[input_index][output_index] = float(raw_scale_sp[input_index][output_index])
                layer_scale_base[input_index][output_index] = float(raw_scale_base[input_index][output_index])
                layer_mask[input_index][output_index] = float(raw_mask[input_index][output_index])

        dense_control_points.append(layer_cp)
        dense_scale_sp.append(layer_scale_sp)
        dense_scale_base.append(layer_scale_base)
        dense_mask.append(layer_mask)

    if len(dense_knots) != num_layers:
        raise ValueError(f"expected {num_layers} layers, built {len(dense_knots)}")

    return dense_knots, dense_control_points, dense_scale_sp, dense_scale_base, dense_mask


def build_flat_edges(
    dense_control_points: list[list[list[list[float]]]],
    dense_scale_sp: list[list[list[float]]],
    dense_scale_base: list[list[list[float]]],
    dense_mask: list[list[list[float]]],
    layer_input_dims: list[int],
    layer_output_dims: list[int],
) -> tuple[list[int], list[list[float]], list[float], list[float], list[float]]:
    edge_offsets: list[int] = []
    edge_control_points: list[list[float]] = []
    edge_scale_sp: list[float] = []
    edge_scale_base: list[float] = []
    edge_mask: list[float] = []

    for layer_index, input_dim in enumerate(layer_input_dims):
        output_dim = layer_output_dims[layer_index]
        edge_offsets.append(len(edge_control_points))
        for input_index in range(input_dim):
            for output_index in range(output_dim):
                edge_control_points.append(
                    dense_control_points[layer_index][input_index][output_index]
                )
                edge_scale_sp.append(
                    dense_scale_sp[layer_index][input_index][output_index]
                )
                edge_scale_base.append(
                    dense_scale_base[layer_index][input_index][output_index]
                )
                edge_mask.append(
                    dense_mask[layer_index][input_index][output_index]
                )

    return edge_offsets, edge_control_points, edge_scale_sp, edge_scale_base, edge_mask


def all_close_to(values: list[float], expected: float, tolerance: float = 1e-7) -> bool:
    return all(abs(float(value) - expected) <= tolerance for value in values)


def main() -> None:
    args = parse_args()
    json_path = resolve_model_path(args.model)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    width = [int(value) for value in data.get("width", [])]
    input_dim = int(require_key(data, "input_dim"))
    output_dim = int(require_key(data, "output_dim"))
    degree = int(require_key(data, "degree"))
    num_control_points = int(require_key(data, "num_control_points"))
    num_knots = int(require_key(data, "num_knots"))
    num_intervals = int(data.get("num_intervals", num_knots - 1 - 2 * degree))
    x_min = float(require_key(data, "x_min"))
    x_max = float(require_key(data, "x_max"))
    target_scale = float(data.get("target_scale", 1.0))
    layers = normalize_layers(data)
    num_layers = len(layers)

    if not width:
        width = [input_dim]
        for layer in layers:
            width.append(int(layer["output_dim"]))

    if len(width) != num_layers + 1:
        raise ValueError(
            f"width has {len(width)} entries but expected {num_layers + 1} for {num_layers} layers"
        )
    if width[0] != input_dim or width[-1] != output_dim:
        raise ValueError("width endpoints do not match input_dim/output_dim")

    max_input_dim = max(int(layer["input_dim"]) for layer in layers)
    max_output_dim = max(int(layer["output_dim"]) for layer in layers)
    max_layer_width = max(width)

    dense_knots, dense_control_points, dense_scale_sp, dense_scale_base, dense_mask = build_dense_layers(
        layers,
        num_layers,
        max_input_dim,
        max_output_dim,
        num_knots,
        num_control_points,
    )

    layer_input_dims = [int(layer["input_dim"]) for layer in layers]
    layer_output_dims = [int(layer["output_dim"]) for layer in layers]
    (
        knot_uniform_flags,
        knot_bases,
        knot_deltas,
        knot_inv_deltas,
        all_active_knots_uniform,
    ) = build_knot_metadata(
        dense_knots,
        layer_input_dims,
        max_input_dim,
    )
    (
        layer_edge_offsets,
        edge_control_points,
        edge_scale_sp,
        edge_scale_base,
        edge_mask,
    ) = build_flat_edges(
        dense_control_points,
        dense_scale_sp,
        dense_scale_base,
        dense_mask,
        layer_input_dims,
        layer_output_dims,
    )
    has_edge_masks = not all_close_to(edge_mask, 1.0)
    base_branch_enabled = bool(data.get("base_branch_enabled", False)) or not all_close_to(
        edge_scale_base,
        0.0,
    )
    edge_mask_block = ""
    if has_edge_masks:
        edge_mask_block = f"""
static const float KAN_EDGE_MASK[KAN_NUM_EDGES] = {{
{format_1d_float_array(edge_mask, "    ")}
}};
"""
    full_knot_block = ""
    if not all_active_knots_uniform:
        full_knot_block = f"""
static const float KAN_LAYER_KNOTS[KAN_NUM_LAYERS][KAN_MAX_INPUT_DIM][KAN_NUM_KNOTS] = {{
{format_3d_float_array(dense_knots, "    ")}
}};
"""

    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.write_text(
        f"""#ifndef KAN_MODEL_H
#define KAN_MODEL_H

/*
 * Auto-generated by scripts/generate/header.py.
 * Source JSON: {json_path}
 * Do not edit this file by hand; regenerate it after changing the JSON export.
 */

#define KAN_INPUT_DIM {input_dim}
#define KAN_OUTPUT_DIM {output_dim}
#define KAN_DEGREE {degree}
#define KAN_NUM_CONTROL_POINTS {num_control_points}
#define KAN_NUM_KNOTS {num_knots}
#define KAN_NUM_INTERVALS {num_intervals}
#define KAN_NUM_LAYERS {num_layers}
#define KAN_MAX_LAYER_WIDTH {max_layer_width}
#define KAN_MAX_INPUT_DIM {max_input_dim}
#define KAN_MAX_OUTPUT_DIM {max_output_dim}
#define KAN_NUM_EDGES {len(edge_control_points)}
#define KAN_HAS_KNOT_METADATA 1
#define KAN_USE_IMPLICIT_UNIFORM_KNOTS {1 if all_active_knots_uniform else 0}
#define KAN_HAS_FLAT_EDGES 1
#define KAN_HAS_EDGE_MASKS {1 if has_edge_masks else 0}
#define KAN_BASE_BRANCH_ENABLED {1 if base_branch_enabled else 0}

static const float KAN_X_MIN = {c_float(x_min)};
static const float KAN_X_MAX = {c_float(x_max)};
static const float KAN_TARGET_SCALE = {c_float(target_scale)};

static const int KAN_WIDTHS[KAN_NUM_LAYERS + 1] = {{
    {c_int_array(width)}
}};

static const int KAN_LAYER_INPUT_DIMS[KAN_NUM_LAYERS] = {{
    {c_int_array(layer_input_dims)}
}};

static const int KAN_LAYER_OUTPUT_DIMS[KAN_NUM_LAYERS] = {{
    {c_int_array(layer_output_dims)}
}};

static const int KAN_LAYER_EDGE_OFFSETS[KAN_NUM_LAYERS] = {{
    {c_int_array(layer_edge_offsets)}
}};

static const int KAN_LAYER_KNOTS_UNIFORM[KAN_NUM_LAYERS][KAN_MAX_INPUT_DIM] = {{
{format_2d_int_array(knot_uniform_flags, "    ")}
}};

static const float KAN_LAYER_KNOT_BASE[KAN_NUM_LAYERS][KAN_MAX_INPUT_DIM] = {{
{format_2d_float_array(knot_bases, "    ")}
}};

static const float KAN_LAYER_KNOT_DELTA[KAN_NUM_LAYERS][KAN_MAX_INPUT_DIM] = {{
{format_2d_float_array(knot_deltas, "    ")}
}};

static const float KAN_LAYER_KNOT_INV_DELTA[KAN_NUM_LAYERS][KAN_MAX_INPUT_DIM] = {{
{format_2d_float_array(knot_inv_deltas, "    ")}
}};
{full_knot_block}
static const float KAN_EDGE_CONTROL_POINTS[KAN_NUM_EDGES][KAN_NUM_CONTROL_POINTS] = {{
{format_2d_float_array(edge_control_points, "    ")}
}};

static const float KAN_EDGE_SCALE_SP[KAN_NUM_EDGES] = {{
{format_1d_float_array(edge_scale_sp, "    ")}
}};

static const float KAN_EDGE_SCALE_BASE[KAN_NUM_EDGES] = {{
{format_1d_float_array(edge_scale_base, "    ")}
}};
{edge_mask_block}

#endif /* KAN_MODEL_H */
""",
        encoding="utf-8",
    )

    print("Generated include/kan_model.h")
    print(f"source_json = {json_path}")
    print(f"model_type = {data.get('model_type', 'N/A')}")
    print(f"function = {data.get('function', 'N/A')}")
    print(f"width = {width}")
    print(f"input_dim = {input_dim}")
    print(f"output_dim = {output_dim}")
    print(f"degree = {degree}")
    print(f"num_control_points = {num_control_points}")
    print(f"num_knots = {num_knots}")
    print(f"num_intervals = {num_intervals}")
    print(f"num_layers = {num_layers}")
    print(f"num_edges = {len(edge_control_points)}")
    print(f"x_min = {x_min}")
    print(f"x_max = {x_max}")
    print(f"target_scale = {target_scale}")


if __name__ == "__main__":
    main()
