"""Plotting helpers shared by multiple conic experiments."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kan_models.common.kan_compat import KAN

from kan_models.models.conic.modeling import predict_classes


def save_final_loss_summary(
    loss_history: list[dict[str, object]],
    output_dir: Path,
) -> tuple[Path, Path] | None:
    """Save final and best loss values for each experiment stage."""
    if not loss_history:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in loss_history:
        train_loss = np.asarray(item["train_loss"], dtype=float)
        test_loss = np.asarray(item["test_loss"], dtype=float)
        rows.append(
            {
                "stage": int(item["stage"]),
                "task": str(item["task"]),
                "final_train_loss": float(item.get("final_train_loss", train_loss[-1])),
                "final_test_loss": float(item.get("final_test_loss", test_loss[-1])),
                "best_train_loss": float(np.min(train_loss)),
                "best_test_loss": float(np.min(test_loss)),
            }
        )

    csv_path = output_dir / "final_loss_summary.csv"
    json_path = output_dir / "final_loss_summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"final_stage_loss": rows[-1], "loss_by_stage": rows}, handle, indent=2)
        handle.write("\n")
    return csv_path, json_path


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    stage: int,
    task_name: str,
) -> tuple[Path, Path]:
    """Save a confusion matrix as both CSV and PNG."""
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    safe_task_name = task_name.replace("+", "_")
    csv_path = confusion_dir / f"stage_{stage:02d}_{safe_task_name}_confusion.csv"
    image_path = confusion_dir / f"stage_{stage:02d}_{safe_task_name}_confusion.png"

    frame = pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )
    frame.to_csv(csv_path)

    size = max(5.0, 1.2 * len(class_names))
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Stage {stage}: {task_name}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    max_value = int(matrix.max()) if matrix.size else 0
    threshold = max_value / 2 if max_value > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return csv_path, image_path


def edge_function_summary(
    model: KAN,
    input_names: list[str],
    output_names: list[str],
) -> pd.DataFrame:
    """Describe the active KAN edge functions in tabular form."""
    rows = []
    n_layers = len(model.act_fun)

    for layer_id in range(n_layers):
        numeric_layer = model.act_fun[layer_id]
        symbolic_layer = model.symbolic_fun[layer_id]
        edge_scores = getattr(model, "edge_scores", [])
        layer_scores = edge_scores[layer_id] if layer_id < len(edge_scores) else None

        for source_id in range(model.width_in[layer_id]):
            for target_id in range(model.width_out[layer_id + 1]):
                numeric_mask = float(numeric_layer.mask[source_id][target_id].detach().cpu())
                symbolic_mask = float(symbolic_layer.mask[target_id][source_id].detach().cpu())
                if numeric_mask <= 0 and symbolic_mask <= 0:
                    continue

                if layer_id == 0 and source_id < len(input_names):
                    source_name = input_names[source_id]
                else:
                    source_name = f"layer_{layer_id}_node_{source_id}"

                if layer_id == n_layers - 1 and target_id < len(output_names):
                    target_name = output_names[target_id]
                else:
                    target_name = f"layer_{layer_id + 1}_node_{target_id}"

                if numeric_mask > 0 and symbolic_mask > 0:
                    function_type = "symbolic + numeric spline"
                elif symbolic_mask > 0:
                    function_type = "symbolic"
                else:
                    function_type = "numeric spline"

                edge_score = np.nan
                if layer_scores is not None:
                    edge_score = float(layer_scores[target_id][source_id].detach().cpu())

                rows.append(
                    {
                        "layer": layer_id,
                        "source_node": source_name,
                        "target_node": target_name,
                        "function_type": function_type,
                        "symbolic_function": str(symbolic_layer.funs_name[target_id][source_id]),
                        "numeric_mask": numeric_mask,
                        "symbolic_mask": symbolic_mask,
                        "spline_scale": float(numeric_layer.scale_sp[source_id][target_id].detach().cpu()),
                        "base_scale": float(numeric_layer.scale_base[source_id][target_id].detach().cpu()),
                        "edge_score": edge_score,
                    }
                )

    return pd.DataFrame(rows)


def plot_kan_edge_functions(
    model: KAN,
    sample_input: torch.Tensor,
    output_dir: Path,
    image_name: str,
    table_name: str,
    input_names: list[str],
    output_names: list[str],
    title: str,
) -> tuple[Path, Path]:
    """Save the learned KAN graph and a CSV summary of active edges."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / image_name
    table_path = output_dir / table_name
    assets_dir = output_dir / f"{Path(image_name).stem}_assets"

    if assets_dir.exists():
        shutil.rmtree(assets_dir)

    model.get_act(sample_input)
    model.attribute(plot=False)
    summary = edge_function_summary(model, input_names, output_names)
    summary.to_csv(table_path, index=False)

    model.plot(
        folder=str(assets_dir),
        metric="backward",
        scale=0.65,
        tick=False,
        in_vars=input_names,
        out_vars=output_names,
        title=title,
        varscale=0.55,
    )
    plt.savefig(image_path, bbox_inches="tight", dpi=220)
    plt.close("all")
    return image_path, table_path


def plot_prediction_examples(
    model: KAN,
    model_features: np.ndarray,
    plot_features: np.ndarray,
    labels: np.ndarray,
    test_by_class: dict[int, np.ndarray],
    shape_names: list[str],
    device: torch.device,
    output_file: Path,
    examples_per_class: int,
    seed: int,
    title: str,
    active_classes: list[int] | None = None,
    class_ids: list[int] | None = None,
) -> None:
    """Plot example predictions grouped by class."""
    if plot_features.ndim != 2 or plot_features.shape[1] < 2:
        raise ValueError("Prediction plots require at least two feature columns for visualization.")

    rows_to_plot = class_ids if class_ids is not None else list(range(len(shape_names)))
    rng = np.random.default_rng(seed)
    selected_indices = []

    for class_id in rows_to_plot:
        candidates = test_by_class[class_id]
        sample_size = min(examples_per_class, len(candidates))
        selected_indices.extend(rng.choice(candidates, size=sample_size, replace=False).tolist())

    selected_indices = np.array(selected_indices, dtype=np.int64)
    predictions = predict_classes(
        model,
        model_features,
        selected_indices,
        device,
        active_classes=active_classes,
    )

    rows = len(rows_to_plot)
    columns = max(1, examples_per_class)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    colors = {
        "parabola": "#2f6fbb",
        "ellipse": "#2a9d8f",
        "hyperbola": "#d16666",
        "circle": "#7a5ab8",
    }

    position = 0
    for row, class_id in enumerate(rows_to_plot):
        class_samples = selected_indices[labels[selected_indices] == class_id]

        for col in range(columns):
            ax = axes[row][col]
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.35, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.45, alpha=0.45)
            ax.axvline(0, color="black", linewidth=0.45, alpha=0.45)

            if col < len(class_samples):
                sample_index = int(class_samples[col])
                prediction = int(predictions[position])
                position += 1
                x_coord = float(plot_features[sample_index, 0])
                y_coord = float(plot_features[sample_index, 1])
                true_name = shape_names[class_id]
                pred_name = shape_names[prediction]
                is_correct = prediction == class_id
                marker = "o" if is_correct else "X"
                edge_color = "#222222" if is_correct else "#b00020"
                ax.scatter(
                    [x_coord],
                    [y_coord],
                    s=180,
                    marker=marker,
                    color=colors.get(true_name, "#5b7c99"),
                    edgecolors=edge_color,
                    linewidths=1.5,
                )
                ax.set_title(f"true={true_name}\npred={pred_name}", fontsize=10)
            else:
                ax.axis("off")

            if col == 0:
                ax.set_ylabel(shape_names[class_id], fontsize=11)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def high_loss_examples(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    shape_names: list[str],
    device: torch.device,
    n_examples: int,
    active_classes: list[int] | None = None,
) -> pd.DataFrame:
    """Return the highest-loss examples in a dataframe."""
    if len(indices) == 0:
        return pd.DataFrame(columns=["index", "true_class", "predicted_class", "loss"])

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features[indices], device=device)
        targets = torch.tensor(labels[indices], device=device)
        logits = model(inputs)
        if active_classes is not None:
            active = torch.tensor(active_classes, dtype=torch.long, device=device)
            logits = logits.index_select(dim=1, index=active)
            label_map = torch.full((len(shape_names),), -1, dtype=torch.long, device=device)
            label_map[active] = torch.arange(len(active), device=device)
            mapped_targets = label_map[targets.long()]
            losses = torch.nn.functional.cross_entropy(logits, mapped_targets, reduction="none")
            predicted = active[logits.argmax(dim=1)].cpu().numpy()
        else:
            losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            predicted = logits.argmax(dim=1).cpu().numpy()

    losses_np = losses.detach().cpu().numpy()
    top_k = min(n_examples, len(indices))
    selected_positions = np.argsort(losses_np)[-top_k:][::-1]
    rows = []
    for position in selected_positions:
        sample_index = int(indices[position])
        true_class = int(labels[sample_index])
        pred_class = int(predicted[position])
        rows.append(
            {
                "index": sample_index,
                "true_class": shape_names[true_class],
                "predicted_class": shape_names[pred_class],
                "loss": float(losses_np[position]),
            }
        )
    return pd.DataFrame(rows)


def plot_high_loss_examples(
    high_loss_frame: pd.DataFrame,
    raw_features: np.ndarray,
    output_file: Path,
) -> None:
    """Plot the most difficult examples highlighted by their loss."""
    if high_loss_frame.empty:
        return
    if raw_features.ndim != 2 or raw_features.shape[1] < 2:
        raise ValueError("High-loss plots require at least two feature columns for visualization.")

    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    cmap = plt.cm.get_cmap("Reds")
    losses = high_loss_frame["loss"].to_numpy(dtype=float)
    normalized = (losses - losses.min()) / max(losses.ptp(), 1e-8)

    for row, alpha in zip(high_loss_frame.itertuples(index=False), normalized):
        x_coord = float(raw_features[int(row.index), 0])
        y_coord = float(raw_features[int(row.index), 1])
        ax.scatter(
            x_coord,
            y_coord,
            s=140,
            color=cmap(0.35 + 0.6 * alpha),
            edgecolors="black",
            linewidths=0.7,
        )
        ax.text(
            x_coord + 0.02,
            y_coord + 0.02,
            f"{row.true_class}->{row.predicted_class}\nloss={row.loss:.2f}",
            fontsize=8.5,
        )

    ax.set_title("Highest-loss test examples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.35, alpha=0.3)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)
