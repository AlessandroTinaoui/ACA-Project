"""Pruning-first experiment for the conic-section dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kan_models.common.kan_compat import KAN
from kan_models.common.runtime import configure_matplotlib

configure_matplotlib()

from kan_models.models.conic.config import DEFAULT_CONIC_CONFIG_PATH, PruningConfig, load_pruning_config
from kan_models.models.conic.data import load_conic_csv, load_feature_names, make_kan_dataset, standardize_from_train, stratified_split
from kan_models.models.conic.modeling import (
    accuracy,
    build_model,
    compute_confusion_matrix,
    cross_entropy_loss,
    score_hidden_nodes,
    top_scoring_nodes,
)
from kan_models.models.conic.plots.common import (
    high_loss_examples,
    plot_high_loss_examples,
    plot_kan_edge_functions,
    plot_prediction_examples,
)
from kan_models.models.conic.plots.pruning import (
    plot_all_loss_curves,
    plot_architecture_schedule,
    plot_confusion,
    plot_growth_progress,
    plot_loss_curve,
    plot_node_scores,
)
from kan_models.common.runtime import detect_device
from kan_models.common.shared import clear_matching_files, copy_kan_model, hidden_units


DEFAULT_CONFIG_PATH = DEFAULT_CONIC_CONFIG_PATH


def evaluate_model(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray | None,
    device: torch.device,
    stage: str,
    train_steps: int,
    selected_nodes: list[int] | None = None,
    include_test: bool = False,
) -> dict[str, float | int | str]:
    """Collect accuracy, loss, and size information for one experiment stage."""
    hidden = hidden_units(model)
    row = {
        "stage": stage,
        "hidden_units": hidden,
        "train_steps": train_steps,
        "cost_proxy": hidden * train_steps,
        "selected_nodes": "" if selected_nodes is None else " ".join(map(str, selected_nodes)),
        "train_accuracy": accuracy(model, features, labels, train_indices, device),
        "train_loss": cross_entropy_loss(model, features, labels, train_indices, device),
        "test_accuracy": np.nan,
        "test_loss": np.nan,
    }

    if include_test and test_indices is not None:
        row["test_accuracy"] = accuracy(model, features, labels, test_indices, device)
        row["test_loss"] = cross_entropy_loss(model, features, labels, test_indices, device)
    return row


def train_phase(
    model: KAN,
    dataset: dict[str, torch.Tensor],
    steps: int,
    loss_fn: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    update_grid: bool = True,
) -> dict[str, list]:
    """Train one KAN phase with the shared pruning configuration."""
    return model.fit(
        dataset,
        opt=optimizer_name,
        steps=steps,
        lr=learning_rate,
        update_grid=update_grid,
        loss_fn=loss_fn,
        display_metrics=["train_loss"],
    )


def run_pruning(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    """Run the full probe-based pruning experiment and save its outputs."""
    config = load_pruning_config(config_path)
    torch.manual_seed(config.data.seed)
    device = detect_device(config.model.device)
    output_dir = config.output.plot_dir
    if output_dir is None:
        raise ValueError("output.plot_dir is required for the pruning experiment.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.plots.clear_old_plots:
        clear_matching_files(output_dir, "*.png", "*.txt", "*_assets")

    input_names = load_feature_names(config.data.csv_path, config.data.target_column)
    raw_features, labels, shape_names = load_conic_csv(config.data.csv_path, config.data.target_column)
    train_indices, test_indices, _, test_by_class = stratified_split(
        labels=labels,
        test_ratio=config.data.test_ratio,
        seed=config.data.seed,
    )

    if config.training.standardize:
        features, _, _ = standardize_from_train(raw_features, train_indices)
    else:
        features = raw_features

    dataset = make_kan_dataset(features, labels, train_indices, train_indices, device)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Probe model: hidden={config.training.probe_hidden}, steps={config.training.probe_steps}")

    probe_model = build_model(
        input_dim=features.shape[1],
        output_dim=len(shape_names),
        config=config.model,
        seed=config.data.seed,
        device=device,
        hidden=config.training.probe_hidden,
    )

    probe_results = train_phase(
        probe_model,
        dataset,
        config.training.probe_steps,
        loss_fn,
        config.training.optimizer,
        config.training.learning_rate,
    )
    loss_runs = {"probe": probe_results}
    plot_loss_curve(probe_results, output_dir, "01_probe_loss.png", "Short probe loss before selecting nodes")

    node_scores = score_hidden_nodes(probe_model, dataset["train_input"])
    first_selected_nodes = top_scoring_nodes(node_scores, config.training.start_hidden)
    plot_node_scores(node_scores, first_selected_nodes, output_dir / "02_probe_node_importance.png")

    records = [
        evaluate_model(
            probe_model,
            features,
            labels,
            train_indices,
            None,
            device,
            stage="probe",
            train_steps=config.training.probe_steps,
        )
    ]

    current_model = probe_model
    final_hidden = config.training.keep_hidden_schedule[-1]

    for target_hidden in config.training.keep_hidden_schedule:
        selected_nodes = top_scoring_nodes(node_scores, target_hidden)
        print(f"Training pruned model with top {target_hidden} probe nodes: {selected_nodes}")

        probe_source = copy_kan_model(probe_model)
        probe_source.get_act(dataset["train_input"])
        current_model = probe_source.prune_node(active_neurons_id=[selected_nodes], log_history=False)
        current_model.auto_save = False
        current_model.get_act(dataset["train_input"])

        stage_name = f"keep_{target_hidden}"
        stage_results = train_phase(
            current_model,
            dataset,
            config.training.train_steps_per_model,
            loss_fn,
            config.training.optimizer,
            config.training.learning_rate,
            update_grid=True,
        )
        loss_runs[stage_name] = stage_results
        plot_loss_curve(stage_results, output_dir, f"loss_keep_{target_hidden:02d}.png", f"Training loss with top {target_hidden} probe nodes")

        records.append(
            evaluate_model(
                current_model,
                features,
                labels,
                train_indices,
                test_indices,
                device,
                stage=stage_name,
                train_steps=config.training.train_steps_per_model,
                selected_nodes=selected_nodes,
                include_test=target_hidden == final_hidden,
            )
        )

    metrics_frame = pd.DataFrame(records)
    if config.output.metrics_path is not None:
        metrics_frame.to_csv(config.output.metrics_path, index=False)

    plot_all_loss_curves(loss_runs, output_dir / "03_all_training_losses.png")
    plot_growth_progress(metrics_frame, output_dir / "04_prune_first_progress.png")
    plot_architecture_schedule(metrics_frame, output_dir / "05_architecture_schedule.png")
    plot_confusion(
        compute_confusion_matrix(current_model, features, labels, test_indices, device, list(range(len(shape_names)))),
        shape_names,
        "Confusion matrix after final growth",
        output_dir / "06_confusion_after_final_growth.png",
    )
    plot_prediction_examples(
        current_model,
        features,
        raw_features,
        labels,
        test_by_class,
        shape_names,
        device,
        output_dir / "07_predictions_after_final_growth.png",
        config.plots.prediction_examples,
        config.data.seed,
        "Predictions after final growth",
    )

    high_loss_frame = high_loss_examples(
        current_model,
        features,
        labels,
        test_indices,
        shape_names,
        device,
        config.plots.high_loss_examples,
    )
    if config.output.high_loss_path is not None:
        high_loss_frame.to_csv(config.output.high_loss_path, index=False)
    plot_high_loss_examples(high_loss_frame, raw_features, output_dir / "08_highest_loss_examples.png")

    edge_image_path = None
    edge_table_path = None
    if config.plots.plot_edge_functions:
        edge_image_path, edge_table_path = plot_kan_edge_functions(
            model=current_model,
            sample_input=dataset["train_input"],
            output_dir=output_dir,
            image_name="09_final_edge_functions.png",
            table_name="09_final_edge_functions.csv",
            input_names=input_names,
            output_names=shape_names,
            title="Final pruned KAN edge functions",
        )

    first = metrics_frame.iloc[0]
    final = metrics_frame.iloc[-1]
    print(f"Probe: train accuracy={first['train_accuracy']:.3f}, hidden={int(first['hidden_units'])}")
    print(
        f"Final pruned-first model: test accuracy={final['test_accuracy']:.3f}, "
        f"test loss={final['test_loss']:.3f}, hidden={int(final['hidden_units'])}"
    )
    print(f"Plots saved in: {output_dir}")
    if config.output.metrics_path is not None:
        print(f"Metrics saved in: {config.output.metrics_path}")
    if config.output.high_loss_path is not None:
        print(f"High-loss examples saved in: {config.output.high_loss_path}")
    if edge_image_path is not None and edge_table_path is not None:
        print(f"Final edge functions saved in: {edge_image_path}")
        print(f"Active edge table saved in: {edge_table_path}")

    return metrics_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pruning-first conic KAN experiment.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the TOML config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args(argv)
    run_pruning(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
