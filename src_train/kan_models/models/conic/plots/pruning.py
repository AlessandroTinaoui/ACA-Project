"""Plotting helpers specific to the conic pruning experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_curve(results: dict[str, list], output_dir: Path, filename: str, title: str) -> None:
    """Save the training loss curve for a single pruning stage."""
    train_loss = np.asarray(results["train_loss"], dtype=float)
    steps = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(steps, train_loss, label="train", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Training loss returned by KAN.fit")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output_dir / filename, dpi=170)
    plt.close(fig)


def plot_all_loss_curves(loss_runs: dict[str, dict[str, list]], output_file: Path) -> None:
    """Save all pruning-stage training losses in one comparison plot."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(loss_runs))))

    for color, (name, results) in zip(colors, loss_runs.items()):
        train_loss = np.asarray(results["train_loss"], dtype=float)
        steps = np.arange(1, len(train_loss) + 1)
        ax.plot(steps, train_loss, marker="o", markersize=3.5, linewidth=2, label=name, color=color)

    ax.set_title("Training losses for all pruning stages")
    ax.set_xlabel("Optimization step inside each stage")
    ax.set_ylabel("Training loss returned by KAN.fit")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def plot_node_scores(scores: np.ndarray, selected_nodes: list[int], output_file: Path) -> None:
    """Plot probe node importance and highlight the first nodes kept."""
    selected = set(selected_nodes)
    colors = ["#2a9d8f" if idx in selected else "#b7b7b7" for idx in range(len(scores))]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(np.arange(len(scores)), scores, color=colors)
    ax.set_title("Probe node importance")
    ax.set_xlabel("Hidden node id in the probe model")
    ax.set_ylabel("Attribution score")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.text(
        0.5,
        0.93,
        f"kept nodes: {selected_nodes}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def plot_growth_progress(metrics_frame: pd.DataFrame, output_file: Path) -> None:
    """Plot how accuracy, loss, model size, and cost change across stages."""
    labels = metrics_frame["stage"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    axes[0, 0].plot(x, metrics_frame["train_accuracy"], marker="o", linewidth=2)
    axes[0, 0].set_title("Train accuracy")
    axes[0, 0].set_ylim(0, 1)

    final_test = metrics_frame.dropna(subset=["test_accuracy", "test_loss"])
    if not final_test.empty:
        final_row = final_test.iloc[-1]
        final_index = int(final_test.index[-1])
        axes[0, 0].scatter(final_index, final_row["test_accuracy"], s=90, marker="D", color="#b85042", label="final test", zorder=3)
        axes[0, 0].legend()

    axes[0, 1].plot(x, metrics_frame["train_loss"], marker="o", linewidth=2, color="#2a9d8f")
    axes[0, 1].set_title("Train loss")
    if not final_test.empty:
        final_row = final_test.iloc[-1]
        final_index = int(final_test.index[-1])
        axes[0, 1].scatter(final_index, final_row["test_loss"], s=90, marker="D", color="#b85042", label="final test", zorder=3)
        axes[0, 1].legend()

    axes[1, 0].bar(x, metrics_frame["hidden_units"], color="#5b7c99")
    axes[1, 0].set_title("Hidden units used")

    cumulative_cost = metrics_frame["cost_proxy"].cumsum()
    axes[1, 1].plot(x, cumulative_cost, marker="o", linewidth=2, color="#7a5ab8")
    axes[1, 1].set_title("Cumulative cost proxy")
    axes[1, 1].set_ylabel("hidden units x training steps")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)

    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def plot_architecture_schedule(metrics_frame: pd.DataFrame, output_file: Path) -> None:
    """Show the hidden-unit schedule used during the pruning experiment."""
    labels = metrics_frame["stage"].tolist()
    hidden = metrics_frame["hidden_units"].to_numpy()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.step(x, hidden, where="mid", linewidth=2.5, color="#2f6fbb")
    ax.scatter(x, hidden, s=70, color="#2f6fbb")
    ax.set_title("Architecture schedule")
    ax.set_xlabel("Experiment stage")
    ax.set_ylabel("Hidden units")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    for xpos, value in zip(x, hidden):
        ax.text(xpos, value + 0.25, str(int(value)), ha="center", fontsize=10)

    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, shape_names: list[str], title: str, output_file: Path) -> None:
    """Save a labelled confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, pad=14)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(shape_names)))
    ax.set_yticks(np.arange(len(shape_names)))
    ax.set_xticklabels(shape_names, rotation=20, ha="right")
    ax.set_yticklabels(shape_names)

    max_value = max(int(matrix.max()), 1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if value > max_value / 2 else "black",
            )

    fig.savefig(output_file, dpi=170)
    plt.close(fig)
