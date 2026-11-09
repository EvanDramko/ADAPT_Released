"""Visualize saved ADAPT training histories with a log-scale loss plot.

Usage examples:
    python results_visualizer_log.py newRes/embedTypes/outputsZTable
    python results_visualizer_log.py outputs/adapt_runtime_histories.npz --plot-path loss_log.png
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HISTORY_SUFFIX = "_histories.npz"


def main() -> None:
    args = _parse_args()
    history_path = _resolve_history_path(args.history_path)
    history = _load_history(history_path)

    print(f"Loaded history from: {history_path}")
    _print_metric_table(history.get("validation_metrics", []))

    plot_path = args.plot_path
    if plot_path is None:
        plot_path = history_path.with_name(history_path.stem + "_training_loss_log.png")
    _plot_training_losses_log(history.get("train", []), plot_path)
    print(f"Saved log-scale training loss plot to: {plot_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print saved validation metrics and create a log-scale training loss "
            "plot from an ADAPT *_histories.npz file."
        )
    )
    parser.add_argument(
        "history_path",
        type=Path,
        help="Path to a *_histories.npz file or a run directory containing one.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Where to save the training loss plot. Defaults beside the history file.",
    )
    return parser.parse_args()


def _resolve_history_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"History path does not exist: {path}")

    candidates = sorted(path.glob(f"*{HISTORY_SUFFIX}"))
    if not candidates:
        raise FileNotFoundError(
            f"No '*{HISTORY_SUFFIX}' files found in run directory: {path}"
        )
    if len(candidates) > 1:
        candidate_list = "\n".join(f"  {candidate}" for candidate in candidates)
        raise ValueError(
            "Run directory contains multiple history files. Pass one explicitly:\n"
            f"{candidate_list}"
        )
    return candidates[0]


def _load_history(history_path: Path) -> dict[str, list[dict[str, float]]]:
    payload = np.load(history_path, allow_pickle=False)
    history: dict[str, list[dict[str, float]]] = {}
    for name in payload.files:
        entries = json.loads(str(payload[name].item()))
        if not isinstance(entries, list):
            raise ValueError(
                f"History field '{name}' must contain a JSON list, got {type(entries).__name__}."
            )
        history[name] = entries
    return history


def _print_metric_table(rows: list[dict[str, float]]) -> None:
    if not rows:
        print("No saved validation metric results found.")
        return

    columns = _ordered_columns(rows)
    printable_rows = [
        {column: _format_value(row.get(column, "")) for column in columns}
        for row in rows
    ]
    widths = {
        column: max(
            len(column),
            *(len(row[column]) for row in printable_rows),
        )
        for column in columns
    }

    print()
    print("Saved Evaluation Metrics")
    print(_format_table_row({column: column for column in columns}, columns, widths))
    print(_format_table_row({column: "-" * widths[column] for column in columns}, columns, widths))
    for row in printable_rows:
        print(_format_table_row(row, columns, widths))
    print()


def _ordered_columns(rows: list[dict[str, float]]) -> list[str]:
    columns = sorted({column for row in rows for column in row})
    if "epoch" in columns:
        columns.remove("epoch")
        return ["epoch", *columns]
    return columns


def _format_table_row(
    row: dict[str, str],
    columns: list[str],
    widths: dict[str, int],
) -> str:
    return "  ".join(row[column].rjust(widths[column]) for column in columns)


def _format_value(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def _plot_training_losses_log(
    rows: list[dict[str, float]],
    plot_path: Path,
) -> None:
    if not rows:
        raise ValueError("No training history found; cannot create a loss plot.")

    epochs = [
        float(row.get("epoch", index + 1))
        for index, row in enumerate(rows)
    ]
    loss_names = [
        name
        for name in rows[0]
        if name != "epoch" and all(name in row for row in rows)
    ]
    if not loss_names:
        raise ValueError("Training history contains no plottable loss fields.")

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False
    for name in loss_names:
        values = [float(row[name]) for row in rows]
        if any(value <= 0.0 for value in values):
            print(f"Skipping '{name}' because log-scale plots require positive values.")
            continue
        ax.plot(epochs, values, label=name)
        plotted_any = True

    if not plotted_any:
        raise ValueError("No positive training loss fields found for a log-scale plot.")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss History (Log Scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()

    plot_path = plot_path.expanduser()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
