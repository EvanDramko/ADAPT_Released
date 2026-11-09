"""Visualize saved ADAPT training histories.

Usage examples:
    python results_visualizer.py newRes/embedTypes/outputsZTable
    python results_visualizer.py outputs/adapt_runtime_histories.npz --plot-path loss.png
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
    _print_evaluation_table(
        history.get("train", []),
        history.get("validation", []),
        history.get("validation_metrics", []),
        history.get("test", []),
        history.get("test_metrics", []),
    )

    plot_path = args.plot_path
    if plot_path is None:
        plot_path = history_path.with_name(history_path.stem + "_losses.png")
    _plot_losses(
        history.get("train", []),
        history.get("validation", []),
        history.get("test", []),
        plot_path,
    )
    print(f"Saved loss plot to: {plot_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print saved validation metrics and create a training loss plot "
            "from an ADAPT *_histories.npz file."
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


def _print_evaluation_table(
    train_rows: list[dict[str, float]],
    validation_rows: list[dict[str, float]],
    validation_metric_rows: list[dict[str, float]],
    test_rows: list[dict[str, float]],
    test_metric_rows: list[dict[str, float]],
) -> None:
    rows = _combine_evaluation_results(
        train_rows,
        validation_rows,
        validation_metric_rows,
        test_rows,
        test_metric_rows,
    )
    if not rows:
        print("No saved evaluation results found.")
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
    print("Saved Evaluation Results")
    print(_format_table_row({column: column for column in columns}, columns, widths))
    print(_format_table_row({column: "-" * widths[column] for column in columns}, columns, widths))
    for row in printable_rows:
        print(_format_table_row(row, columns, widths))
    print()


def _combine_evaluation_results(
    train_rows: list[dict[str, float]],
    validation_rows: list[dict[str, float]],
    validation_metric_rows: list[dict[str, float]],
    test_rows: list[dict[str, float]],
    test_metric_rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    train_by_epoch = _index_by_epoch(train_rows)
    validation_by_epoch = _index_by_epoch(validation_rows)
    validation_metrics_by_epoch = _index_by_epoch(validation_metric_rows)
    test_by_epoch = _index_by_epoch(test_rows)
    test_metrics_by_epoch = _index_by_epoch(test_metric_rows)

    epochs = sorted(
        set(validation_by_epoch)
        | set(validation_metrics_by_epoch)
        | set(test_by_epoch)
        | set(test_metrics_by_epoch)
    )

    combined_rows: list[dict[str, float]] = []
    for epoch in epochs:
        row: dict[str, float] = {"epoch": epoch}

        _add_prefixed_values(row, train_by_epoch.get(epoch), "train")
        _add_prefixed_values(row, validation_by_epoch.get(epoch), "val")
        _add_prefixed_values(row, validation_metrics_by_epoch.get(epoch), "val")
        _add_prefixed_values(row, test_by_epoch.get(epoch), "test")
        _add_prefixed_values(row, test_metrics_by_epoch.get(epoch), "test")

        combined_rows.append(row)

    return combined_rows


def _index_by_epoch(rows: list[dict[str, float]]) -> dict[float, dict[str, float]]:
    return {
        float(row["epoch"]): row
        for row in rows
        if "epoch" in row
    }


def _add_prefixed_values(
    destination: dict[str, float],
    row: dict[str, float] | None,
    prefix: str,
) -> None:
    if row is None:
        return
    for name, value in row.items():
        if name != "epoch":
            destination[f"{prefix}_{name}"] = value


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


def _plot_losses(
    train_rows: list[dict[str, float]],
    validation_rows: list[dict[str, float]],
    test_rows: list[dict[str, float]],
    plot_path: Path,
) -> None:
    if not train_rows:
        raise ValueError("No training history found; cannot create a loss plot.")

    series = []
    series.extend(_loss_series(train_rows, "train"))
    series.extend(_loss_series(validation_rows, "val"))
    series.extend(_loss_series(test_rows, "test"))
    if not series:
        raise ValueError("Training history contains no plottable loss fields.")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, epochs, values in series:
        ax.plot(epochs, values, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = plot_path.expanduser()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def _loss_series(
    rows: list[dict[str, float]],
    prefix: str,
) -> list[tuple[str, list[float], list[float]]]:
    if not rows:
        return []

    epochs = [
        float(row.get("epoch", index + 1))
        for index, row in enumerate(rows)
    ]
    loss_names = [
        name
        for name in rows[0]
        if name != "epoch" and all(name in row for row in rows)
    ]
    return [
        (
            f"{prefix}_{name}",
            epochs,
            [float(row[name]) for row in rows],
        )
        for name in loss_names
    ]


if __name__ == "__main__":
    main()
