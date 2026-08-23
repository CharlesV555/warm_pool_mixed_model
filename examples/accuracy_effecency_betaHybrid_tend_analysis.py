from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


RUN_METADATA_NAME = "run_metadata.json"
ANALYSIS_DIR_NAME = "accuracy_effecency_betaHybrid_tend_analysis"


def analyze_tend_gradient(
    batch_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    reached_only: bool = False,
) -> dict[str, Any]:
    batch_path = Path(batch_dir)
    metadata_path = batch_path / RUN_METADATA_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {RUN_METADATA_NAME}: {metadata_path}")

    out_dir = Path(output_dir) if output_dir is not None else batch_path / ANALYSIS_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = normalize_rows(payload.get("records", []), reached_only=bool(reached_only))
    sample_csv = out_dir / "t_end_wall_time_samples.csv"
    write_csv(sample_csv, rows)
    plot_path = plot_wall_time_by_t_end(
        rows,
        out_dir / "t_end_wall_time_box_scatter_horizontal.png",
    )

    summary = {
        "batch_dir": str(batch_path),
        "metadata_path": str(metadata_path),
        "output_dir": str(out_dir),
        "reached_only": bool(reached_only),
        "n_input_records": len(payload.get("records", [])),
        "n_plotted_records": len(rows),
        "sample_csv": str(sample_csv),
        "wall_time_plot": str(plot_path) if plot_path is not None else None,
    }
    summary_path = out_dir / "tend_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[tend-analysis] wrote sample CSV: {sample_csv}")
    print(f"[tend-analysis] wrote wall-time plot: {plot_path}")
    print(f"[tend-analysis] wrote summary: {summary_path}")
    return summary


def normalize_rows(records: list[dict[str, Any]], *, reached_only: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        requested_t_end = optional_float(record.get("requested_t_end"))
        wall_seconds = optional_float(record.get("wall_runtime_seconds"))
        if requested_t_end is None or wall_seconds is None:
            continue
        if bool(reached_only) and not bool(record.get("reached_requested_t_end", False)):
            continue
        output.append(
            {
                "network": str(record.get("network", "")),
                "t_end": float(requested_t_end),
                "algorithm_label": algorithm_label(record),
                "method": str(record.get("method", "")),
                "run_index": record.get("run_index"),
                "seed": record.get("seed"),
                "wall_runtime_seconds": float(wall_seconds),
                "simulation_final_time": optional_float(record.get("simulation_final_time")),
                "reached_requested_t_end": bool(record.get("reached_requested_t_end", False)),
                "stop_reason": record.get("stop_reason"),
            }
        )
    return output


def plot_wall_time_by_t_end(rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    if not rows:
        return None
    networks = sorted({str(row["network"]) for row in rows})
    t_ends_by_network = {
        network: sorted({float(row["t_end"]) for row in rows if str(row["network"]) == network})
        for network in networks
    }
    labels = sorted({str(row["algorithm_label"]) for row in rows}, key=algorithm_sort_key)
    cmap = plt.get_cmap("tab10")

    positions: list[float] = []
    data: list[np.ndarray] = []
    reasons_by_group: list[list[str]] = []
    colors: list[Any] = []
    y_tick_positions: list[float] = []
    y_tick_labels: list[str] = []
    network_label_positions: list[tuple[float, str]] = []

    current_y = 0.0
    algorithm_gap = 0.28
    t_end_gap = 0.95
    network_gap = 1.25
    box_height = 0.22

    for network in networks:
        network_start = current_y
        for t_end in t_ends_by_network[network]:
            center = current_y
            y_tick_positions.append(center)
            y_tick_labels.append(f"t={t_end:g}")
            offsets = (np.arange(len(labels), dtype=float) - (len(labels) - 1) / 2.0) * algorithm_gap
            for label_index, label in enumerate(labels):
                group_rows = [
                    row
                    for row in rows
                    if str(row["network"]) == network
                    and float(row["t_end"]) == float(t_end)
                    and str(row["algorithm_label"]) == label
                    and np.isfinite(float(row["wall_runtime_seconds"]))
                ]
                values = [float(row["wall_runtime_seconds"]) for row in group_rows]
                if not values:
                    continue
                positions.append(center + float(offsets[label_index]))
                data.append(np.asarray(values, dtype=float))
                reasons_by_group.append([str(row.get("stop_reason", "unknown")) for row in group_rows])
                colors.append(cmap(label_index % 10))
            current_y += t_end_gap
        network_end = current_y - t_end_gap if t_ends_by_network[network] else network_start
        network_label_positions.append(((network_start + network_end) / 2.0, network))
        current_y += network_gap

    fig_height = max(6.0, 0.55 * max(len(y_tick_positions), 1) + 1.2 * len(networks))
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    if data:
        box = ax.boxplot(
            data,
            positions=positions,
            vert=False,
            widths=box_height,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#555555", "linewidth": 0.9},
            capprops={"color": "#555555", "linewidth": 0.9},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.24)
        for index, (pos, values, color, reasons) in enumerate(zip(positions, data, colors, reasons_by_group)):
            rng = np.random.default_rng(2000 + int(index))
            jitter = rng.uniform(-box_height * 0.32, box_height * 0.32, size=values.shape)
            reason_arr = np.asarray(reasons, dtype=object)
            for reason in sorted(set(reasons), key=stop_reason_sort_key):
                mask = reason_arr == reason
                marker = stop_reason_marker(reason)
                ax.scatter(
                    values[mask],
                    np.full(int(np.count_nonzero(mask)), pos, dtype=float) + jitter[mask],
                    s=22 if marker == "^" else 16,
                    marker=marker,
                    color=color,
                    alpha=0.66,
                    edgecolors="none",
                    zorder=3,
                )

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=cmap(index % 10), edgecolor=cmap(index % 10), alpha=0.28, label=short_label(label))
        for index, label in enumerate(labels)
    ]
    ax.set_xlabel("wall runtime seconds")
    ax.set_ylabel("network / target t_end")
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    algorithm_legend = ax.legend(handles=handles, fontsize=8, loc="upper right", title="algorithm")
    ax.add_artist(algorithm_legend)
    reason_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=stop_reason_marker(reason),
            color="none",
            markerfacecolor="#555555",
            markersize=6,
            label=short_label(reason, max_len=28),
        )
        for reason in sorted({str(row.get("stop_reason", "unknown")) for row in rows}, key=stop_reason_sort_key)
    ]
    ax.legend(handles=reason_handles, fontsize=8, loc="lower right", title="stop reason")
    ax.set_title("Wall Time To Reach Target Simulation Time")

    x_left, x_right = ax.get_xlim()
    label_x = x_left - 0.02 * max(x_right - x_left, 1.0)
    for y_value, network in network_label_positions:
        ax.text(
            label_x,
            y_value,
            str(network),
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )
    fig.subplots_adjust(left=0.22)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def algorithm_label(record: dict[str, Any]) -> str:
    if record.get("algorithm_label"):
        return str(record["algorithm_label"])
    parameter = record.get("parameter_case")
    if isinstance(parameter, dict) and parameter.get("label"):
        return str(parameter["label"])
    return str(record.get("method", "unknown"))


def algorithm_sort_key(label: str) -> tuple[int, str]:
    text = str(label).lower()
    return (1 if text == "ssa" else 0, text)


def stop_reason_marker(reason: str) -> str:
    text = str(reason).lower()
    if text in {"max_runtime_seconds", "max_run_time", "max_runtime", "runtime_limit"}:
        return "^"
    if text == "reached_t_end":
        return "o"
    if text == "max_steps":
        return "s"
    if "exception" in text or "error" in text:
        return "x"
    return "D"


def stop_reason_sort_key(reason: str) -> tuple[int, str]:
    order = {
        "reached_t_end": 0,
        "max_runtime_seconds": 1,
        "max_steps": 2,
    }
    text = str(reason)
    return (order.get(text, 99), text)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def short_label(value: str, max_len: int = 40) -> str:
    text = str(value)
    return text if len(text) <= int(max_len) else text[: int(max_len) - 3] + "..."


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Analyze accuracy_effecency_betaHybrid_tend_gradient output."
    )
    arg_parser.add_argument("batch_dir", help="Directory containing run_metadata.json")
    arg_parser.add_argument("--output-dir", default=None)
    arg_parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Deprecated; failed runs are included by default.",
    )
    arg_parser.add_argument(
        "--reached-only",
        action="store_true",
        help="Only plot runs that reached requested_t_end.",
    )
    return arg_parser


def main() -> None:
    args = parser().parse_args()
    analyze_tend_gradient(
        args.batch_dir,
        output_dir=args.output_dir,
        reached_only=bool(args.reached_only),
    )


if __name__ == "__main__":
    main()
