from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (  # noqa: E402
    has_trajectory_sidecar,
    sample_trajectory_states_from_path,
    trajectory_sidecar_dir,
    trajectory_storage_exists,
)


RUN_METADATA_NAME = "run_metadata.json"
ANALYSIS_DIR_NAME = "accuracy_effecency_betaHybrid_analysis"
N_TIME_POINTS = 100
SLICED_WASSERSTEIN_PROJECTIONS = 128
SLICED_WASSERSTEIN_SEED = 20260820


@dataclass(slots=True)
class RunRow:
    raw: dict[str, Any]
    network: str
    method: str
    algorithm_label: str
    trajectory_path: Path | None
    requested_t_end: float | None
    simulation_final_time: float | None
    wall_runtime_seconds: float | None
    status: str


def analyze_batch_output(
    batch_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    n_time_points: int = N_TIME_POINTS,
    n_projections: int = SLICED_WASSERSTEIN_PROJECTIONS,
    random_seed: int = SLICED_WASSERSTEIN_SEED,
) -> dict[str, Any]:
    batch_path = Path(batch_dir)
    metadata_path = batch_path / RUN_METADATA_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {RUN_METADATA_NAME}: {metadata_path}")

    out_dir = Path(output_dir) if output_dir is not None else batch_path / ANALYSIS_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [normalize_run_row(record, metadata_path) for record in payload.get("records", [])]
    ok_rows = [row for row in rows if row.status == "ok" and row.trajectory_path is not None]

    wall_rows = wall_time_rows(ok_rows)
    wall_csv = out_dir / "wall_time_to_common_t_end.csv"
    write_csv(wall_csv, wall_rows)
    wall_sample_rows = wall_time_sample_rows(ok_rows, wall_rows)
    wall_sample_csv = out_dir / "wall_time_to_common_t_end_samples.csv"
    write_csv(wall_sample_csv, wall_sample_rows)
    wall_plot = plot_wall_time_box_scatter(
        wall_sample_rows,
        out_dir / "wall_time_to_common_t_end_box_scatter.png",
    )

    distribution_rows, moment_npz_paths = distribution_comparison_rows(
        ok_rows,
        out_dir=out_dir,
        n_time_points=int(n_time_points),
        n_projections=int(n_projections),
        random_seed=int(random_seed),
    )
    distribution_csv = out_dir / "state_distribution_vs_ssa.csv"
    write_csv(distribution_csv, distribution_rows)
    swd_plots = plot_swd_lines_by_network(distribution_rows, out_dir)
    final_total_rows = final_total_vs_time_rows([row for row in rows if row.status == "ok"])
    final_total_csv = out_dir / "final_total_abundance_vs_simulation_time.csv"
    write_csv(final_total_csv, final_total_rows)
    final_total_plots = plot_final_total_vs_time_by_network(final_total_rows, out_dir)

    summary = {
        "batch_dir": str(batch_path),
        "metadata_path": str(metadata_path),
        "output_dir": str(out_dir),
        "n_input_records": len(rows),
        "n_ok_trajectory_records": len(ok_rows),
        "wall_time_csv": str(wall_csv),
        "wall_time_sample_csv": str(wall_sample_csv),
        "wall_time_plot": str(wall_plot) if wall_plot is not None else None,
        "distribution_csv": str(distribution_csv),
        "sliced_wasserstein_plots": [str(path) for path in swd_plots],
        "final_total_abundance_csv": str(final_total_csv),
        "final_total_abundance_plots": [str(path) for path in final_total_plots],
        "moment_npz_paths": [str(path) for path in moment_npz_paths],
        "parameters": {
            "n_time_points": int(n_time_points),
            "n_projections": int(n_projections),
            "random_seed": int(random_seed),
        },
    }
    summary_path = out_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[analysis] wrote wall-time CSV: {wall_csv}")
    print(f"[analysis] wrote wall-time sample CSV: {wall_sample_csv}")
    print(f"[analysis] wrote wall-time plot: {wall_plot}")
    print(f"[analysis] wrote distribution CSV: {distribution_csv}")
    for path in swd_plots:
        print(f"[analysis] wrote network accuracy plot: {path}")
    print(f"[analysis] wrote final-total CSV: {final_total_csv}")
    for path in final_total_plots:
        print(f"[analysis] wrote final-total scatter plot: {path}")
    print(f"[analysis] wrote summary: {summary_path}")
    return summary


def normalize_run_row(record: dict[str, Any], metadata_path: Path) -> RunRow:
    method = str(record.get("method", "")).lower()
    parameter = record.get("parameter_case")
    if method == "ssa":
        label = "ssa"
    elif isinstance(parameter, dict) and parameter.get("label"):
        label = str(parameter["label"])
    else:
        label = method or "unknown"

    trajectory_path = resolve_path(record.get("trajectory_path"), metadata_path)
    return RunRow(
        raw=dict(record),
        network=str(record.get("network", "")),
        method=method,
        algorithm_label=label,
        trajectory_path=trajectory_path,
        requested_t_end=optional_float(record.get("requested_t_end")),
        simulation_final_time=optional_float(record.get("simulation_final_time")),
        wall_runtime_seconds=optional_float(record.get("wall_runtime_seconds")),
        status=str(record.get("status", "")),
    )


def wall_time_rows(rows: list[RunRow]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for network, network_rows in group_by(rows, lambda row: row.network).items():
        ssa_rows = [row for row in network_rows if row.algorithm_label == "ssa"]
        if not ssa_rows:
            continue
        common_t_end = median(
            [
                row.simulation_final_time
                for row in ssa_rows
                if row.simulation_final_time is not None and np.isfinite(row.simulation_final_time)
            ]
        )
        if common_t_end is None:
            common_t_end = median(
                [
                    row.requested_t_end
                    for row in network_rows
                    if row.requested_t_end is not None and np.isfinite(row.requested_t_end)
                ]
            )
        if common_t_end is None:
            continue
        for label, label_rows in group_by(network_rows, lambda row: row.algorithm_label).items():
            reached = [
                row
                for row in label_rows
                if row.wall_runtime_seconds is not None
                and row.simulation_final_time is not None
                and row.simulation_final_time >= float(common_t_end) - 1e-12
            ]
            wall_values = [float(row.wall_runtime_seconds) for row in reached]
            output.append(
                {
                    "network": network,
                    "algorithm_label": label,
                    "common_t_end": float(common_t_end),
                    "n_runs": len(label_rows),
                    "n_reached_common_t_end": len(reached),
                    "wall_seconds_mean": float(np.mean(wall_values)) if wall_values else float("nan"),
                    "wall_seconds_median": float(np.median(wall_values)) if wall_values else float("nan"),
                    "wall_seconds_std": float(np.std(wall_values)) if wall_values else float("nan"),
                    "wall_seconds_min": float(np.min(wall_values)) if wall_values else float("nan"),
                    "wall_seconds_max": float(np.max(wall_values)) if wall_values else float("nan"),
                }
            )
    return output


def wall_time_sample_rows(
    rows: list[RunRow],
    wall_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    common_by_network: dict[str, float] = {}
    for row in wall_rows:
        network = str(row["network"])
        common_t_end = optional_float(row.get("common_t_end"))
        if common_t_end is not None:
            common_by_network.setdefault(network, common_t_end)

    output: list[dict[str, Any]] = []
    for row in rows:
        common_t_end = common_by_network.get(row.network)
        if common_t_end is None:
            continue
        if row.wall_runtime_seconds is None or row.simulation_final_time is None:
            continue
        if row.simulation_final_time < common_t_end - 1e-12:
            continue
        output.append(
            {
                "network": row.network,
                "algorithm_label": row.algorithm_label,
                "method": row.method,
                "run_index": row.raw.get("run_index"),
                "seed": row.raw.get("seed"),
                "common_t_end": float(common_t_end),
                "simulation_final_time": float(row.simulation_final_time),
                "wall_runtime_seconds": float(row.wall_runtime_seconds),
            }
        )
    return output


def final_total_vs_time_rows(rows: list[RunRow]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        final_time = row.simulation_final_time
        total = optional_float(row.raw.get("final_total_abundance"))
        if final_time is None or total is None:
            continue
        output.append(
            {
                "network": row.network,
                "algorithm_label": row.algorithm_label,
                "method": row.method,
                "run_index": row.raw.get("run_index"),
                "seed": row.raw.get("seed"),
                "simulation_final_time": float(final_time),
                "final_total_abundance": float(total),
                "wall_runtime_seconds": row.wall_runtime_seconds,
                "stop_reason": row.raw.get("stop_reason"),
            }
        )
    return output


def distribution_comparison_rows(
    rows: list[RunRow],
    *,
    out_dir: Path,
    n_time_points: int,
    n_projections: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    rng = np.random.default_rng(int(random_seed))
    output: list[dict[str, Any]] = []
    moment_paths: list[Path] = []
    for network, network_rows in group_by(rows, lambda row: row.network).items():
        network_started = perf_counter()
        groups = group_by(network_rows, lambda row: row.algorithm_label)
        ssa_rows = groups.get("ssa", [])
        if not ssa_rows:
            print(f"[analysis] network={network} skipped: no SSA rows")
            continue
        common_t_end = median(
            [
                row.simulation_final_time
                for row in ssa_rows
                if row.simulation_final_time is not None and np.isfinite(row.simulation_final_time)
            ]
        )
        if common_t_end is None or common_t_end <= 0.0:
            print(f"[analysis] network={network} skipped: invalid common_t_end={common_t_end}")
            continue
        time_points = np.linspace(0.0, float(common_t_end), int(n_time_points))
        print(
            f"[analysis] network={network} common_t_end={common_t_end:.6g} "
            f"time_points={len(time_points)} labels={len(groups)}"
        )

        load_started = perf_counter()
        ssa_samples, species_names = load_group_samples(ssa_rows, time_points)
        print(
            f"[analysis] network={network} label=ssa "
            f"load+sample={perf_counter() - load_started:.3f}s "
            f"samples_shape={ssa_samples.shape}"
        )
        if ssa_samples.size == 0:
            print(f"[analysis] network={network} skipped: no usable SSA samples")
            continue
        moment_started = perf_counter()
        ssa_first = np.nanmean(ssa_samples, axis=0)
        ssa_second = np.nanmean(np.square(ssa_samples), axis=0)
        print(
            f"[analysis] network={network} label=ssa "
            f"moments={perf_counter() - moment_started:.3f}s"
        )

        network_moment_path = out_dir / f"{safe_name(network)}_moments.npz"
        moment_payload: dict[str, np.ndarray] = {
            "time_points": time_points,
            "species_names": np.asarray(species_names, dtype=object),
            "ssa_first_moment": ssa_first,
            "ssa_second_raw_moment": ssa_second,
        }

        for label, label_rows in groups.items():
            if label == "ssa":
                continue
            label_started = perf_counter()
            load_started = perf_counter()
            samples, _ = load_group_samples(label_rows, time_points)
            print(
                f"[analysis] network={network} label={label} "
                f"load+sample={perf_counter() - load_started:.3f}s "
                f"samples_shape={samples.shape}"
            )
            if samples.size == 0:
                print(f"[analysis] network={network} label={label} skipped: no usable samples")
                continue
            moment_started = perf_counter()
            first = np.nanmean(samples, axis=0)
            second = np.nanmean(np.square(samples), axis=0)
            moment_payload[f"{safe_name(label)}_first_moment"] = first
            moment_payload[f"{safe_name(label)}_second_raw_moment"] = second
            print(
                f"[analysis] network={network} label={label} "
                f"moments={perf_counter() - moment_started:.3f}s"
            )

            swd_started = perf_counter()
            swd_values = sliced_wasserstein_by_time(
                ssa_samples,
                samples,
                n_projections=int(n_projections),
                rng=rng,
            )
            print(
                f"[analysis] network={network} label={label} "
                f"sliced_wasserstein={perf_counter() - swd_started:.3f}s "
                f"projections={int(n_projections)}"
            )
            row_started = perf_counter()
            first_l2 = vector_l2_by_time(first - ssa_first)
            second_l2 = vector_l2_by_time(second - ssa_second)
            first_mean_abs = vector_mean_abs_by_time(first - ssa_first)
            second_mean_abs = vector_mean_abs_by_time(second - ssa_second)

            for index, time_value in enumerate(time_points):
                output.append(
                    {
                        "network": network,
                        "algorithm_label": label,
                        "time_index": int(index),
                        "time": float(time_value),
                        "n_ssa_samples": int(np.count_nonzero(valid_sample_mask(ssa_samples[:, index, :]))),
                        "n_blended_samples": int(np.count_nonzero(valid_sample_mask(samples[:, index, :]))),
                        "first_moment_l2_vs_ssa": float(first_l2[index]),
                        "first_moment_mean_abs_vs_ssa": float(first_mean_abs[index]),
                        "second_raw_moment_l2_vs_ssa": float(second_l2[index]),
                        "second_raw_moment_mean_abs_vs_ssa": float(second_mean_abs[index]),
                        "sliced_wasserstein_distance": float(swd_values[index]),
                    }
                )
            print(
                f"[analysis] network={network} label={label} "
                f"row_build={perf_counter() - row_started:.3f}s "
                f"total_label={perf_counter() - label_started:.3f}s"
            )

        save_started = perf_counter()
        np.savez_compressed(network_moment_path, **moment_payload)
        print(
            f"[analysis] network={network} moment_save={perf_counter() - save_started:.3f}s "
            f"path={network_moment_path}"
        )
        moment_paths.append(network_moment_path)
        print(f"[analysis] network={network} total={perf_counter() - network_started:.3f}s")
    return output, moment_paths


def load_group_samples(rows: list[RunRow], time_points: np.ndarray) -> tuple[np.ndarray, list[str]]:
    samples = []
    species_names: list[str] = []
    for row in rows:
        if row.trajectory_path is None or not trajectory_storage_exists(row.trajectory_path):
            continue
        file_size = trajectory_storage_size(row.trajectory_path)
        load_started = perf_counter()
        try:
            sampled, loaded_species_names, load_info = sample_trajectory_states_from_path(
                row.trajectory_path,
                time_points,
                mmap=True,
            )
        except Exception as exc:
            print(f"[analysis] skip trajectory load error: {row.trajectory_path} {exc!r}")
            continue
        load_elapsed = perf_counter() - load_started
        if not species_names:
            species_names = list(loaded_species_names)
        elif list(loaded_species_names) != species_names:
            print(f"[analysis] skip species mismatch: {row.trajectory_path}")
            continue
        samples.append(sampled)
        print(
            f"[analysis] trajectory method={row.method} label={row.algorithm_label} "
            f"run={row.raw.get('run_index')} file={row.trajectory_path.name} "
            f"storage={load_info['storage']} mmap={load_info['mmap']} "
            f"size={format_bytes(file_size)} load+sample={load_elapsed:.3f}s "
            f"times_shape={load_info['times_shape']} states_shape={load_info['states_shape']} "
            f"sampled_shape={sampled.shape}"
        )
    if not samples:
        return np.empty((0, len(time_points), 0), dtype=float), species_names
    stack_started = perf_counter()
    stacked = np.stack(samples, axis=0)
    print(
        f"[analysis] stacked group label={rows[0].algorithm_label if rows else ''} "
        f"n_trajectories={len(samples)} stack={perf_counter() - stack_started:.3f}s "
        f"stacked_shape={stacked.shape}"
    )
    return stacked, species_names


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}GiB"


def trajectory_storage_size(path: Path) -> int:
    if has_trajectory_sidecar(path):
        sidecar = trajectory_sidecar_dir(path)
        return sum(item.stat().st_size for item in sidecar.iterdir() if item.is_file())
    return path.stat().st_size if path.exists() else 0


def sample_trajectory_states(times: np.ndarray, states: np.ndarray, time_points: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    x = states if isinstance(states, np.memmap) else np.asarray(states, dtype=float)
    points = np.asarray(time_points, dtype=float)
    if t.ndim != 1 or x.ndim != 2 or x.shape[0] != t.shape[0]:
        raise ValueError("invalid trajectory arrays")
    result = np.full((points.size, x.shape[1]), np.nan, dtype=float)
    indices = np.searchsorted(t, points, side="right") - 1
    valid = (indices >= 0) & (points <= t[-1] + 1e-12)
    if np.any(valid):
        result[valid] = x[indices[valid]]
    return result


def sliced_wasserstein_by_time(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    n_projections: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_times = int(reference.shape[1])
    values = np.full(n_times, np.nan, dtype=float)
    for time_index in range(n_times):
        ref = reference[:, time_index, :]
        obs = observed[:, time_index, :]
        ref = ref[valid_sample_mask(ref)]
        obs = obs[valid_sample_mask(obs)]
        if ref.size == 0 or obs.size == 0:
            continue
        values[time_index] = sliced_wasserstein_distance(ref, obs, n_projections=n_projections, rng=rng)
    return values


def sliced_wasserstein_distance(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    n_projections: int,
    rng: np.random.Generator,
) -> float:
    ref = np.asarray(reference, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if ref.ndim != 2 or obs.ndim != 2 or ref.shape[1] != obs.shape[1]:
        raise ValueError("samples must be 2D arrays with the same feature dimension")
    dim = int(ref.shape[1])
    if dim == 0:
        return float("nan")
    directions = rng.normal(size=(int(n_projections), dim))
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 0.0] / norms[norms > 0.0, None]
    if directions.size == 0:
        return float("nan")
    distances = []
    for direction in directions:
        distances.append(wasserstein_1d(ref @ direction, obs @ direction))
    return float(np.mean(distances))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    q = (np.arange(max(x.size, y.size), dtype=float) + 0.5) / float(max(x.size, y.size))
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))


def vector_l2_by_time(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(np.square(np.asarray(values, dtype=float)), axis=1))


def vector_mean_abs_by_time(values: np.ndarray) -> np.ndarray:
    return np.nanmean(np.abs(np.asarray(values, dtype=float)), axis=1)


def valid_sample_mask(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.isfinite(arr)
    return np.all(np.isfinite(arr), axis=1)


def plot_wall_time_box_scatter(rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    if not rows:
        return None
    networks = sorted({str(row["network"]) for row in rows})
    labels = sorted({str(row["algorithm_label"]) for row in rows})
    if not networks or not labels:
        return None

    data: list[np.ndarray] = []
    positions: list[float] = []
    colors: list[Any] = []
    tick_positions: list[float] = []
    cmap = plt.get_cmap("tab10")
    group_width = 0.78
    label_width = group_width / max(len(labels), 1)

    for network_index, network in enumerate(networks):
        tick_positions.append(float(network_index))
        for label_index, label in enumerate(labels):
            values = [
                float(row["wall_runtime_seconds"])
                for row in rows
                if str(row["network"]) == network
                and str(row["algorithm_label"]) == label
                and np.isfinite(float(row["wall_runtime_seconds"]))
            ]
            if not values:
                continue
            offset = (label_index - (len(labels) - 1) / 2.0) * label_width
            data.append(np.asarray(values, dtype=float))
            positions.append(float(network_index) + offset)
            colors.append(cmap(label_index % 10))

    fig_width = max(9.0, 1.7 * len(networks) + 0.5 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    if data:
        box = ax.boxplot(
            data,
            positions=positions,
            widths=label_width * 0.72,
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
        for index, (pos, values, color) in enumerate(zip(positions, data, colors)):
            jitter_rng = np.random.default_rng(1000 + int(index))
            jitter = jitter_rng.uniform(
                -label_width * 0.18,
                label_width * 0.18,
                size=values.shape,
            )
            ax.scatter(
                np.full(values.shape, pos, dtype=float) + jitter,
                values,
                s=15,
                color=color,
                alpha=0.62,
                edgecolors="none",
                zorder=3,
            )

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=cmap(index % 10), edgecolor=cmap(index % 10), alpha=0.35, label=short_label(label))
        for index, label in enumerate(labels)
    ]
    ax.set_ylabel("wall seconds to common t_end")
    ax.set_xlabel("network")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(networks, rotation=30, ha="right")
    ax.legend(handles=legend_handles, fontsize=8)
    ax.set_title("Wall Time Distribution To Common Simulation Time")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_swd_lines_by_network(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not rows:
        return []
    output_paths: list[Path] = []
    for network, network_rows in group_by(rows, lambda row: str(row["network"])).items():
        path = output_dir / f"sliced_wasserstein_vs_time__{safe_name(network)}.png"
        plotted = plot_swd_lines_for_network(network_rows, path, network)
        if plotted is not None:
            output_paths.append(plotted)
    return output_paths


def plot_swd_lines_for_network(
    rows: list[dict[str, Any]],
    output_path: Path,
    network: str,
) -> Path | None:
    if not rows:
        return None
    grouped = group_by(rows, lambda row: str(row["algorithm_label"]))
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for label, group_rows in grouped.items():
        ordered = sorted(group_rows, key=lambda row: int(row["time_index"]))
        ax.plot(
            [float(row["time"]) for row in ordered],
            [float(row["sliced_wasserstein_distance"]) for row in ordered],
            linewidth=1.0,
            alpha=0.75,
            label=short_label(label),
        )
    ax.set_xlabel("simulation time")
    ax.set_ylabel("Sliced Wasserstein Distance vs SSA")
    ax.set_title(f"Blended State Distribution Distance To SSA: {network}")
    ax.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_final_total_vs_time_by_network(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not rows:
        return []
    paths: list[Path] = []
    for network, network_rows in group_by(rows, lambda row: str(row["network"])).items():
        path = output_dir / f"final_total_abundance_vs_simulation_time__{safe_name(network)}.png"
        plotted = plot_final_total_vs_time_for_network(network_rows, path, network)
        if plotted is not None:
            paths.append(plotted)
    return paths


def plot_final_total_vs_time_for_network(
    rows: list[dict[str, Any]],
    output_path: Path,
    network: str,
) -> Path | None:
    if not rows:
        return None
    grouped = group_by(rows, lambda row: str(row["algorithm_label"]))
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    cmap = plt.get_cmap("tab10")
    for index, (label, group_rows) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        pairs = [
            (float(row["simulation_final_time"]), float(row["final_total_abundance"]))
            for row in group_rows
            if np.isfinite(float(row["simulation_final_time"]))
            and np.isfinite(float(row["final_total_abundance"]))
        ]
        if not pairs:
            continue
        x_values, y_values = zip(*pairs)
        ax.scatter(
            x_values,
            y_values,
            s=18,
            alpha=0.65,
            color=cmap(index % 10),
            label=short_label(label),
            edgecolors="none",
        )
    ax.set_xlabel("simulation final time")
    ax.set_ylabel("final total abundance")
    ax.set_title(f"Final Total Abundance vs Simulation Time: {network}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


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


def group_by(items: list[Any], key_fn) -> dict[Any, list[Any]]:
    groups: dict[Any, list[Any]] = {}
    for item in items:
        key = key_fn(item)
        groups.setdefault(key, []).append(item)
    return groups


def resolve_path(value: Any, metadata_path: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (metadata_path.parent / path).resolve()


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def median(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not clean:
        return None
    return float(np.median(np.asarray(clean, dtype=float)))


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))[:120]


def short_label(value: str, max_len: int = 34) -> str:
    text = str(value)
    return text if len(text) <= int(max_len) else text[: int(max_len) - 3] + "..."


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Analyze accuracy_effecency_betaHybrid_test output."
    )
    arg_parser.add_argument("batch_dir", help="Directory containing run_metadata.json")
    arg_parser.add_argument("--output-dir", default=None)
    arg_parser.add_argument("--time-points", type=int, default=N_TIME_POINTS)
    arg_parser.add_argument("--projections", type=int, default=SLICED_WASSERSTEIN_PROJECTIONS)
    arg_parser.add_argument("--seed", type=int, default=SLICED_WASSERSTEIN_SEED)
    return arg_parser


def main() -> None:
    args = parser().parse_args()
    analyze_batch_output(
        args.batch_dir,
        output_dir=args.output_dir,
        n_time_points=int(args.time_points),
        n_projections=int(args.projections),
        random_seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
