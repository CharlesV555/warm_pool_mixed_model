"""单次运行轨迹绘图函数。

本模块只依赖离线记录对象或记录文件路径，不依赖主模拟器内部对象。
第一轮先提供最常用的时间序列绘图接口，后续可继续扩展更多分析函数。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from polymer_sim.core.enums import ChannelBlock
from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState
from polymer_sim.recording.base import PathLike
from polymer_sim.recording.summary import RunSummary, load_summary
from polymer_sim.recording.trajectory import TrajectoryRecord, load_trajectory_record


def plot_time_series(
    record_or_path: TrajectoryRecord | PathLike,
    species_indices: list[int] | np.ndarray | None = None,
    title: str | None = None,
):
    """绘制单次轨迹的时间序列图。

    横轴为时间，纵轴为浓度或拷贝数。返回 `(fig, ax)`，便于调用方继续自定义样式。
    """

    record = _as_trajectory_record(record_or_path)
    if species_indices is None:
        indices = np.arange(record.states.shape[1], dtype=np.int64)
    else:
        indices = np.asarray(species_indices, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sid in indices:
        ax.plot(record.times, record.states[:, sid], label=record.species_names[int(sid)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Count / Concentration")
    ax.set_title(title or "Single Run Time Series")
    if indices.size <= 12:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_reaction_trigger_frequency(
    record_or_path: TrajectoryRecord | PathLike,
    title: str | None = None,
    top_n: int | None = None,
):
    record = _as_trajectory_record(record_or_path)
    counts = np.asarray(record.run_metadata.get("channel_trigger_counts", []), dtype=float)
    labels_meta = record.run_metadata.get("channel_labels", [])
    if counts.size == 0:
        raise ValueError("record metadata does not contain channel_trigger_counts")

    labels = []
    for idx in range(counts.shape[0]):
        if idx < len(labels_meta):
            item = labels_meta[idx]
            reactants = item.get("reactants", ())
            products = item.get("products", ())
            labels.append(f"{idx}: {tuple(reactants)}->{tuple(products)}")
        else:
            labels.append(str(idx))

    order = np.arange(counts.shape[0], dtype=np.int64)
    if top_n is not None and int(top_n) < counts.shape[0]:
        order = np.argsort(counts)[-int(top_n) :]
        order = order[np.argsort(counts[order])[::-1]]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(order.shape[0]), counts[order])
    ax.set_xticks(np.arange(order.shape[0]))
    ax.set_xticklabels([labels[int(idx)] for idx in order], rotation=45, ha="right")
    ax.set_ylabel("Trigger Count")
    ax.set_xlabel("Reaction Channel")
    ax.set_title(title or "Reaction Trigger Frequency")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_reaction_frequency_over_time(
    record_or_path: TrajectoryRecord | PathLike,
    title: str | None = None,
    *,
    n_bins: int = 100,
    channel_ids: list[int] | np.ndarray | None = None,
    top_n: int | None = None,
    rate: bool = True,
):
    record = _as_trajectory_record(record_or_path)
    event_times = np.asarray(record.run_metadata.get("channel_event_times", []), dtype=float)
    event_ids = np.asarray(record.run_metadata.get("channel_event_ids", []), dtype=np.int64)
    if "channel_event_times" not in record.run_metadata or "channel_event_ids" not in record.run_metadata:
        raise ValueError(
            "record metadata does not contain channel_event_times/channel_event_ids; "
            "rerun the simulation with TrajectoryRecorder"
        )
    if event_times.shape[0] != event_ids.shape[0]:
        raise ValueError("channel_event_times and channel_event_ids must have the same length")

    bins = int(n_bins)
    if bins <= 0:
        raise ValueError("n_bins must be > 0")

    n_channels = int(record.run_metadata.get("n_channels", 0))
    if n_channels <= 0 and event_ids.size:
        n_channels = int(np.max(event_ids)) + 1
    if n_channels <= 0:
        n_channels = 1

    selected = _select_record_channels(n_channels, channel_ids)
    trigger_counts = np.asarray(record.run_metadata.get("channel_trigger_counts", []), dtype=float)
    if top_n is not None:
        limit = int(top_n)
        if limit <= 0:
            raise ValueError("top_n must be > 0")
    if top_n is not None and limit < selected.shape[0]:
        if trigger_counts.shape[0] < n_channels:
            trigger_counts = np.bincount(event_ids, minlength=n_channels).astype(float)
        scores = trigger_counts[selected]
        order = np.argsort(scores)[-limit:]
        selected = selected[order[np.argsort(scores[order])[::-1]]]

    start = float(record.times[0]) if record.times.size else 0.0
    end = float(record.times[-1]) if record.times.size else start + 1.0
    if end <= start:
        end = start + 1.0
    edges = np.linspace(start, end, bins + 1)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2.0

    values = np.zeros((selected.shape[0], bins), dtype=float)
    for row, channel_id in enumerate(selected):
        channel_event_times = event_times[event_ids == int(channel_id)]
        counts, _ = np.histogram(channel_event_times, bins=edges)
        values[row, :] = counts
    if rate:
        values = values / widths[np.newaxis, :]

    labels_meta = record.run_metadata.get("channel_labels", [])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bottom = np.zeros(bins, dtype=float)
    plotted = False
    for row, channel_id in enumerate(selected):
        y = values[row]
        if not np.any(y):
            continue
        label = _format_record_channel_label(labels_meta, int(channel_id)) if selected.shape[0] <= 12 else None
        ax.bar(
            centers,
            y,
            width=widths * 0.92,
            bottom=bottom,
            align="center",
            label=label,
        )
        bottom += y
        plotted = True
    if not plotted:
        ax.bar(centers, np.zeros_like(centers), width=widths * 0.92, align="center")

    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("Time")
    ax.set_ylabel("Trigger Rate" if rate else "Trigger Count")
    ax.set_title(title or "Reaction Frequency Over Time")
    if selected.shape[0] <= 12 and plotted:
        ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_reaction_interval_bar(
    record_or_path: TrajectoryRecord | PathLike,
    title: str | None = None,
    log: bool = False,
    *,
    n_bars: int = 100,
    time_range: tuple[float | None, float | None] | None = None,
):
    record = _as_trajectory_record(record_or_path)
    if time_range is None:
        intervals = _reaction_intervals_from_metadata(record)
    else:
        times, intervals = _reaction_interval_series_from_metadata(record)
        intervals = _filter_interval_series_by_time(times, intervals, time_range)[1]
    if log:
        intervals = np.log10(intervals[intervals > 0.0])
    bars = int(n_bars)
    if bars <= 0:
        raise ValueError("n_bars must be > 0")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if intervals.size == 0:
        centers = np.arange(bars, dtype=float)
        ax.bar(centers, np.zeros(bars, dtype=float), width=0.9)
        ax.set_xlabel("Reaction Interval Bin")
    else:
        counts, edges = np.histogram(intervals, bins=bars)
        widths = np.diff(edges)
        centers = edges[:-1] + widths / 2.0
        ax.bar(centers, counts, width=widths * 0.92, align="center")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel("Reaction Interval Time")

    ax.set_ylabel("Reaction Count")
    ax.set_title(title or "Reaction Interval Bar Plot")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_reaction_interval_wave(
    record_or_path: TrajectoryRecord | PathLike,
    time_windows: list[dict] | list[tuple] | None = None,
    title: str | None = None,
    log: bool = False,
    *,
    n_bins: int = 100,
    bw_adjust: float = 0.5,
    aspect: float = 15.0,
    height: float = 0.5,
):
    record = _as_trajectory_record(record_or_path)
    event_times, intervals = _reaction_interval_series_from_metadata(record)
    windows = _normalize_time_windows(time_windows)
    rows = []
    labels = []
    values_by_window = []
    
    ###
    print("windows:")
    for w in windows:
        print(w)

    print("event_times range:", np.min(event_times), np.max(event_times))
    print("n_event_times:", len(event_times))
    print("n_intervals:", len(intervals))
    ###
    
    for start, end, _label in windows:
        _times, window_intervals = _filter_interval_series_by_time(event_times, intervals, (start, end))
        values = _transform_intervals(window_intervals, log)
        print(
            f"{_label}:",
            f"window=({start}, {end})",
            f"n_times={len(_times)}",
            f"n_intervals={len(window_intervals)}",
            f"n_values={len(values)}",
        )
        labels.append(_label)
        values_by_window.append(values)
        for value in values:
            rows.append({"interval": float(value), "time_window": _label})
    x_label = "log10(Reaction Interval Time)" if log else "Reaction Interval Time"
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})    
    
    df = pd.DataFrame(rows)
    
    print(df["time_window"].value_counts())
    
    palette = sns.cubehelix_palette(len(labels), rot=-0.25, light=0.7)
    grid = sns.FacetGrid(
        df,
        row="time_window",
        hue="time_window",
        row_order=labels,
        hue_order=labels,
        aspect=float(aspect),
        height=float(height),
        palette=palette,
    )
    grid.map(
        sns.kdeplot,
        "interval",
        bw_adjust=float(bw_adjust),
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
        warn_singular=False,
    )
    grid.map(
        sns.kdeplot,
        "interval",
        bw_adjust=float(bw_adjust),
        clip_on=False,
        color="w",
        lw=2,
        warn_singular=False,
    )
    if hasattr(grid, "refline"):
        grid.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    else:
        def refline(_x, color=None):
            plt.axhline(y=0, linewidth=2, linestyle="-", color=color, clip_on=False)

        grid.map(refline, "interval")
        
    return grid.figure, grid 


def plot_species_with_outflow(
    record_or_path: TrajectoryRecord | PathLike,
    species_indices: list[int] | np.ndarray | None = None,
    title: str | None = None,
):
    record = _as_trajectory_record(record_or_path)
    outflow = record.run_metadata.get("tracked_outflow")
    if outflow is None:
        raise ValueError("record metadata does not contain tracked_outflow")

    if species_indices is None:
        indices = np.arange(record.states.shape[1], dtype=np.int64)
    else:
        indices = np.asarray(species_indices, dtype=np.int64)

    outflow_times = np.asarray(outflow.get("times", []), dtype=float)
    outflow_removed = np.asarray(outflow.get("removed", []), dtype=float)
    outflow_names = list(outflow.get("species_names", []))

    fig, ax_left = plt.subplots(figsize=(9, 5))
    for sid in indices:
        ax_left.plot(record.times, record.states[:, sid], label=record.species_names[int(sid)])
    ax_left.set_xlabel("Time")
    ax_left.set_ylabel("Count / Concentration")
    ax_left.grid(True, alpha=0.3)
    if indices.size <= 12:
        ax_left.legend(loc="upper left")

    ax_right = ax_left.twinx()
    for col in range(outflow_removed.shape[1] if outflow_removed.ndim == 2 else 0):
        ax_right.plot(
            outflow_times,
            outflow_removed[:, col],
            linestyle="--",
            linewidth=1.5,
            label=f"{outflow_names[col]} outflow",
        )
    ax_right.set_ylabel("Outflow Removed Per Step")
    if outflow_removed.size > 0:
        ax_right.legend(loc="upper right")

    ax_left.set_title(title or "Species Trajectory With Outflow")
    fig.tight_layout()
    return fig, (ax_left, ax_right)


def plot_channel_propensity_time_series(
    record_or_path: TrajectoryRecord | PathLike,
    network: ReactionNetworkData,
    channel_ids: list[int] | np.ndarray | None = None,
    *,
    block_type: ChannelBlock | str | int | None = None,
    title: str | None = None,
    top_n: int | None = None,
):
    record = _as_trajectory_record(record_or_path)
    channels = _select_channels(network, channel_ids, block_type)
    if channels.size == 0:
        raise ValueError("no channels selected")

    values = np.empty((record.times.shape[0], channels.shape[0]), dtype=float)
    for row, (time, state_values) in enumerate(zip(record.times, record.states)):
        state = SystemState(t=float(time), x=np.asarray(state_values, dtype=float))
        for col, channel_id in enumerate(channels):
            values[row, col] = network.compute_propensity(int(channel_id), state)

    order = np.arange(channels.shape[0], dtype=np.int64)
    if top_n is not None and int(top_n) < channels.shape[0]:
        scores = np.max(values, axis=0)
        order = np.argsort(scores)[-int(top_n) :]
        order = order[np.argsort(scores[order])[::-1]]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for idx in order:
        channel_id = int(channels[int(idx)])
        ax.plot(
            record.times,
            values[:, int(idx)],
            label=_format_channel_label(network, channel_id),
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Propensity")
    ax.set_title(title or _propensity_title(block_type))
    if order.size <= 12:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_final_state_distribution(
    summaries_or_paths: list[RunSummary] | list[PathLike],
    species_index: int,
    bins: int = 20,
    title: str | None = None,
):
    summaries = _as_summary_list(summaries_or_paths)
    values = np.asarray([summary.final_state[int(species_index)] for summary in summaries], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=int(bins))
    ax.set_xlabel("Final Count / Concentration")
    ax.set_ylabel("Frequency")
    ax.set_title(title or f"Final State Distribution (species {species_index})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_event_time_distribution(
    summaries_or_paths: list[RunSummary] | list[PathLike],
    bins: int = 20,
    title: str | None = None,
):
    summaries = _as_summary_list(summaries_or_paths)
    values_list = []
    for summary in summaries:
        if summary.event_times is not None and summary.event_times.size > 0:
            values_list.append(np.asarray(summary.event_times, dtype=float))
        else:
            values_list.append(np.asarray([summary.final_time], dtype=float))
    values = np.concatenate(values_list) if values_list else np.empty(0, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=int(bins))
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title(title or "Event Time Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_mean_std_over_runs(
    summaries_or_paths: list[RunSummary] | list[PathLike],
    species_indices: list[int] | np.ndarray | None = None,
    title: str | None = None,
):
    summaries = _as_summary_list(summaries_or_paths)
    final_states = np.vstack([summary.final_state for summary in summaries]) if summaries else np.empty((0, 0))
    if species_indices is None:
        indices = np.arange(final_states.shape[1], dtype=np.int64)
    else:
        indices = np.asarray(species_indices, dtype=np.int64)

    means = np.mean(final_states[:, indices], axis=0)
    stds = np.std(final_states[:, indices], axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(indices, means, yerr=stds, fmt="-o", capsize=4)
    ax.set_xlabel("Species Index")
    ax.set_ylabel("Final Count / Concentration")
    ax.set_title(title or "Mean +/- Std Over Runs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def _as_trajectory_record(record_or_path: TrajectoryRecord | PathLike) -> TrajectoryRecord:
    if isinstance(record_or_path, TrajectoryRecord):
        return record_or_path
    return load_trajectory_record(Path(record_or_path))


def _reaction_intervals_from_metadata(record: TrajectoryRecord) -> np.ndarray:
    if "reaction_intervals" in record.run_metadata:
        return _clean_intervals(record.run_metadata.get("reaction_intervals", []))

    if "channel_event_times" in record.run_metadata:
        event_times = np.asarray(record.run_metadata.get("channel_event_times", []), dtype=float)
        if event_times.size == 0:
            return np.empty(0, dtype=float)
        start = float(record.times[0]) if record.times.size else 0.0
        return _clean_intervals(np.diff(np.concatenate(([start], event_times))))

    n_events = int(record.run_metadata.get("n_events", -1))
    if n_events == max(0, int(record.times.shape[0]) - 1):
        return _clean_intervals(np.diff(record.times))

    raise ValueError(
        "record metadata does not contain reaction_intervals or channel_event_times; "
        "rerun the simulation with TrajectoryRecorder"
    )


def _reaction_interval_series_from_metadata(record: TrajectoryRecord) -> tuple[np.ndarray, np.ndarray]:
    if "reaction_intervals" in record.run_metadata:
        intervals = np.asarray(record.run_metadata.get("reaction_intervals", []), dtype=float)
        if intervals.ndim != 1:
            raise ValueError("reaction intervals must have shape (n_events,)")
        if intervals.size == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        if "reaction_interval_times" in record.run_metadata:
            times = np.asarray(record.run_metadata.get("reaction_interval_times", []), dtype=float)
        elif "channel_event_times" in record.run_metadata:
            times = np.asarray(record.run_metadata.get("channel_event_times", []), dtype=float)
        elif int(record.run_metadata.get("n_events", -1)) == max(0, int(record.times.shape[0]) - 1):
            times = np.asarray(record.times[1:], dtype=float)
        else:
            raise ValueError(
                "record metadata does not contain reaction_interval_times; "
                "rerun the simulation with the updated TrajectoryRecorder"
            )
        return _clean_interval_series(times, intervals)

    if "channel_event_times" in record.run_metadata:
        event_times = np.asarray(record.run_metadata.get("channel_event_times", []), dtype=float)
        if event_times.size == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        start = float(record.times[0]) if record.times.size else 0.0
        intervals = np.diff(np.concatenate(([start], event_times)))
        return _clean_interval_series(event_times, intervals)

    n_events = int(record.run_metadata.get("n_events", -1))
    if n_events == max(0, int(record.times.shape[0]) - 1):
        return _clean_interval_series(record.times[1:], np.diff(record.times))

    raise ValueError(
        "record metadata does not contain reaction_interval_times or channel_event_times; "
        "rerun the simulation with TrajectoryRecorder"
    )


def _clean_intervals(values) -> np.ndarray:
    intervals = np.asarray(values, dtype=float)
    if intervals.ndim != 1:
        raise ValueError("reaction intervals must have shape (n_events,)")
    intervals = intervals[np.isfinite(intervals)]
    return intervals[intervals >= 0.0]


def _clean_interval_series(times, intervals) -> tuple[np.ndarray, np.ndarray]:
    event_times = np.asarray(times, dtype=float)
    interval_values = np.asarray(intervals, dtype=float)
    if event_times.ndim != 1:
        raise ValueError("reaction interval times must have shape (n_events,)")
    if interval_values.ndim != 1:
        raise ValueError("reaction intervals must have shape (n_events,)")
    if event_times.shape[0] != interval_values.shape[0]:
        raise ValueError("reaction_interval_times and reaction_intervals must have the same length")
    valid = np.isfinite(event_times) & np.isfinite(interval_values) & (interval_values >= 0.0)
    return event_times[valid], interval_values[valid]


def _filter_interval_series_by_time(
    times: np.ndarray,
    intervals: np.ndarray,
    time_range: tuple[float | None, float | None],
) -> tuple[np.ndarray, np.ndarray]:
    start, end = time_range
    lower = None if start is None else float(start)
    upper = None if end is None else float(end)
    if lower is not None and upper is not None and upper <= lower:
        raise ValueError("time window end must be greater than start")

    mask = np.ones(times.shape[0], dtype=bool)
    if lower is not None:
        mask &= times >= lower
    if upper is not None:
        mask &= times < upper
    return times[mask], intervals[mask]


def _normalize_time_windows(time_windows: list[dict] | list[tuple] | None) -> list[tuple[float | None, float | None, str]]:
    if time_windows is None:
        return [(None, None, "all")]

    normalized: list[tuple[float | None, float | None, str]] = []
    for item in time_windows:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            label = item.get("label")
        else:
            values = tuple(item)
            if len(values) == 2:
                start, end = values
                label = None
            elif len(values) == 3 and isinstance(values[0], str):
                label, start, end = values
            elif len(values) == 3:
                start, end, label = values
            else:
                raise ValueError("time windows must be dicts, (start, end), or (label, start, end)")

        start_value = None if start is None else float(start)
        end_value = None if end is None else float(end)
        if start_value is not None and end_value is not None and end_value <= start_value:
            raise ValueError("time window end must be greater than start")
        window_label = str(label) if label is not None else _auto_time_window_label(start_value, end_value)
        normalized.append((start_value, end_value, window_label))

    if not normalized:
        raise ValueError("time_windows must contain at least one window")
    return normalized


def _auto_time_window_label(start: float | None, end: float | None) -> str:
    start_label = "start" if start is None else f"{start:g}"
    end_label = "end" if end is None else f"{end:g}"
    return f"{start_label}-{end_label}"


def _transform_intervals(intervals: np.ndarray, log: bool) -> np.ndarray:
    values = np.asarray(intervals, dtype=float)
    if not log:
        return values
    return np.log10(values[values > 0.0])


def _plot_reaction_interval_wave_fallback(
    values_by_window: list[np.ndarray],
    labels: list[str],
    title: str | None,
    x_label: str,
    n_bins: int,
):
    bins = int(n_bins)
    if bins <= 0:
        raise ValueError("n_bins must be > 0")

    all_values = np.concatenate([values for values in values_by_window if values.size]) if any(
        values.size for values in values_by_window
    ) else np.empty(0, dtype=float)
    edges = _interval_histogram_edges(all_values, bins)
    centers = edges[:-1] + np.diff(edges) / 2.0

    fig, ax = plt.subplots(figsize=(10, max(3.5, 1.15 * len(labels) + 1.5)))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(labels), 1)))
    for row, (label, values, color) in enumerate(zip(labels, values_by_window, colors)):
        y = np.zeros(edges.shape[0] - 1, dtype=float)
        if values.size:
            y, _ = np.histogram(values, bins=edges, density=True)
            y = np.asarray(y, dtype=float)
            y[~np.isfinite(y)] = 0.0
            peak = float(np.max(y)) if y.size else 0.0
            if peak > 0.0:
                y = y / peak * 0.82
        ax.fill_between(centers, row, row + y, color=color, alpha=0.35)
        ax.plot(centers, row + y, color=color, linewidth=1.5)

    ax.set_yticks(np.arange(len(labels), dtype=float))
    ax.set_yticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time Window")
    ax.set_title(title or "Reaction Interval Wave Plot")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def _interval_histogram_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.linspace(0.0, 1.0, int(n_bins) + 1)

    left = float(np.min(data))
    right = float(np.max(data))
    if not np.isfinite(left) or not np.isfinite(right):
        return np.linspace(0.0, 1.0, int(n_bins) + 1)
    if right <= left:
        padding = max(abs(left) * 0.05, 1e-12)
        left -= padding
        right += padding
    return np.linspace(left, right, int(n_bins) + 1)


def _as_summary_list(summaries_or_paths: list[RunSummary] | list[PathLike]) -> list[RunSummary]:
    if not summaries_or_paths:
        return []
    first = summaries_or_paths[0]
    if isinstance(first, RunSummary):
        return [item for item in summaries_or_paths if isinstance(item, RunSummary)]

    result: list[RunSummary] = []
    for path in summaries_or_paths:
        loaded = load_summary(Path(path))
        if isinstance(loaded, RunSummary):
            result.append(loaded)
        else:
            result.extend(loaded)
    return result


def _select_record_channels(n_channels: int, channel_ids: list[int] | np.ndarray | None) -> np.ndarray:
    if channel_ids is None:
        selected = np.arange(int(n_channels), dtype=np.int64)
    else:
        selected = np.asarray(channel_ids, dtype=np.int64)
    if np.any(selected < 0) or np.any(selected >= int(n_channels)):
        raise IndexError("channel_id out of range")
    return selected


def _format_record_channel_label(labels_meta: list[dict] | list, channel_id: int) -> str:
    if channel_id < len(labels_meta):
        item = labels_meta[channel_id]
        reactants = item.get("reactants", ())
        products = item.get("products", ())
        return f"{channel_id}: {tuple(reactants)}->{tuple(products)}"
    return str(channel_id)


def _select_channels(
    network: ReactionNetworkData,
    channel_ids: list[int] | np.ndarray | None,
    block_type: ChannelBlock | str | int | None,
) -> np.ndarray:
    if channel_ids is None:
        selected = np.arange(network.n_channels, dtype=np.int64)
    else:
        selected = np.asarray(channel_ids, dtype=np.int64)

    if np.any(selected < 0) or np.any(selected >= network.n_channels):
        raise IndexError("channel_id out of range")

    if block_type is None:
        return selected

    block = _coerce_block_type(block_type)
    mask = np.array(
        [network.get_channel_block(int(channel_id)) == block for channel_id in selected],
        dtype=bool,
    )
    return selected[mask]


def _coerce_block_type(block_type: ChannelBlock | str | int) -> ChannelBlock:
    if isinstance(block_type, ChannelBlock):
        return block_type
    if isinstance(block_type, str):
        normalized = block_type.upper()
        try:
            return ChannelBlock[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown channel block type: {block_type}") from exc
    return ChannelBlock(int(block_type))


def _format_channel_label(network: ReactionNetworkData, channel_id: int) -> str:
    reactants = "+".join(network.species_names[int(sid)] for sid in network.get_channel_reactants(channel_id))
    products_tuple = network.get_channel_products(channel_id)
    products = "+".join(network.species_names[int(sid)] for sid in products_tuple) if products_tuple else "out"
    return f"{channel_id}: {network.get_channel_block_name(channel_id)} {reactants}->{products}"


def _propensity_title(block_type: ChannelBlock | str | int | None) -> str:
    if block_type is None:
        return "Reaction Propensity Over Time"
    block = _coerce_block_type(block_type)
    return f"{block.name} Propensity Over Time"
