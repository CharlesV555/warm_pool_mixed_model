"""单次运行轨迹绘图函数。

本模块只依赖离线记录对象或记录文件路径，不依赖主模拟器内部对象。
第一轮先提供最常用的时间序列绘图接口，后续可继续扩展更多分析函数。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Patch
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
    time_range: tuple[float | None, float | None] | None = None,
):
    """绘制单次轨迹的时间序列图。

    横轴为时间，纵轴为浓度或拷贝数。返回 `(fig, ax)`，便于调用方继续自定义样式。
    """

    record = _as_trajectory_record(record_or_path)
    if species_indices is None:
        indices = np.arange(record.states.shape[1], dtype=np.int64)
    else:
        indices = np.asarray(species_indices, dtype=np.int64)

    times, states = _time_series_window(record, time_range)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sid in indices:
        ax.plot(times, states[:, sid], label=record.species_names[int(sid)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Count / Concentration")
    ax.set_title(title or "Single Run Time Series")

    if indices.size <= 12:
        ax.legend(
            facecolor="white",      # 图例背景
            edgecolor="black",      # 图例边框
            framealpha=1.0,         # 不透明
            labelcolor="black"      # 图例文字颜色
        )

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


def plot_reaction_network_state_tree(
    record_or_path: TrajectoryRecord | PathLike,
    time_points: tuple[float | None, float | None] | None = None,
    *,
    state_pair: tuple[np.ndarray, np.ndarray] | None = None,
    species_lengths: list[int] | np.ndarray | None = None,
    radius_mode: str = "area",
    radius_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    max_node_radius: float = 0.18,
    spacing_area_factor: float = 1.2,
    start_color: str = "#2ca02c",
    end_color: str = "#d62728",
    edge_color: str = "#8a8a8a",
    show_layer_guides: bool = True,
    label_alpha: float = 1.0,
    title: str | None = None,
):
    """Plot a radial reaction-network tree comparing two recorded states.

    Nodes are grouped by species length. Each node has two concentric circles:
    green for the earlier state and red for the later state. By default circle
    area is proportional to concentration; use radius_mode="radius" or
    radius_transform=... to change that mapping.
    """

    record = _as_trajectory_record(record_or_path)
    if record.times.size == 0 or record.states.size == 0:
        raise ValueError("trajectory record is empty")

    start_time, end_time, start_state, end_state = _state_pair_from_record(
        record,
        time_points,
        state_pair,
    )
    lengths = _species_lengths(record.species_names, species_lengths)
    scale_reference = max(float(np.max(start_state)), float(np.max(end_state)))
    start_radii = _node_radii(start_state, max_node_radius, radius_mode, radius_transform, scale_reference)
    end_radii = _node_radii(end_state, max_node_radius, radius_mode, radius_transform, scale_reference)
    node_radii = np.maximum(start_radii, end_radii)
    positions, layer_radii = _radial_species_layout(
        record.species_names,
        lengths,
        node_radii,
        spacing_area_factor,
    )
    edges = _reaction_tree_edges(record, lengths)

    fig, ax = plt.subplots(figsize=(8, 8))
    if show_layer_guides:
        for length, radius in layer_radii.items():
            if radius > 0.0:
                ax.add_patch(
                    Circle(
                        (0.0, 0.0),
                        radius,
                        fill=False,
                        edgecolor="#d0d0d0",
                        linewidth=0.7,
                        linestyle="--",
                        zorder=0,
                    )
                )
            label_x = radius + max(float(np.max(node_radii)), 0.04) * 1.4
            ax.text(label_x, 0.0, f"L{length}", fontsize=8, color="#666666", va="center")

    for left, right in edges:
        x0, y0 = positions[int(left)]
        x1, y1 = positions[int(right)]
        ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=0.8, alpha=0.45, zorder=1)

    for sid, name in enumerate(record.species_names):
        x, y = positions[int(sid)]
        circles = [
            (float(start_radii[sid]), start_color, 2),
            (float(end_radii[sid]), end_color, 3),
        ]
        for radius, color, zorder in sorted(circles, key=lambda item: item[0], reverse=True):
            if radius <= 0.0:
                continue
            ax.add_patch(
                Circle(
                    (x, y),
                    radius,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.24,
                    linewidth=1.2,
                    zorder=zorder,
                )
            )
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            alpha=float(label_alpha),
            zorder=4,
        )

    legend_handles = [
        Patch(facecolor=start_color, edgecolor=start_color, alpha=0.24, label=f"t={start_time:g}"),
        Patch(facecolor=end_color, edgecolor=end_color, alpha=0.24, label=f"t={end_time:g}"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title(title or "Reaction Network State Tree")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    _set_equal_tree_limits(ax, positions, node_radii)
    fig.tight_layout()
    return fig, ax


def animate_reaction_network_state_tree(
    record_or_path: TrajectoryRecord | PathLike,
    dt: float,
    *,
    time_range: tuple[float | None, float | None] | None = None,
    species_lengths: list[int] | np.ndarray | None = None,
    radius_mode: str = "area",
    radius_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    max_node_radius: float = 0.18,
    spacing_area_factor: float = 1.2,
    show_previous_state: bool = True,
    start_color: str = "#2ca02c",
    end_color: str = "#d62728",
    edge_color: str = "#8a8a8a",
    show_layer_guides: bool = True,
    label_alpha: float = 1.0,
    frame_interval_ms: int = 200,
    repeat: bool = False,
    title: str | None = None,
    save_path: PathLike | None = None,
    writer: str | None = None,
    dpi: int = 120,
):
    """Animate the radial reaction-network state tree over simulated time.

    Frame times are sampled every `dt` along the trajectory time axis. Each
    frame displays the recorded state nearest to its frame time. By default,
    the previous frame's nearest state is shown in green and the current
    nearest state in red.
    """

    record = _as_trajectory_record(record_or_path)
    if record.times.size == 0 or record.states.size == 0:
        raise ValueError("trajectory record is empty")

    frame_times = _animation_frame_times(record, dt, time_range)
    frame_indices = _nearest_record_indices(record.times, frame_times)
    frame_states = record.states[frame_indices]
    frame_radii = _animation_frame_radii(
        frame_states,
        max_node_radius,
        radius_mode,
        radius_transform,
    )
    layout_node_radii = np.max(frame_radii, axis=0) if frame_radii.size else np.zeros(record.states.shape[1])

    lengths = _species_lengths(record.species_names, species_lengths)
    positions, layer_radii = _radial_species_layout(
        record.species_names,
        lengths,
        layout_node_radii,
        spacing_area_factor,
    )
    edges = _reaction_tree_edges(record, lengths)

    fig, ax = plt.subplots(figsize=(8, 8))
    static_artists = _draw_reaction_tree_static_layers(
        ax,
        record,
        positions,
        layer_radii,
        layout_node_radii,
        edges,
        edge_color,
        show_layer_guides,
    )

    previous_patches: list[Circle] = []
    current_patches: list[Circle] = []
    for sid in range(len(record.species_names)):
        x, y = positions[int(sid)]
        if show_previous_state:
            previous = Circle(
                (x, y),
                0.0,
                facecolor=start_color,
                edgecolor=start_color,
                alpha=0.18,
                linewidth=1.1,
                zorder=2,
            )
            ax.add_patch(previous)
            previous_patches.append(previous)
        current = Circle(
            (x, y),
            0.0,
            facecolor=end_color,
            edgecolor=end_color,
            alpha=0.28,
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(current)
        current_patches.append(current)

    for sid, name in enumerate(record.species_names):
        x, y = positions[int(sid)]
        static_artists.append(
            ax.text(
                x,
                y,
                name,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                alpha=float(label_alpha),
                zorder=4,
            )
        )

    legend_handles = []
    if show_previous_state:
        legend_handles.append(Patch(facecolor=start_color, edgecolor=start_color, alpha=0.18, label="previous frame"))
    legend_handles.append(Patch(facecolor=end_color, edgecolor=end_color, alpha=0.28, label="current frame"))
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    _set_equal_tree_limits(ax, positions, layout_node_radii)
    fig.tight_layout()

    def update(frame: int):
        current_radii = frame_radii[int(frame)]
        if show_previous_state:
            previous_frame = max(int(frame) - 1, 0)
            previous_radii = frame_radii[previous_frame]
            for patch, radius in zip(previous_patches, previous_radii):
                patch.set_radius(float(radius))
        for patch, radius in zip(current_patches, current_radii):
            patch.set_radius(float(radius))

        actual_time = float(record.times[int(frame_indices[int(frame)])])
        target_time = float(frame_times[int(frame)])
        base_title = title or "Reaction Network State Tree"
        ax.set_title(f"{base_title}  target t={target_time:g}, nearest record t={actual_time:g}")
        return [*previous_patches, *current_patches, *static_artists]

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_times.shape[0],
        interval=int(frame_interval_ms),
        repeat=bool(repeat),
        blit=False,
    )
    update(0)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_writer = writer
        if save_writer is None and output_path.suffix.lower() == ".gif":
            save_writer = "pillow"
        animation.save(str(output_path), writer=save_writer, dpi=int(dpi))

    return fig, ax, animation


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


def _time_series_window(
    record: TrajectoryRecord,
    time_range: tuple[float | None, float | None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if time_range is None:
        return record.times, record.states
    if len(time_range) != 2:
        raise ValueError("time_range must contain exactly two values")

    start, end = time_range
    lower = float(record.times[0]) if start is None else float(start)
    upper = float(record.times[-1]) if end is None else float(end)
    if upper < lower:
        raise ValueError("time_range end must be greater than or equal to start")

    mask = (record.times >= lower) & (record.times <= upper)
    if not np.any(mask):
        raise ValueError("time_range does not include any recorded trajectory points")
    return record.times[mask], record.states[mask]


def _animation_frame_times(
    record: TrajectoryRecord,
    dt: float,
    time_range: tuple[float | None, float | None] | None,
) -> np.ndarray:
    step = float(dt)
    if step <= 0.0:
        raise ValueError("dt must be > 0")

    if time_range is None:
        start = float(record.times[0])
        end = float(record.times[-1])
    else:
        start = float(record.times[0]) if time_range[0] is None else float(time_range[0])
        end = float(record.times[-1]) if time_range[1] is None else float(time_range[1])

    record_start = float(record.times[0])
    record_end = float(record.times[-1])
    if start < record_start or start > record_end:
        raise ValueError(f"time range start {start:g} is outside [{record_start:g}, {record_end:g}]")
    if end < record_start or end > record_end:
        raise ValueError(f"time range end {end:g} is outside [{record_start:g}, {record_end:g}]")
    if end < start:
        raise ValueError("time range end must be greater than or equal to start")
    if end == start:
        return np.asarray([start], dtype=float)

    n_steps = int(np.floor((end - start) / step))
    frame_times = start + np.arange(n_steps + 1, dtype=float) * step
    if frame_times[-1] < end:
        frame_times = np.append(frame_times, end)
    return frame_times


def _nearest_record_indices(times: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    record_times = np.asarray(times, dtype=float)
    targets = np.asarray(frame_times, dtype=float)
    if record_times.ndim != 1 or targets.ndim != 1:
        raise ValueError("times and frame_times must be one-dimensional")
    if record_times.size == 0:
        raise ValueError("record times are empty")

    right = np.searchsorted(record_times, targets, side="left")
    right = np.clip(right, 0, record_times.shape[0] - 1)
    left = np.clip(right - 1, 0, record_times.shape[0] - 1)
    choose_left = np.abs(targets - record_times[left]) <= np.abs(record_times[right] - targets)
    return np.where(choose_left, left, right).astype(np.int64, copy=False)


def _animation_frame_radii(
    frame_states: np.ndarray,
    max_node_radius: float,
    radius_mode: str,
    radius_transform: Callable[[np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    states = np.asarray(frame_states, dtype=float)
    if states.ndim != 2:
        raise ValueError("frame states must have shape (n_frames, n_species)")
    scale_reference = None if radius_transform is not None else float(np.max(np.maximum(states, 0.0)))
    rows = [
        _node_radii(state, max_node_radius, radius_mode, radius_transform, scale_reference)
        for state in states
    ]
    return np.vstack(rows) if rows else np.empty_like(states, dtype=float)


def _draw_reaction_tree_static_layers(
    ax,
    record: TrajectoryRecord,
    positions: dict[int, tuple[float, float]],
    layer_radii: dict[int, float],
    node_radii: np.ndarray,
    edges: list[tuple[int, int]],
    edge_color: str,
    show_layer_guides: bool,
) -> list:
    artists = []
    if show_layer_guides:
        for length, radius in layer_radii.items():
            if radius > 0.0:
                guide = Circle(
                    (0.0, 0.0),
                    radius,
                    fill=False,
                    edgecolor="#d0d0d0",
                    linewidth=0.7,
                    linestyle="--",
                    zorder=0,
                )
                ax.add_patch(guide)
                artists.append(guide)
            label_x = radius + max(float(np.max(node_radii)), 0.04) * 1.4
            artists.append(ax.text(label_x, 0.0, f"L{length}", fontsize=8, color="#666666", va="center"))

    for left, right in edges:
        x0, y0 = positions[int(left)]
        x1, y1 = positions[int(right)]
        line = ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=0.8, alpha=0.45, zorder=1)[0]
        artists.append(line)

    return artists


def _state_pair_from_record(
    record: TrajectoryRecord,
    time_points: tuple[float | None, float | None] | None,
    state_pair: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    start_time, end_time = _normalize_state_tree_time_points(record, time_points)
    if state_pair is not None:
        first, second = state_pair
        return (
            start_time,
            end_time,
            _clean_state_vector(first, record.states.shape[1]),
            _clean_state_vector(second, record.states.shape[1]),
        )

    actual_start_time, start_state = _nearest_record_state(record, start_time)
    actual_end_time, end_state = _nearest_record_state(record, end_time)
    return actual_start_time, actual_end_time, start_state, end_state


def _normalize_state_tree_time_points(
    record: TrajectoryRecord,
    time_points: tuple[float | None, float | None] | None,
) -> tuple[float, float]:
    if time_points is None:
        return float(record.times[0]), float(record.times[-1])
    if len(time_points) != 2:
        raise ValueError("time_points must contain exactly two values")
    start = float(record.times[0]) if time_points[0] is None else float(time_points[0])
    end = float(record.times[-1]) if time_points[1] is None else float(time_points[1])
    if end < start:
        raise ValueError("the second time point must be greater than or equal to the first")
    return start, end


def _nearest_record_state(record: TrajectoryRecord, time_point: float) -> tuple[float, np.ndarray]:
    target = float(time_point)
    first = float(record.times[0])
    last = float(record.times[-1])
    if target < first or target > last:
        raise ValueError(f"time point {target:g} is outside the trajectory range [{first:g}, {last:g}]")
    index = int(np.argmin(np.abs(record.times - target)))
    return float(record.times[index]), _clean_state_vector(record.states[index], record.states.shape[1])


def _clean_state_vector(values: np.ndarray, n_species: int) -> np.ndarray:
    state = np.asarray(values, dtype=float)
    if state.shape != (int(n_species),):
        raise ValueError(f"state vector must have shape ({int(n_species)},)")
    if not np.all(np.isfinite(state)):
        raise ValueError("state vector contains non-finite values")
    return np.maximum(state, 0.0)


def _species_lengths(species_names: list[str], species_lengths: list[int] | np.ndarray | None) -> np.ndarray:
    if species_lengths is None:
        return np.asarray([len(name) for name in species_names], dtype=np.int64)
    lengths = np.asarray(species_lengths, dtype=np.int64)
    if lengths.shape != (len(species_names),):
        raise ValueError(f"species_lengths must have shape ({len(species_names)},)")
    return lengths


def _node_radii(
    values: np.ndarray,
    max_node_radius: float,
    radius_mode: str,
    radius_transform: Callable[[np.ndarray], np.ndarray] | None,
    scale_reference: float | None = None,
) -> np.ndarray:
    state = np.maximum(np.asarray(values, dtype=float), 0.0)
    if radius_transform is not None:
        radii = np.asarray(radius_transform(state), dtype=float)
        if radii.shape != state.shape:
            raise ValueError("radius_transform must return one radius per species")
        if np.any(~np.isfinite(radii)) or np.any(radii < 0.0):
            raise ValueError("radius_transform returned invalid radii")
        return radii

    scale = float(max_node_radius)
    if scale < 0.0:
        raise ValueError("max_node_radius must be >= 0")
    largest = float(scale_reference) if scale_reference is not None else float(np.max(state)) if state.size else 0.0
    if largest <= 0.0 or scale == 0.0:
        return np.zeros_like(state, dtype=float)

    normalized = state / largest
    mode = radius_mode.lower()
    if mode == "area":
        return scale * np.sqrt(normalized)
    if mode == "radius":
        return scale * normalized
    raise ValueError("radius_mode must be 'area' or 'radius'")


def _radial_species_layout(
    species_names: list[str],
    lengths: np.ndarray,
    node_radii: np.ndarray,
    spacing_area_factor: float,
) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    if lengths.shape != node_radii.shape:
        raise ValueError("lengths and node_radii must have the same shape")
    factor = float(spacing_area_factor)
    if factor < 0.0:
        raise ValueError("spacing_area_factor must be >= 0")

    global_max_radius = float(np.max(node_radii)) if node_radii.size else 0.0
    min_gap = max(global_max_radius * 0.8, 0.08)
    positions: dict[int, tuple[float, float]] = {}
    layer_radii: dict[int, float] = {}
    previous_radius = 0.0
    previous_area = 0.0
    previous_node_radius = 0.0

    for layer_index, length in enumerate(sorted(int(item) for item in np.unique(lengths))):
        layer_indices = np.flatnonzero(lengths == length).astype(np.int64, copy=False)
        ordered = np.asarray(sorted(layer_indices, key=lambda sid: species_names[int(sid)]), dtype=np.int64)
        layer_node_radius = float(np.max(node_radii[ordered])) if ordered.size else 0.0
        layer_area = float(np.pi * np.sum(node_radii[ordered] ** 2))
        circumference_radius = _minimum_layer_radius(int(ordered.size), layer_node_radius, min_gap)

        if layer_index == 0:
            layer_radius = 0.0 if ordered.size == 1 else max(circumference_radius, layer_node_radius + min_gap)
        else:
            area_radius = float(np.sqrt(previous_radius**2 + factor * previous_area / np.pi))
            non_overlap_radius = previous_radius + previous_node_radius + layer_node_radius + min_gap
            layer_radius = max(area_radius, non_overlap_radius, circumference_radius)

        layer_radii[length] = layer_radius
        if ordered.size == 1 and layer_radius == 0.0:
            positions[int(ordered[0])] = (0.0, 0.0)
        else:
            angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - 2.0 * np.pi, ordered.size, endpoint=False)
            for sid, angle in zip(ordered, angles):
                positions[int(sid)] = (float(layer_radius * np.cos(angle)), float(layer_radius * np.sin(angle)))

        previous_radius = layer_radius
        previous_area = layer_area
        previous_node_radius = layer_node_radius

    return positions, layer_radii


def _minimum_layer_radius(n_nodes: int, node_radius: float, min_gap: float) -> float:
    if int(n_nodes) <= 1:
        return 0.0
    diameter_with_gap = 2.0 * float(node_radius) + float(min_gap)
    return max(float(n_nodes) * diameter_with_gap / (2.0 * np.pi), float(node_radius) + float(min_gap))


def _reaction_tree_edges(record: TrajectoryRecord, lengths: np.ndarray) -> list[tuple[int, int]]:
    n_species = len(record.species_names)
    edges: set[tuple[int, int]] = set()
    for item in record.run_metadata.get("channel_labels", []):
        if not isinstance(item, dict):
            continue
        reactants = _metadata_species_ids(item.get("reactants", ()), n_species)
        products = _metadata_species_ids(item.get("products", ()), n_species)
        for reactant in reactants:
            for product in products:
                if reactant == product:
                    continue
                if abs(int(lengths[reactant]) - int(lengths[product])) != 1:
                    continue
                edges.add(tuple(sorted((int(reactant), int(product)))))

    if not edges:
        edges = _prefix_tree_edges(record.species_names, lengths)

    return sorted(edges, key=lambda pair: (min(lengths[pair[0]], lengths[pair[1]]), pair[0], pair[1]))


def _metadata_species_ids(values, n_species: int) -> list[int]:
    result = []
    for value in values or ():
        sid = int(value)
        if 0 <= sid < int(n_species):
            result.append(sid)
    return result


def _prefix_tree_edges(species_names: list[str], lengths: np.ndarray) -> set[tuple[int, int]]:
    name_to_idx = {name: idx for idx, name in enumerate(species_names)}
    edges: set[tuple[int, int]] = set()
    for sid, name in enumerate(species_names):
        if int(lengths[sid]) <= int(np.min(lengths)):
            continue
        parent_names = {name[:-1], name[1:]}
        for parent_name in parent_names:
            parent = name_to_idx.get(parent_name)
            if parent is None or abs(int(lengths[sid]) - int(lengths[parent])) != 1:
                continue
            edges.add(tuple(sorted((int(parent), int(sid)))))
    return edges


def _set_equal_tree_limits(
    ax,
    positions: dict[int, tuple[float, float]],
    node_radii: np.ndarray,
) -> None:
    coords = np.asarray(list(positions.values()), dtype=float)
    if coords.size == 0:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        return
    padding = max(float(np.max(node_radii)) if node_radii.size else 0.0, 0.1) * 2.5
    x_min = float(np.min(coords[:, 0]) - padding)
    x_max = float(np.max(coords[:, 0]) + padding)
    y_min = float(np.min(coords[:, 1]) - padding)
    y_max = float(np.max(coords[:, 1]) + padding)
    half_width = max(x_max - x_min, y_max - y_min) / 2.0
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    ax.set_xlim(center_x - half_width, center_x + half_width)
    ax.set_ylim(center_y - half_width, center_y + half_width)


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
