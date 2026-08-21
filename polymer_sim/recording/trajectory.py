"""单次完整轨迹记录的数据结构与离线保存/读取接口。

本模块专门服务于“需要完整轨迹”的路径。默认模拟流程不强制调用这里的保存逻辑，
从而满足“ExperimentRunner 默认不保存完整轨迹”的要求。

完整轨迹遵循“一个文件只存一次完整记录”的原则：一个 `.npz` 文件对应一次运行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np

from polymer_sim.recording.base import BaseRecorder, BaseTrajectoryRecord, PathLike


SIDECAR_FORMAT = "polymer_sim_trajectory_sidecar_v1"
SIDECAR_TIMES_NAME = "times.npy"
SIDECAR_STATES_NAME = "states.npy"
SIDECAR_SPECIES_NAMES_NAME = "species_names.json"
SIDECAR_METADATA_NAME = "metadata.json"


def _as_float_array_preserve_mmap(value: np.ndarray) -> np.ndarray:
    """Convert array-like values to floating arrays without materializing mmap arrays."""

    if isinstance(value, np.memmap):
        if np.issubdtype(value.dtype, np.floating):
            return value
        return np.asarray(value, dtype=float)
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.floating):
        return arr
    return arr.astype(float)


@dataclass(slots=True)
class DTStatistics:
    """Lightweight accepted-step interval statistics from a trajectory file."""

    count: int
    total_time: float
    min: float
    max: float
    mean: float
    median: float
    std: float
    histogram_counts: np.ndarray
    histogram_edges: np.ndarray
    plot_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "total_time": float(self.total_time),
            "min": float(self.min),
            "max": float(self.max),
            "mean": float(self.mean),
            "median": float(self.median),
            "std": float(self.std),
            "histogram_counts": np.asarray(self.histogram_counts, dtype=np.int64).tolist(),
            "histogram_edges": np.asarray(self.histogram_edges, dtype=float).tolist(),
            "plot_path": self.plot_path,
        }


@dataclass(slots=True)
class TrajectoryRecord(BaseTrajectoryRecord):
    """单次运行的完整轨迹记录。

    字段约定：

    - `times.shape == (T,)`
    - `states.shape == (T, n_species)`
    - `len(species_names) == n_species`

    `run_metadata` 预留给 seed、stepper 类型、参数标签等后续扩展信息。
    """

    times: np.ndarray
    states: np.ndarray
    species_names: list[str]
    run_metadata: dict
    accepted_step_intervals: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.times = _as_float_array_preserve_mmap(self.times)
        self.states = _as_float_array_preserve_mmap(self.states)
        if self.accepted_step_intervals is not None:
            self.accepted_step_intervals = _as_float_array_preserve_mmap(self.accepted_step_intervals)
            if self.accepted_step_intervals.ndim != 1:
                raise ValueError("accepted_step_intervals must have shape (T - 1,)")
        if self.times.ndim != 1:
            raise ValueError("times must have shape (T,)")
        if self.states.ndim != 2:
            raise ValueError("states must have shape (T, n_species)")
        if self.states.shape[0] != self.times.shape[0]:
            raise ValueError("states.shape[0] must match times.shape[0]")
        if self.states.shape[1] != len(self.species_names):
            raise ValueError("states.shape[1] must match len(species_names)")

    @staticmethod
    def dt_statistics(
        path: PathLike,
        *,
        bins: int | str | np.ndarray = 50,
        plot_path: PathLike | None = None,
        x_log: bool = False,
        y_log: bool = False,
    ) -> DTStatistics:
        """Load only accepted-step intervals from a saved trajectory and summarize them.

        This is intentionally file-based and lightweight: it reads the
        ``accepted_step_intervals`` array from the npz without materializing the
        potentially large ``states`` matrix.  Older files without that array fall
        back to reading only ``times`` and computing ``np.diff(times)``.
        """

        return trajectory_dt_statistics(path, bins=bins, plot_path=plot_path, x_log=x_log, y_log=y_log)


class TrajectoryRecorder(BaseRecorder):
    """完整轨迹 recorder。

    该 recorder 会保存所有记录时刻的完整状态，因此不作为默认路径使用。
    当用户明确需要完整轨迹离线分析时，再显式启用并单独保存为一个文件。
    """

    def __init__(self):
        self._species_names: list[str] = []
        self._times: list[float] = []
        self._states: list[np.ndarray] = []
        self._metadata: dict = {}
        self._accepted_step_intervals: list[float] = []
        self._channel_trigger_counts: np.ndarray | None = None
        self._channel_continuous_trigger_counts: np.ndarray | None = None
        self._channel_event_times: list[float] = []
        self._channel_event_ids: list[int] = []
        self._reaction_intervals: list[float] = []
        self._reaction_interval_times: list[float] = []
        self._last_reaction_event_time: float = 0.0
        self._tracked_outflow_species: list[str] = []
        self._tracked_outflow_channel_to_col: dict[int, int] = {}
        self._tracked_outflow_times: list[float] = []
        self._tracked_outflow_removed: list[list[float]] = []

    def initialize(self, species_names: list[str], initial_state: np.ndarray, metadata: dict | None = None) -> None:
        self._species_names = list(species_names)
        self._times = [0.0]
        self._states = [np.asarray(initial_state, dtype=float).copy()]
        self._metadata = dict(metadata or {})
        self._accepted_step_intervals = []
        n_channels = self._metadata.get("n_channels")
        self._channel_trigger_counts = (
            np.zeros(int(n_channels), dtype=np.int64) if n_channels is not None else None
        )
        self._channel_continuous_trigger_counts = (
            np.zeros(int(n_channels), dtype=float) if n_channels is not None else None
        )
        self._channel_event_times = []
        self._channel_event_ids = []
        self._reaction_intervals = []
        self._reaction_interval_times = []
        self._last_reaction_event_time = 0.0
        self._tracked_outflow_species = []
        self._tracked_outflow_channel_to_col = {}
        self._tracked_outflow_times = []
        self._tracked_outflow_removed = []
        for channel_label in self._metadata.get("channel_labels", []):
            if channel_label.get("block_type") != "OUTFLOW":
                continue
            reactants = channel_label.get("reactants", ())
            if not reactants:
                continue
            source_sid = int(reactants[0])
            source_name = self._species_names[source_sid]
            if source_name not in self._tracked_outflow_species:
                self._tracked_outflow_species.append(source_name)
            self._tracked_outflow_channel_to_col[int(channel_label["channel_id"])] = self._tracked_outflow_species.index(source_name)

    def record_step(
        self,
        *,
        time: float,
        state: np.ndarray,
        step_count: int,
        event_count: int,
        event_time: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        current_time = float(time)
        previous_time = float(self._times[-1]) if self._times else 0.0
        self._accepted_step_intervals.append(float(max(current_time - previous_time, 0.0)))
        self._times.append(current_time)
        self._states.append(np.asarray(state, dtype=float).copy())
        step_metadata = dict(metadata or {})
        continuous_increments = step_metadata.pop("continuous_channel_abs_increments", None)
        if continuous_increments is not None and self._channel_continuous_trigger_counts is not None:
            increments = np.asarray(continuous_increments, dtype=float)
            if increments.shape != self._channel_continuous_trigger_counts.shape:
                raise ValueError("continuous_channel_abs_increments shape does not match n_channels")
            if not np.all(np.isfinite(increments)):
                raise ValueError("continuous_channel_abs_increments contains non-finite values")
            self._channel_continuous_trigger_counts += np.maximum(increments, 0.0)
        discrete_event_ids = step_metadata.pop("discrete_event_ids", None)
        discrete_event_times = step_metadata.pop("discrete_event_times", None)
        channel_id = step_metadata.get("channel_id")
        if discrete_event_ids is not None:
            event_ids = [int(cid) for cid in np.asarray(discrete_event_ids, dtype=np.int64).tolist()]
        elif channel_id is not None:
            event_ids = [int(channel_id)]
        else:
            event_ids = []
        if discrete_event_times is not None:
            event_times = [float(value) for value in np.asarray(discrete_event_times, dtype=float).tolist()]
            if len(event_times) != len(event_ids):
                raise ValueError("discrete_event_times must have the same length as discrete_event_ids")
        elif event_ids:
            event_times = [float(event_time if event_time is not None else time) for _ in event_ids]
        else:
            event_times = []
        is_reaction_event = bool(event_ids) or event_time is not None
        if step_metadata:
            self._metadata.update(step_metadata)
        if is_reaction_event:
            event_timestamp = float(event_time if event_time is not None else time)
            if event_ids:
                for cid, event_timestamp in zip(event_ids, event_times):
                    if self._channel_trigger_counts is not None:
                        self._channel_trigger_counts[int(cid)] += 1
                        self._channel_event_times.append(float(event_timestamp))
                        self._channel_event_ids.append(int(cid))
                    interval = float(event_timestamp) - self._last_reaction_event_time
                    self._reaction_intervals.append(float(max(interval, 0.0)))
                    self._reaction_interval_times.append(float(event_timestamp))
                    self._last_reaction_event_time = float(event_timestamp)
            else:
                interval = event_timestamp - self._last_reaction_event_time
                self._reaction_intervals.append(float(max(interval, 0.0)))
                self._reaction_interval_times.append(event_timestamp)
                self._last_reaction_event_time = event_timestamp
        if self._tracked_outflow_species:
            outflow_row = [0.0 for _ in self._tracked_outflow_species]
            for cid in event_ids:
                if int(cid) in self._tracked_outflow_channel_to_col:
                    outflow_row[self._tracked_outflow_channel_to_col[int(cid)]] += 1.0
            self._tracked_outflow_times.append(float(time))
            self._tracked_outflow_removed.append(outflow_row)
        self._metadata["n_steps"] = int(step_count)
        self._metadata["n_events"] = int(event_count)

    def finalize(self) -> TrajectoryRecord:
        if self._channel_trigger_counts is not None:
            self._metadata["channel_trigger_counts"] = self._channel_trigger_counts.tolist()
        if self._channel_continuous_trigger_counts is not None:
            self._metadata["channel_continuous_trigger_counts"] = self._channel_continuous_trigger_counts.tolist()
        self._metadata["channel_event_times"] = list(self._channel_event_times)
        self._metadata["channel_event_ids"] = list(self._channel_event_ids)
        self._metadata["reaction_intervals"] = list(self._reaction_intervals)
        self._metadata["reaction_interval_times"] = list(self._reaction_interval_times)
        if self._tracked_outflow_times:
            self._metadata["tracked_outflow"] = {
                "times": list(self._tracked_outflow_times),
                "species_names": list(self._tracked_outflow_species),
                "removed": [list(row) for row in self._tracked_outflow_removed],
            }
        return TrajectoryRecord(
            times=np.asarray(self._times, dtype=float),
            states=np.vstack(self._states) if self._states else np.empty((0, 0), dtype=float),
            species_names=list(self._species_names),
            run_metadata=dict(self._metadata),
            accepted_step_intervals=np.asarray(self._accepted_step_intervals, dtype=float),
        )


def save_trajectory_record(path: PathLike, record: TrajectoryRecord) -> None:
    """保存单次完整轨迹记录到压缩 npz 文件。"""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path_obj,
        times=record.times,
        states=record.states,
        accepted_step_intervals=(
            np.asarray(record.accepted_step_intervals, dtype=float)
            if record.accepted_step_intervals is not None
            else np.diff(np.asarray(record.times, dtype=float))
        ),
        species_names=np.asarray(record.species_names, dtype=object),
        run_metadata_json=json.dumps(record.run_metadata, ensure_ascii=True),
    )


def load_trajectory_record(path: PathLike) -> TrajectoryRecord:
    """从压缩 npz 文件读取单次完整轨迹记录。"""

    with np.load(Path(path), allow_pickle=True) as data:
        metadata = json.loads(str(data["run_metadata_json"]))
        return TrajectoryRecord(
            times=np.asarray(data["times"], dtype=float),
            states=np.asarray(data["states"], dtype=float),
            species_names=[str(name) for name in data["species_names"].tolist()],
            run_metadata=metadata,
            accepted_step_intervals=(
                np.asarray(data["accepted_step_intervals"], dtype=float)
                if "accepted_step_intervals" in data.files
                else np.diff(np.asarray(data["times"], dtype=float))
            ),
        )


def trajectory_dt_statistics(
    path: PathLike,
    *,
    bins: int | str | np.ndarray = 50,
    plot_path: PathLike | None = None,
    x_log: bool = False,
    y_log: bool = False,
) -> DTStatistics:
    """Summarize accepted-step intervals from a trajectory npz without loading states."""

    intervals = _load_accepted_step_intervals(path)
    intervals = intervals[np.isfinite(intervals)]
    histogram_intervals = intervals[intervals > 0.0] if x_log else intervals
    if intervals.size == 0:
        counts = np.zeros(0, dtype=np.int64)
        edges = np.zeros(0, dtype=float)
        stats = DTStatistics(
            count=0,
            total_time=0.0,
            min=float("nan"),
            max=float("nan"),
            mean=float("nan"),
            median=float("nan"),
            std=float("nan"),
            histogram_counts=counts,
            histogram_edges=edges,
        )
    else:
        counts, edges = _dt_histogram(histogram_intervals, bins=bins, x_log=bool(x_log))
        stats = DTStatistics(
            count=int(intervals.size),
            total_time=float(np.sum(intervals)),
            min=float(np.min(intervals)),
            max=float(np.max(intervals)),
            mean=float(np.mean(intervals)),
            median=float(np.median(intervals)),
            std=float(np.std(intervals)),
            histogram_counts=np.asarray(counts, dtype=np.int64),
            histogram_edges=np.asarray(edges, dtype=float),
        )
    if plot_path is not None:
        stats.plot_path = str(_save_dt_statistics_bar_plot(plot_path, stats, x_log=bool(x_log), y_log=bool(y_log)))
    return stats


def _load_accepted_step_intervals(path: PathLike) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as data:
        if "accepted_step_intervals" in data.files:
            intervals = np.asarray(data["accepted_step_intervals"], dtype=float)
        else:
            times = np.asarray(data["times"], dtype=float)
            intervals = np.diff(times)
    if intervals.ndim != 1:
        raise ValueError("accepted_step_intervals must have shape (T - 1,)")
    return intervals


def _dt_histogram(intervals: np.ndarray, *, bins: int | str | np.ndarray, x_log: bool) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(intervals, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)
    if not x_log:
        return np.histogram(values, bins=bins)

    if isinstance(bins, str):
        bin_count = 50
    elif np.isscalar(bins):
        bin_count = max(int(bins), 1)
    else:
        edges = np.asarray(bins, dtype=float)
        if np.any(edges <= 0.0):
            raise ValueError("log-x histogram bin edges must be positive")
        return np.histogram(values, bins=edges)

    min_dt = float(np.min(values))
    max_dt = float(np.max(values))
    if min_dt <= 0.0:
        raise ValueError("log-x dt histogram requires positive intervals")
    if min_dt == max_dt:
        lower = min_dt / np.sqrt(10.0)
        upper = max_dt * np.sqrt(10.0)
    else:
        lower = min_dt
        upper = max_dt
    edges = np.logspace(np.log10(lower), np.log10(upper), bin_count + 1)
    return np.histogram(values, bins=edges)


def _save_dt_statistics_bar_plot(path: PathLike, stats: DTStatistics, *, x_log: bool, y_log: bool) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    counts = np.asarray(stats.histogram_counts, dtype=float)
    edges = np.asarray(stats.histogram_edges, dtype=float)
    if counts.size and edges.size == counts.size + 1:
        widths = np.diff(edges)
        ax.bar(edges[:-1], counts, width=widths, align="edge", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("accepted step interval dt")
    ax.set_ylabel("count")
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def trajectory_final_time(path: PathLike) -> float:
    """Return the final simulation time from a saved trajectory npz.

    This lightweight inspection helper reads only the ``times`` array, so it
    avoids loading the potentially large ``states`` matrix.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=float)
        if times.ndim != 1:
            raise ValueError("trajectory times must have shape (T,)")
        if times.size == 0:
            raise ValueError("trajectory contains no time points")
        return float(times[-1])


# ---------------------------------------------------------------------------
# mmap sidecar trajectory I/O.
#
# These definitions intentionally appear after the legacy functions above so
# they preserve the public names while adding the new storage path without
# changing call sites.  Existing ``.npz`` files remain readable; newly saved
# records write both the legacy ``.npz`` and a same-name sidecar directory.
# ---------------------------------------------------------------------------


def trajectory_sidecar_dir(path: PathLike) -> Path:
    """Return the default sidecar directory for a trajectory path.

    ``run_001.npz`` maps to ``run_001/``. Passing an existing directory returns
    it unchanged, which lets offline tools accept either path style.
    """

    path_obj = Path(path)
    if path_obj.is_dir():
        return path_obj
    if path_obj.suffix:
        return path_obj.with_suffix("")
    return path_obj


def has_trajectory_sidecar(path: PathLike) -> bool:
    """Return True when the mmap-friendly sidecar trajectory exists."""

    sidecar = trajectory_sidecar_dir(path)
    return (
        sidecar.is_dir()
        and (sidecar / SIDECAR_TIMES_NAME).exists()
        and (sidecar / SIDECAR_STATES_NAME).exists()
        and (sidecar / SIDECAR_SPECIES_NAMES_NAME).exists()
        and (sidecar / SIDECAR_METADATA_NAME).exists()
    )


def save_trajectory_sidecar(path: PathLike, record: TrajectoryRecord) -> Path:
    """Write the mmap-friendly sidecar representation and return its directory."""

    sidecar = trajectory_sidecar_dir(path)
    sidecar.mkdir(parents=True, exist_ok=True)
    np.save(sidecar / SIDECAR_TIMES_NAME, np.asarray(record.times, dtype=float))
    np.save(sidecar / SIDECAR_STATES_NAME, np.asarray(record.states, dtype=float))
    (sidecar / SIDECAR_SPECIES_NAMES_NAME).write_text(
        json.dumps(list(record.species_names), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    metadata_payload = {
        "format": SIDECAR_FORMAT,
        "run_metadata": record.run_metadata,
        "times_shape": list(np.asarray(record.times).shape),
        "states_shape": list(np.asarray(record.states).shape),
        "accepted_step_intervals_source": "times_diff",
    }
    (sidecar / SIDECAR_METADATA_NAME).write_text(
        json.dumps(metadata_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return sidecar


def save_trajectory_record(path: PathLike, record: TrajectoryRecord, *, write_sidecar: bool = True) -> None:
    """Save one complete trajectory.

    The legacy compressed ``.npz`` file is still written for compatibility.
    By default, a same-name sidecar directory is also written:
    ``times.npy``, ``states.npy``, ``species_names.json`` and ``metadata.json``.
    The sidecar arrays are uncompressed ``.npy`` files so readers can use mmap
    and sample a few rows without loading the full state matrix.
    """

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path_obj,
        times=record.times,
        states=record.states,
        accepted_step_intervals=(
            np.asarray(record.accepted_step_intervals, dtype=float)
            if record.accepted_step_intervals is not None
            else np.diff(np.asarray(record.times, dtype=float))
        ),
        species_names=np.asarray(record.species_names, dtype=object),
        run_metadata_json=json.dumps(record.run_metadata, ensure_ascii=True),
    )
    if write_sidecar:
        save_trajectory_sidecar(path_obj, record)


def load_trajectory_arrays(
    path: PathLike,
    *,
    mmap: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Load trajectory arrays and metadata.

    If a sidecar exists, ``mmap=None`` and ``mmap=True`` load ``times.npy`` and
    ``states.npy`` with ``mmap_mode='r'``. ``mmap=False`` forces eager loading.
    If no sidecar exists, this falls back to the legacy compressed ``.npz``.
    """

    path_obj = Path(path)
    prefer_sidecar = mmap is not False
    if path_obj.is_dir() or (prefer_sidecar and has_trajectory_sidecar(path_obj)):
        sidecar = trajectory_sidecar_dir(path_obj)
        mmap_mode = "r" if mmap is not False else None
        times = np.load(sidecar / SIDECAR_TIMES_NAME, mmap_mode=mmap_mode, allow_pickle=False)
        states = np.load(sidecar / SIDECAR_STATES_NAME, mmap_mode=mmap_mode, allow_pickle=False)
        species_names = json.loads((sidecar / SIDECAR_SPECIES_NAMES_NAME).read_text(encoding="utf-8"))
        metadata_payload = json.loads((sidecar / SIDECAR_METADATA_NAME).read_text(encoding="utf-8"))
        metadata = metadata_payload.get("run_metadata", metadata_payload)
        return times, states, [str(name) for name in species_names], dict(metadata)

    with np.load(path_obj, allow_pickle=True) as data:
        metadata = json.loads(str(data["run_metadata_json"]))
        return (
            np.asarray(data["times"], dtype=float),
            np.asarray(data["states"], dtype=float),
            [str(name) for name in data["species_names"].tolist()],
            metadata,
        )


def load_trajectory_record(path: PathLike, *, mmap: bool | None = None) -> TrajectoryRecord:
    """Load one complete trajectory from sidecar/mmap or legacy npz."""

    path_obj = Path(path)
    if path_obj.is_dir() or (mmap is not False and has_trajectory_sidecar(path_obj)):
        times, states, species_names, metadata = load_trajectory_arrays(path_obj, mmap=mmap)
        return TrajectoryRecord(
            times=times,
            states=states,
            species_names=species_names,
            run_metadata=metadata,
            accepted_step_intervals=np.diff(np.asarray(times, dtype=float)),
        )

    with np.load(path_obj, allow_pickle=True) as data:
        metadata = json.loads(str(data["run_metadata_json"]))
        times = np.asarray(data["times"], dtype=float)
        return TrajectoryRecord(
            times=times,
            states=np.asarray(data["states"], dtype=float),
            species_names=[str(name) for name in data["species_names"].tolist()],
            run_metadata=metadata,
            accepted_step_intervals=(
                np.asarray(data["accepted_step_intervals"], dtype=float)
                if "accepted_step_intervals" in data.files
                else np.diff(times)
            ),
        )


def _load_accepted_step_intervals(path: PathLike) -> np.ndarray:
    """Load accepted-step intervals without materializing the full state matrix."""

    if Path(path).is_dir() or has_trajectory_sidecar(path):
        times = np.load(trajectory_sidecar_dir(path) / SIDECAR_TIMES_NAME, mmap_mode="r", allow_pickle=False)
        intervals = np.diff(np.asarray(times, dtype=float))
    else:
        with np.load(Path(path), allow_pickle=False) as data:
            if "accepted_step_intervals" in data.files:
                intervals = np.asarray(data["accepted_step_intervals"], dtype=float)
            else:
                intervals = np.diff(np.asarray(data["times"], dtype=float))
    if intervals.ndim != 1:
        raise ValueError("accepted_step_intervals must have shape (T - 1,)")
    return intervals


def trajectory_final_time(path: PathLike) -> float:
    """Return the final simulation time without loading the full state matrix."""

    if Path(path).is_dir() or has_trajectory_sidecar(path):
        times = np.load(trajectory_sidecar_dir(path) / SIDECAR_TIMES_NAME, mmap_mode="r", allow_pickle=False)
    else:
        with np.load(Path(path), allow_pickle=False) as data:
            times = np.asarray(data["times"], dtype=float)
    if times.ndim != 1:
        raise ValueError("trajectory times must have shape (T,)")
    if times.size == 0:
        raise ValueError("trajectory contains no time points")
    return float(times[-1])


def sample_trajectory_states_from_path(
    path: PathLike,
    time_points: np.ndarray,
    *,
    mmap: bool | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Sample trajectory states at requested times without loading all states.

    The returned state array has shape ``(len(time_points), n_species)``. With a
    sidecar, only selected rows of ``states.npy`` are read from disk.
    """

    times, states, species_names, metadata = load_trajectory_arrays(path, mmap=mmap)
    t = np.asarray(times, dtype=float)
    points = np.asarray(time_points, dtype=float)
    if t.ndim != 1 or states.ndim != 2 or states.shape[0] != t.shape[0]:
        raise ValueError("invalid trajectory arrays")
    result = np.full((points.size, states.shape[1]), np.nan, dtype=float)
    if t.size:
        indices = np.searchsorted(t, points, side="right") - 1
        valid = (indices >= 0) & (points <= t[-1] + 1e-12)
        if np.any(valid):
            result[valid] = np.asarray(states[indices[valid]], dtype=float)
    return result, species_names, {
        "storage": "sidecar" if has_trajectory_sidecar(path) else "npz",
        "mmap": bool(mmap is not False and has_trajectory_sidecar(path)),
        "times_shape": tuple(t.shape),
        "states_shape": tuple(states.shape),
        "metadata": metadata,
    }
