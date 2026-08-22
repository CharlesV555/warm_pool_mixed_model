"""默认轻量 summary 记录路径。

本模块是 recording 层的默认工作路径：

1. 模拟运行期间默认只累计轻量信息。
2. 运行结束后生成 `RunSummary`。
3. 多次运行可以在一个文件中集中保存，支持离线读取和后续统计/绘图。

这样可以避免默认保存完整轨迹带来的额外内存与 I/O 成本。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from polymer_sim.recording.base import BaseRecorder, BaseRunSummary, PathLike
from polymer_sim.recording.trajectory import load_trajectory_arrays, trajectory_storage_exists


@dataclass(slots=True)
class RunSummary(BaseRunSummary):
    """单次模拟的轻量结果摘要。

    必备字段用于复现和聚合统计：

    - `final_time`
    - `final_state.shape == (n_species,)`
    - `n_steps`
    - `n_events`
    - `metadata`

    第一轮额外保留 `species_names` 与可选 `event_times`，便于离线分析与绘图。
    """

    final_time: float
    final_state: np.ndarray
    n_steps: int
    n_events: int
    metadata: dict = field(default_factory=dict)
    species_names: list[str] = field(default_factory=list)
    event_times: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.final_state = np.asarray(self.final_state, dtype=float)
        if self.final_state.ndim != 1:
            raise ValueError("final_state must have shape (n_species,)")
        if self.species_names and len(self.species_names) != self.final_state.shape[0]:
            raise ValueError("len(species_names) must match final_state.shape[0]")
        if self.event_times is not None:
            self.event_times = np.asarray(self.event_times, dtype=float)
            if self.event_times.ndim != 1:
                raise ValueError("event_times must have shape (n_events,) when provided")


@dataclass(slots=True)
class BatchRunItem:
    index: int
    data: dict
    source_path: Path | None = None

    @property
    def mode(self) -> str:
        return str(self.data.get("mode", self.data.get("stepper_method", "run")))

    @property
    def seed(self) -> int | None:
        value = self.data.get("seed", self.data.get("pair_seed"))
        return None if value is None else int(value)

    @property
    def simulation_time(self) -> float:
        for key in ("simulation_final_time", "final_time", "requested_t_end"):
            if key in self.data:
                return float(self.data[key])
        raise KeyError("run item does not contain simulation_final_time or final_time")

    @property
    def requested_t_end(self) -> float | None:
        value = self.data.get("requested_t_end")
        return None if value is None else float(value)

    def label(self) -> str:
        if "pair_order" in self.data:
            return f"{self.mode} pair={int(self.data['pair_order'])}"
        if "run_index" in self.data:
            return f"{self.mode} run={int(self.data['run_index'])}"
        return f"{self.mode} index={self.index}"

    def scale(self, *, width: int = 60, print_output: bool = True) -> str:
        return BatchRunSelection([self]).scale(width=width, print_output=print_output)

    def mol_num(
        self,
        time_points: Iterable[float],
        *,
        print_output: bool = True,
    ) -> dict[str, object]:
        return BatchRunSelection([self]).mol_num(time_points, print_output=print_output)

    def dt_compare(
        self,
        *,
        method: str | None = "blended",
        ax=None,
        cmap: str = "viridis",
        annotate: bool = True,
        annotation_format: str = ".4g",
        title: str | None = None,
        save_output: bool = True,
        output_dir: PathLike | None = None,
        figure_filename: str = "dt_compare_heatmap.png",
        table_filename: str = "dt_compare_report.csv",
        dpi: int = 200,
    ):
        return BatchRunSelection([self]).dt_compare(
            method=method,
            ax=ax,
            cmap=cmap,
            annotate=annotate,
            annotation_format=annotation_format,
            title=title,
            save_output=save_output,
            output_dir=output_dir,
            figure_filename=figure_filename,
            table_filename=table_filename,
            dpi=dpi,
        )


@dataclass(slots=True)
class BatchRunSelection:
    items: list[BatchRunItem]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return _select_items(self.items, index)

    def modes(self) -> dict[str, int]:
        return _mode_counts(self.items)

    def where(self, *, mode: str | None = None, stop_reason: str | None = None) -> "BatchRunSelection":
        return BatchRunSelection(_filter_items(self.items, mode=mode, stop_reason=stop_reason))

    def scale(
        self,
        *,
        method: str | None = None,
        width: int = 60,
        t_min: float | None = None,
        t_max: float | None = None,
        print_output: bool = True,
    ) -> str:
        """Return a character scatter plot of selected simulation times."""

        selected_items = self.items if method is None else _filter_items(self.items, mode=method)
        if not selected_items:
            text = "empty selection"
            if print_output:
                print(text)
            return text

        values = np.asarray([item.simulation_time for item in selected_items], dtype=float)
        requested = [
            item.requested_t_end
            for item in selected_items
            if item.requested_t_end is not None and np.isfinite(item.requested_t_end)
        ]
        left = 0.0 if t_min is None else float(t_min)
        if t_max is None:
            right = max(float(values.max()), max(requested) if requested else float(values.max()))
        else:
            right = float(t_max)
        if not np.isfinite(left) or not np.isfinite(right):
            raise ValueError("t_min and t_max must be finite")
        if right <= left:
            right = left + 1.0

        plot_width = max(int(width), 10)
        lines = [
            "simulation_time scale",
            f"n={len(selected_items)}, range=[{left:.6g}, {right:.6g}]",
            f"{left:.6g} |{'-' * plot_width}| {right:.6g}",
        ]
        for item, value in zip(selected_items, values):
            marker = _scatter_marker(value, left, right, plot_width)
            seed_text = "" if item.seed is None else f" seed={item.seed}"
            lines.append(f"{item.index:04d} {item.label():<22} {value:>12.6g} |{marker}|{seed_text}")

        text = "\n".join(lines)
        if print_output:
            print(text)
        return text

    def mol_num(
        self,
        time_points: Iterable[float],
        *,
        method: str | None = None,
        print_output: bool = True,
    ) -> dict[str, object]:
        """Read trajectories and report total molecule counts at time points."""

        selected_items = self.items if method is None else _filter_items(self.items, mode=method)
        payload = _molecule_number_payload(selected_items, time_points)
        if print_output:
            print(_format_molecule_number_payload(payload))
        return payload

    def dt_compare(
        self,
        *,
        method: str | None = "blended",
        ax=None,
        cmap: str = "viridis",
        annotate: bool = True,
        annotation_format: str = ".4g",
        title: str | None = None,
        save_output: bool = True,
        output_dir: PathLike | None = None,
        figure_filename: str = "dt_compare_heatmap.png",
        table_filename: str = "dt_compare_report.csv",
        dpi: int = 200,
    ):
        """Plot mean simulation time over the blended dt_cle x dt_macro grid."""

        selected_items = self.items if method is None else _filter_items(self.items, mode=method)
        return _plot_dt_compare(
            selected_items,
            ax=ax,
            cmap=cmap,
            annotate=annotate,
            annotation_format=annotation_format,
            title=title,
            save_output=save_output,
            output_dir=output_dir,
            figure_filename=figure_filename,
            table_filename=table_filename,
            dpi=dpi,
        )


@dataclass(slots=True)
class BatchSummary:
    path: Path
    raw: dict | list
    shared: dict
    runs: list[BatchRunItem]

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def __getitem__(self, index):
        return _select_items(self.runs, index)

    def __repr__(self) -> str:
        modes = ", ".join(f"{mode}={count}" for mode, count in sorted(self.modes().items()))
        return f"BatchSummary(n_runs={len(self)}, modes={{ {modes} }}, path='{self.path}')"

    def modes(self) -> dict[str, int]:
        return _mode_counts(self.runs)

    def where(self, *, mode: str | None = None, stop_reason: str | None = None) -> BatchRunSelection:
        return BatchRunSelection(_filter_items(self.runs, mode=mode, stop_reason=stop_reason))

    def scale(
        self,
        indices: Iterable[int] | slice | int | None = None,
        *,
        method: str | None = None,
        width: int = 60,
        print_output: bool = True,
    ) -> str:
        if method is None:
            # No method filter: indices refer to the original file order.
            selected = self.runs if indices is None else _selection_to_list(_select_items(self.runs, indices))
        else:
            # With a method filter: first filter by mode, then apply indices in
            # the filtered order.  For example, indices=[0] and method="ssa"
            # selects the first SSA run, not original file row 0.
            filtered = _filter_items(self.runs, mode=method)
            selected = filtered if indices is None else _selection_to_list(_select_items(filtered, indices))
        selection = BatchRunSelection(selected)
        return selection.scale(width=width, print_output=print_output)

    def mol_num(
        self,
        time_points: Iterable[float],
        indices: Iterable[int] | slice | int | None = None,
        *,
        method: str | None = None,
        print_output: bool = True,
    ) -> dict[str, object]:
        if method is None:
            # No method filter: indices refer to the original file order.
            selected = self.runs if indices is None else _selection_to_list(_select_items(self.runs, indices))
        else:
            # With a method filter: first filter by mode, then apply indices in
            # the filtered order.
            filtered = _filter_items(self.runs, mode=method)
            selected = filtered if indices is None else _selection_to_list(_select_items(filtered, indices))
        return BatchRunSelection(selected).mol_num(time_points, print_output=print_output)

    def dt_compare(
        self,
        indices: Iterable[int] | slice | int | None = None,
        *,
        method: str | None = "blended",
        ax=None,
        cmap: str = "viridis",
        annotate: bool = True,
        annotation_format: str = ".4g",
        title: str | None = None,
        save_output: bool = True,
        output_dir: PathLike | None = None,
        figure_filename: str = "dt_compare_heatmap.png",
        table_filename: str = "dt_compare_report.csv",
        dpi: int = 200,
    ):
        if method is None:
            # No method filter: indices refer to the original file order.
            selected = self.runs if indices is None else _selection_to_list(_select_items(self.runs, indices))
        else:
            # With a method filter: first filter by mode, then apply indices in
            # the filtered order, matching scale(...).
            filtered = _filter_items(self.runs, mode=method)
            selected = filtered if indices is None else _selection_to_list(_select_items(filtered, indices))
        return BatchRunSelection(selected).dt_compare(
            method=None,
            ax=ax,
            cmap=cmap,
            annotate=annotate,
            annotation_format=annotation_format,
            title=title,
            save_output=save_output,
            output_dir=output_dir,
            figure_filename=figure_filename,
            table_filename=table_filename,
            dpi=dpi,
        )

    def overview(self) -> str:
        times = np.asarray([item.simulation_time for item in self.runs], dtype=float)
        if times.size == 0:
            return f"BatchSummary(path='{self.path}', n_runs=0)"
        return (
            f"BatchSummary(path='{self.path}', n_runs={len(self)}, modes={self.modes()}, "
            f"simulation_time_min={times.min():.6g}, "
            f"simulation_time_mean={times.mean():.6g}, "
            f"simulation_time_max={times.max():.6g})"
        )


def format_stepper_info(metadata: dict | None) -> str:
    """Format stepper name and scalar parameters for console summaries."""

    data = dict(metadata or {})
    info = data.get("stepper_info")
    if not isinstance(info, dict):
        name = data.get("stepper_name") or data.get("stepper_method") or data.get("mode") or "unknown"
        return f"stepper={name}"

    name = str(info.get("name", data.get("stepper_name", "unknown")))
    parts = [f"stepper={name}"]
    config = info.get("config")
    if isinstance(config, dict) and config:
        parts.append(f"config={_compact_mapping(config)}")
    nrm_config = info.get("nrm_config")
    if isinstance(nrm_config, dict) and nrm_config:
        parts.append(f"nrm_config={_compact_mapping(nrm_config)}")
    return ", ".join(parts)


def _compact_mapping(values: dict) -> str:
    items = []
    for key in sorted(values):
        value = values[key]
        if value is None:
            text = "None"
        elif isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        items.append(f"{key}={text}")
    return "(" + ", ".join(items) + ")"


class SummaryRecorder(BaseRecorder):
    """默认 summary recorder。

    该 recorder 不保存完整状态轨迹，只在运行中累计最轻量的信息，最后产出
    一个 `RunSummary`。如需完整轨迹，应显式使用 `TrajectoryRecorder`。
    """

    def __init__(self, *, include_event_times: bool = False):
        self.include_event_times = bool(include_event_times)
        self._species_names: list[str] = []
        self._final_time: float = 0.0
        self._final_state: np.ndarray | None = None
        self._n_steps: int = 0
        self._n_events: int = 0
        self._metadata: dict = {}
        self._event_times: list[float] = []

    def initialize(self, species_names: list[str], initial_state: np.ndarray, metadata: dict | None = None) -> None:
        self._species_names = list(species_names)
        self._final_state = np.asarray(initial_state, dtype=float).copy()
        self._final_time = 0.0
        self._n_steps = 0
        self._n_events = 0
        self._metadata = dict(metadata or {})
        self._event_times = []

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
        self._final_time = float(time)
        self._final_state = np.asarray(state, dtype=float).copy()
        self._n_steps = int(step_count)
        self._n_events = int(event_count)
        discrete_event_times = None
        if metadata:
            step_metadata = dict(metadata)
            step_metadata.pop("continuous_channel_abs_increments", None)
            step_metadata.pop("discrete_event_ids", None)
            discrete_event_times = step_metadata.pop("discrete_event_times", None)
            self._metadata.update(step_metadata)
        if self.include_event_times:
            if discrete_event_times is not None:
                self._event_times.extend(float(value) for value in np.asarray(discrete_event_times, dtype=float))
            elif event_time is not None:
                self._event_times.append(float(event_time))

    def finalize(self) -> RunSummary:
        if self._final_state is None:
            raise RuntimeError("SummaryRecorder has not been initialized")
        return RunSummary(
            final_time=self._final_time,
            final_state=self._final_state,
            n_steps=self._n_steps,
            n_events=self._n_events,
            metadata=dict(self._metadata),
            species_names=list(self._species_names),
            event_times=np.asarray(self._event_times, dtype=float) if self.include_event_times else None,
        )


def save_summary(path: PathLike, summary: RunSummary | list[RunSummary]) -> None:
    """保存单个或多个 summary。

    一个 summary 文件可以聚合程序本次运行产生的多次模拟结果，便于离线统计。
    """

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    summaries = [summary] if isinstance(summary, RunSummary) else list(summary)
    payload = []
    for item in summaries:
        payload.append(
            {
                "final_time": float(item.final_time),
                "final_state": np.asarray(item.final_state, dtype=float).tolist(),
                "n_steps": int(item.n_steps),
                "n_events": int(item.n_events),
                "metadata": dict(item.metadata),
                "species_names": list(item.species_names),
                "event_times": None if item.event_times is None else np.asarray(item.event_times, dtype=float).tolist(),
            }
        )
    path_obj.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_summary(path: PathLike) -> RunSummary | list[RunSummary]:
    """读取单个或多个 summary。"""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    summaries = [
        RunSummary(
            final_time=float(item["final_time"]),
            final_state=np.asarray(item["final_state"], dtype=float),
            n_steps=int(item["n_steps"]),
            n_events=int(item["n_events"]),
            metadata=dict(item.get("metadata", {})),
            species_names=[str(name) for name in item.get("species_names", [])],
            event_times=None if item.get("event_times") is None else np.asarray(item["event_times"], dtype=float),
        )
        for item in raw
    ]
    return summaries[0] if len(summaries) == 1 else summaries


def load(path: PathLike) -> BatchSummary:
    """Load a lightweight batch summary wrapper.

    Supported inputs:
    - metadata JSON from examples/multiple_run.py paired runs, with a top-level
      ``runs`` list;
    - JSON files produced by ``save_summary(...)``.
    """

    path_obj = Path(path)
    raw = json.loads(path_obj.read_text(encoding="utf-8"))

    # Paired batch metadata file structure:
    # {
    #   "experiment": "paired_ssa_blended",
    #   "shared": {... common network/config/restriction metadata ...},
    #   "runs": [
    #       {
    #           "pair_order": int,
    #           "mode": "ssa" | "blended",
    #           "seed": int,
    #           "trajectory_path": str,
    #           "requested_t_end": float,
    #           "simulation_final_time": float,
    #           "wall_runtime_seconds": float,
    #           "n_steps": int,
    #           "n_events": int,
    #           "stop_reason": str,
    #           ...
    #       },
    #       ...
    #   ]
    # }
    #
    # load(...) keeps that raw JSON in BatchSummary.raw, stores the shared block
    # in BatchSummary.shared, and wraps each row in BatchRunItem(index, data,
    # source_path). BatchRunItem.index is the original row number in the file;
    # BatchRunItem.data is the unchanged per-run dictionary; source_path is the
    # metadata file path used to resolve relative trajectory_path values.
    if isinstance(raw, dict) and isinstance(raw.get("runs"), list):
        runs = [
            BatchRunItem(index=index, data=dict(item), source_path=path_obj)
            for index, item in enumerate(raw["runs"])
        ]
        return BatchSummary(
            path=path_obj,
            raw=raw,
            shared=dict(raw.get("shared", {})),
            runs=runs,
        )

    # save_summary(...) file structure:
    # [
    #   {
    #       "final_time": float,
    #       "final_state": list[float],
    #       "n_steps": int,
    #       "n_events": int,
    #       "metadata": {...},
    #       "species_names": list[str],
    #       "event_times": list[float] | None
    #   },
    #   ...
    # ]
    #
    # For this older compact format, load(...) converts each summary row into a
    # minimal BatchRunItem.data dictionary with fields used by BatchSummary:
    # mode, seed, final_time, n_steps, n_events, and stop_reason.
    if isinstance(raw, list):
        runs = [
            BatchRunItem(
                index=index,
                data={
                    "mode": item.get("metadata", {}).get("stepper_method", "summary"),
                    "seed": item.get("metadata", {}).get("seed"),
                    "final_time": item["final_time"],
                    "n_steps": item.get("n_steps"),
                    "n_events": item.get("n_events"),
                    "stop_reason": item.get("metadata", {}).get("stop_reason"),
                },
                source_path=path_obj,
            )
            for index, item in enumerate(raw)
        ]
        return BatchSummary(path=path_obj, raw=raw, shared={}, runs=runs)
    raise ValueError("unsupported summary file: expected a top-level runs list or save_summary list")


def _select_items(items: list[BatchRunItem], index) -> BatchRunItem | BatchRunSelection:
    if isinstance(index, slice):
        return BatchRunSelection(items[index])
    if isinstance(index, range):
        return BatchRunSelection([items[int(i)] for i in index])
    if isinstance(index, (list, tuple, np.ndarray)):
        arr = np.asarray(index)
        if arr.dtype == bool:
            if arr.shape != (len(items),):
                raise IndexError("boolean index must match number of runs")
            return BatchRunSelection([item for item, keep in zip(items, arr) if bool(keep)])
        return BatchRunSelection([items[int(i)] for i in arr.tolist()])
    return items[int(index)]


def _selection_to_list(selection: BatchRunItem | BatchRunSelection) -> list[BatchRunItem]:
    if isinstance(selection, BatchRunItem):
        return [selection]
    return list(selection.items)


def _filter_items(
    items: list[BatchRunItem],
    *,
    mode: str | None = None,
    stop_reason: str | None = None,
) -> list[BatchRunItem]:
    selected = items
    if mode is not None:
        mode_key = str(mode).lower()
        selected = [item for item in selected if item.mode.lower() == mode_key]
    if stop_reason is not None:
        selected = [item for item in selected if item.data.get("stop_reason") == str(stop_reason)]
    return list(selected)


def _mode_counts(items: list[BatchRunItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.mode] = counts.get(item.mode, 0) + 1
    return counts


def _molecule_number_payload(
    items: list[BatchRunItem],
    time_points: Iterable[float],
) -> dict[str, object]:
    points = _time_points_array(time_points)
    rows = []
    for item in items:
        mol_numbers = _load_molecule_numbers(item, points)
        rows.append(
            {
                "index": int(item.index),
                "label": item.label(),
                "mode": item.mode,
                "seed": item.seed,
                "time_points": points.tolist(),
                "mol_num": mol_numbers.tolist(),
                "trajectory_path": item.data.get("trajectory_path"),
            }
        )
    return {
        "time_points": points.tolist(),
        "rows": rows,
    }


def _time_points_array(time_points: Iterable[float]) -> np.ndarray:
    if np.isscalar(time_points):
        points = np.asarray([float(time_points)], dtype=float)
    else:
        points = np.asarray(list(time_points), dtype=float)
    if points.ndim != 1:
        raise ValueError("time_points must be a scalar or one-dimensional iterable")
    if not np.all(np.isfinite(points)):
        raise ValueError("time_points must be finite")
    return points


def _load_molecule_numbers(item: BatchRunItem, time_points: np.ndarray) -> np.ndarray:
    trajectory_path = _resolve_trajectory_path(item)
    times, states, _species_names, _metadata = load_trajectory_arrays(trajectory_path, mmap=True)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or states.ndim != 2 or states.shape[0] != times.shape[0]:
        raise ValueError(f"invalid trajectory shape in {trajectory_path}")
    if times.size == 0:
        return np.full(time_points.shape, np.nan, dtype=float)
    total_molecules = states.sum(axis=1)
    indices = np.searchsorted(times, time_points, side="right") - 1
    indices = np.clip(indices, 0, times.size - 1)
    return np.asarray(total_molecules[indices], dtype=float)


def _plot_dt_compare(
    items: list[BatchRunItem],
    *,
    ax=None,
    cmap: str = "viridis",
    annotate: bool = True,
    annotation_format: str = ".4g",
    title: str | None = None,
    save_output: bool = True,
    output_dir: PathLike | None = None,
    figure_filename: str = "dt_compare_heatmap.png",
    table_filename: str = "dt_compare_report.csv",
    dpi: int = 200,
):
    payload = _dt_compare_payload(items)

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=_dt_compare_figsize(payload))
    else:
        fig = ax.figure

    values = np.asarray(payload["mean_simulation_time"], dtype=float)
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(payload["dt_macro_values"])))
    ax.set_xticklabels([_format_dt_value(value) for value in payload["dt_macro_values"]])
    ax.set_yticks(np.arange(len(payload["dt_cle_values"])))
    ax.set_yticklabels([_format_dt_value(value) for value in payload["dt_cle_values"]])
    ax.set_xlabel("dt_macro")
    ax.set_ylabel("dt_cle")
    ax.set_title(title or "Mean simulation time by blended dt")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("mean simulation_final_time")

    if annotate:
        counts = np.asarray(payload["count"], dtype=int)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if not np.isfinite(value):
                    continue
                text = format(float(value), annotation_format)
                if counts[row, col] > 1:
                    text = f"{text}\nn={int(counts[row, col])}"
                ax.text(col, row, text, ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    if save_output:
        _write_dt_compare_outputs(
            payload,
            fig,
            items,
            output_dir=output_dir,
            figure_filename=figure_filename,
            table_filename=table_filename,
            dpi=dpi,
        )
    return fig, ax, payload


def _dt_compare_payload(items: list[BatchRunItem]) -> dict[str, object]:
    grouped: dict[tuple[float, float | None], list[BatchRunItem]] = {}
    for item in items:
        dt_cle = item.data.get("blended_dt_cle")
        dt_macro = item.data.get("blended_dt_macro")
        if dt_cle is None:
            continue
        key = (float(dt_cle), None if dt_macro is None else float(dt_macro))
        grouped.setdefault(key, []).append(item)

    if not grouped:
        raise ValueError("selected runs do not contain blended_dt_cle/blended_dt_macro metadata")

    dt_cle_values = sorted({key[0] for key in grouped})
    dt_macro_values = sorted({key[1] for key in grouped}, key=_optional_float_sort_key)
    row_by_dt_cle = {value: index for index, value in enumerate(dt_cle_values)}
    col_by_dt_macro = {value: index for index, value in enumerate(dt_macro_values)}

    values = np.full((len(dt_cle_values), len(dt_macro_values)), np.nan, dtype=float)
    counts = np.zeros(values.shape, dtype=int)
    run_indices: list[list[list[int]]] = [[[] for _ in dt_macro_values] for _ in dt_cle_values]
    for (dt_cle, dt_macro), group_items in grouped.items():
        row = row_by_dt_cle[dt_cle]
        col = col_by_dt_macro[dt_macro]
        times = np.asarray([item.simulation_time for item in group_items], dtype=float)
        values[row, col] = float(np.mean(times))
        counts[row, col] = int(times.size)
        run_indices[row][col] = [int(item.index) for item in group_items]

    return {
        "dt_cle_values": [float(value) for value in dt_cle_values],
        "dt_macro_values": [None if value is None else float(value) for value in dt_macro_values],
        "mean_simulation_time": values,
        "count": counts,
        "run_indices": run_indices,
    }


def _optional_float_sort_key(value: float | None) -> tuple[int, float]:
    return (1, 0.0) if value is None else (0, float(value))


def _format_dt_value(value: float | None) -> str:
    return "None" if value is None else f"{float(value):.6g}"


def _dt_compare_figsize(payload: dict[str, object]) -> tuple[float, float]:
    n_cols = len(payload["dt_macro_values"])
    n_rows = len(payload["dt_cle_values"])
    return (max(5.0, 1.35 * n_cols + 2.5), max(4.0, 0.95 * n_rows + 2.0))


def _write_dt_compare_outputs(
    payload: dict[str, object],
    fig,
    items: list[BatchRunItem],
    *,
    output_dir: PathLike | None,
    figure_filename: str,
    table_filename: str,
    dpi: int,
) -> None:
    output_path = _dt_compare_output_dir(items, output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path = output_path / str(figure_filename)
    table_path = output_path / str(table_filename)

    fig.savefig(figure_path, dpi=int(dpi), bbox_inches="tight")
    _write_dt_compare_csv(payload, table_path)
    payload["output_dir"] = str(output_path)
    payload["figure_path"] = str(figure_path)
    payload["table_path"] = str(table_path)


def _dt_compare_output_dir(items: list[BatchRunItem], output_dir: PathLike | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    source_paths = [item.source_path for item in items if item.source_path is not None]
    if not source_paths:
        raise ValueError("output_dir must be provided when selected runs do not come from a loaded JSON file")
    return Path(source_paths[0]).parent / "output"


def _write_dt_compare_csv(payload: dict[str, object], table_path: Path) -> None:
    values = np.asarray(payload["mean_simulation_time"], dtype=float)
    counts = np.asarray(payload["count"], dtype=int)
    dt_cle_values = payload["dt_cle_values"]
    dt_macro_values = payload["dt_macro_values"]
    run_indices = payload["run_indices"]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dt_cle",
                "dt_macro",
                "mean_simulation_time",
                "n_runs",
                "run_indices",
            ],
        )
        writer.writeheader()
        for row, dt_cle in enumerate(dt_cle_values):
            for col, dt_macro in enumerate(dt_macro_values):
                value = values[row, col]
                count = int(counts[row, col])
                if not np.isfinite(value) or count <= 0:
                    continue
                writer.writerow(
                    {
                        "dt_cle": _format_dt_value(dt_cle),
                        "dt_macro": _format_dt_value(dt_macro),
                        "mean_simulation_time": f"{float(value):.12g}",
                        "n_runs": count,
                        "run_indices": ";".join(str(int(index)) for index in run_indices[row][col]),
                    }
                )


def _resolve_trajectory_path(item: BatchRunItem) -> Path:
    raw_path = item.data.get("trajectory_path")
    if raw_path is None:
        raise ValueError(f"{item.label()} does not contain trajectory_path; mol_num requires saved trajectories")
    path = Path(str(raw_path))
    if path.is_absolute():
        if trajectory_storage_exists(path):
            return path
        raise FileNotFoundError(path)
    if trajectory_storage_exists(path):
        return path
    if item.source_path is not None:
        for parent in [item.source_path.parent, *item.source_path.parents]:
            candidate = parent / path
            if trajectory_storage_exists(candidate):
                return candidate
    raise FileNotFoundError(path)


def _format_molecule_number_payload(payload: dict[str, object]) -> str:
    time_points = [float(value) for value in payload["time_points"]]
    lines = [
        "mol_num",
        "time_points: " + _format_float_list(time_points),
    ]
    for row in payload["rows"]:
        seed = row.get("seed")
        seed_text = "" if seed is None else f" seed={int(seed)}"
        values = _format_float_list([float(value) for value in row["mol_num"]])
        lines.append(f"{int(row['index']):04d} {str(row['label']):<22}{seed_text}: {values}")
    return "\n".join(lines)


def _format_float_list(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def _scatter_marker(value: float, left: float, right: float, width: int) -> str:
    position = int(round((float(value) - left) / (right - left) * (int(width) - 1)))
    position = min(max(position, 0), int(width) - 1)
    chars = ["."] * int(width)
    chars[position] = "*"
    return "".join(chars)


def _main(argv: list[str]) -> int:
    if len(argv) <= 1:
        print("Usage: python -i polymer_sim/recording/summary.py <summary-json>")
        print("Then try: sum = load('<summary-json>'); sum[[0, 1]].scale()")
        return 0
    batch = load(argv[1])
    print(batch.overview())
    batch.scale()
    return 0


if __name__ == "__main__":
    _main(sys.argv)
