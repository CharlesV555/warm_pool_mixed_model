"""默认轻量 summary 记录路径。

本模块是 recording 层的默认工作路径：

1. 模拟运行期间默认只累计轻量信息。
2. 运行结束后生成 `RunSummary`。
3. 多次运行可以在一个文件中集中保存，支持离线读取和后续统计/绘图。

这样可以避免默认保存完整轨迹带来的额外内存与 I/O 成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import sys
from typing import Iterable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from polymer_sim.recording.base import BaseRecorder, BaseRunSummary, PathLike


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
        if metadata:
            self._metadata.update(metadata)
        if self.include_event_times and event_time is not None:
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
    with np.load(trajectory_path, allow_pickle=True) as data:
        times = np.asarray(data["times"], dtype=float)
        states = np.asarray(data["states"], dtype=float)
    if times.ndim != 1 or states.ndim != 2 or states.shape[0] != times.shape[0]:
        raise ValueError(f"invalid trajectory shape in {trajectory_path}")
    if times.size == 0:
        return np.full(time_points.shape, np.nan, dtype=float)
    total_molecules = states.sum(axis=1)
    indices = np.searchsorted(times, time_points, side="right") - 1
    indices = np.clip(indices, 0, times.size - 1)
    return np.asarray(total_molecules[indices], dtype=float)


def _resolve_trajectory_path(item: BatchRunItem) -> Path:
    raw_path = item.data.get("trajectory_path")
    if raw_path is None:
        raise ValueError(f"{item.label()} does not contain trajectory_path; mol_num requires saved trajectories")
    path = Path(str(raw_path))
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(path)
    if path.exists():
        return path
    if item.source_path is not None:
        for parent in [item.source_path.parent, *item.source_path.parents]:
            candidate = parent / path
            if candidate.exists():
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
