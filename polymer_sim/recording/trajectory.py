"""单次完整轨迹记录的数据结构与离线保存/读取接口。

本模块专门服务于“需要完整轨迹”的路径。默认模拟流程不强制调用这里的保存逻辑，
从而满足“ExperimentRunner 默认不保存完整轨迹”的要求。

完整轨迹遵循“一个文件只存一次完整记录”的原则：一个 `.npz` 文件对应一次运行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from polymer_sim.recording.base import BaseRecorder, BaseTrajectoryRecord, PathLike


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

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.states = np.asarray(self.states, dtype=float)
        if self.times.ndim != 1:
            raise ValueError("times must have shape (T,)")
        if self.states.ndim != 2:
            raise ValueError("states must have shape (T, n_species)")
        if self.states.shape[0] != self.times.shape[0]:
            raise ValueError("states.shape[0] must match times.shape[0]")
        if self.states.shape[1] != len(self.species_names):
            raise ValueError("states.shape[1] must match len(species_names)")


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
        self._times.append(float(time))
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
        channel_id = step_metadata.get("channel_id")
        is_reaction_event = channel_id is not None or event_time is not None
        if step_metadata:
            self._metadata.update(step_metadata)
        if is_reaction_event:
            event_timestamp = float(event_time if event_time is not None else time)
            if channel_id is not None and self._channel_trigger_counts is not None:
                self._channel_trigger_counts[int(channel_id)] += 1
                self._channel_event_times.append(event_timestamp)
                self._channel_event_ids.append(int(channel_id))
            interval = event_timestamp - self._last_reaction_event_time
            self._reaction_intervals.append(float(max(interval, 0.0)))
            self._reaction_interval_times.append(event_timestamp)
            self._last_reaction_event_time = event_timestamp
        if self._tracked_outflow_species:
            outflow_row = [0.0 for _ in self._tracked_outflow_species]
            if step_metadata:
                channel_id = step_metadata.get("channel_id")
                if channel_id is not None and int(channel_id) in self._tracked_outflow_channel_to_col:
                    outflow_row[self._tracked_outflow_channel_to_col[int(channel_id)]] = 1.0
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
        )


def save_trajectory_record(path: PathLike, record: TrajectoryRecord) -> None:
    """保存单次完整轨迹记录到压缩 npz 文件。"""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path_obj,
        times=record.times,
        states=record.states,
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
        )
