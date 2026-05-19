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

    def initialize(self, species_names: list[str], initial_state: np.ndarray, metadata: dict | None = None) -> None:
        self._species_names = list(species_names)
        self._times = [0.0]
        self._states = [np.asarray(initial_state, dtype=float).copy()]
        self._metadata = dict(metadata or {})

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
        if metadata:
            self._metadata.update(metadata)
        self._metadata["n_steps"] = int(step_count)
        self._metadata["n_events"] = int(event_count)

    def finalize(self) -> TrajectoryRecord:
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
