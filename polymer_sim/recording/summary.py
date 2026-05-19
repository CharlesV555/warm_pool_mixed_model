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

import numpy as np

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
