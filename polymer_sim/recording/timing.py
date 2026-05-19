"""低侵入耗时监控工具。

本模块用于记录模拟不同阶段的耗时和调用次数，重点是：

1. 支持 context manager 风格计时。
2. 支持多模块累计统计。
3. 不强耦合具体 stepper 或 recorder 实现。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
import json

from polymer_sim.recording.base import PathLike


DEFAULT_TIMING_KEYS = (
    "propensity",
    "partition",
    "ssa_step",
    "ode_step",
    "cle_step",
    "hybrid_step",
    "recording",
    "total",
)


@dataclass(slots=True)
class TimingSummary:
    """耗时累计摘要。

    `elapsed` 与 `counts` 使用普通 dict，便于保存成 json，也便于离线查看。
    """

    elapsed: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


class TimingRecorder:
    """低侵入计时 recorder。

    典型用法：

    ```python
    timer = TimingRecorder()
    with timer.measure("ssa_step"):
        ...
    ```
    """

    def __init__(self, keys: tuple[str, ...] = DEFAULT_TIMING_KEYS):
        self.elapsed = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    @contextmanager
    def measure(self, key: str):
        start = perf_counter()
        try:
            yield self
        finally:
            delta = perf_counter() - start
            self.elapsed[key] = self.elapsed.get(key, 0.0) + float(delta)
            self.counts[key] = self.counts.get(key, 0) + 1

    def summary(self) -> TimingSummary:
        return TimingSummary(elapsed=dict(self.elapsed), counts=dict(self.counts))


def save_timing_summary(path: PathLike, summary: TimingSummary) -> None:
    """保存耗时摘要。"""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = {"elapsed": dict(summary.elapsed), "counts": dict(summary.counts)}
    path_obj.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_timing_summary(path: PathLike) -> TimingSummary:
    """读取耗时摘要。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TimingSummary(
        elapsed={str(k): float(v) for k, v in payload.get("elapsed", {}).items()},
        counts={str(k): int(v) for k, v in payload.get("counts", {}).items()},
    )
