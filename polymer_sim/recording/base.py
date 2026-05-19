"""recording 层基础抽象与通用协议。

本模块只定义 recording 子系统的最小抽象接口，不绑定具体的模拟器实现。
设计目标是：

1. 让 summary 路径和完整 trajectory 路径共享统一风格的接口。
2. 让离线读取、统计和绘图代码不依赖主模拟器对象。
3. 保持 recording 层低侵入，便于后续扩展成批量实验或更复杂的分析工作流。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


PathLike = str | Path


@dataclass(slots=True)
class BaseTrajectoryRecord:
    """单次完整轨迹记录的基础协议。

    `times` 与 `states` 的 shape 约定分别为 `(T,)` 和 `(T, n_species)`。
    该结构用于离线保存、读取、统计和绘图，默认不要求在所有模拟运行中都启用。
    """

    times: np.ndarray
    states: np.ndarray
    species_names: list[str]
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaseRunSummary:
    """单次运行轻量 summary 的基础协议。

    summary 路径是默认 recorder 路径，目标是在保留复现实验关键结果的同时，
    避免默认保存完整轨迹带来的内存与文件体积开销。
    """

    final_time: float
    final_state: np.ndarray
    n_steps: int
    n_events: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRecorder(ABC):
    """recording 层 recorder 抽象接口。

    实现类可以选择只做轻量 summary，也可以记录完整轨迹。
    recorder 生命周期与一次模拟运行对应：

    - `initialize`：运行开始时调用。
    - `record_step`：每一步推进后调用。
    - `finalize`：运行结束后返回最终记录对象。
    """

    @abstractmethod
    def initialize(self, species_names: list[str], initial_state: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_step(
        self,
        *,
        time: float,
        state: np.ndarray,
        step_count: int,
        event_count: int,
        event_time: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> BaseRunSummary | BaseTrajectoryRecord | Any:
        raise NotImplementedError
