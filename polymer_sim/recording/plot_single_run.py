"""单次运行轨迹绘图函数。

本模块只依赖离线记录对象或记录文件路径，不依赖主模拟器内部对象。
第一轮先提供最常用的时间序列绘图接口，后续可继续扩展更多分析函数。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polymer_sim.recording.base import PathLike
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


def _as_trajectory_record(record_or_path: TrajectoryRecord | PathLike) -> TrajectoryRecord:
    if isinstance(record_or_path, TrajectoryRecord):
        return record_or_path
    return load_trajectory_record(Path(record_or_path))
