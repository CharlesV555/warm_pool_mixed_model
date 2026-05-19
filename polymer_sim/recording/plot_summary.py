"""多次重复运行结果的轻量统计绘图。

本模块面向离线工作流：输入可以是 `RunSummary` 列表，也可以是 summary 文件路径列表。
第一轮只实现最常用的最小统计图函数，保持结构清晰、依赖轻量。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polymer_sim.recording.base import PathLike
from polymer_sim.recording.summary import RunSummary, load_summary


def plot_final_state_distribution(
    summaries_or_paths: list[RunSummary] | list[PathLike],
    species_index: int,
    bins: int = 20,
    title: str | None = None,
):
    """绘制多次重复中某个物种最终状态分布。"""

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
    """绘制事件时间分布。

    第一轮优先使用 `RunSummary.event_times`；如果没有该字段，则回退为各次运行的 `final_time`
    分布，用于提供一个最小可工作的统计图。
    """

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
    """绘制多次重复最终状态的均值与标准差。

    第一轮采用最终状态 across runs 的最小版本，横轴为物种索引。
    """

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
