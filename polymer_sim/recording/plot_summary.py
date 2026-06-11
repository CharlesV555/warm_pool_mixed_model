"""Pipeline helpers for repeated-run summary plotting.
只在最后批量跑数据时来用，单次跑数据时直接调用 plot_single_run 里的函数就好了。
我有点只想保留plot_generation.py了，plot_summary.py感觉有点鸡肋了，毕竟现在的summary也不太复杂，直接在plot_generation里写个函数来调用plot_single_run里的函数就好了。"""

from __future__ import annotations

import numpy as np

from polymer_sim.recording.base import PathLike
from polymer_sim.recording.plot_single_run import (
    plot_event_time_distribution,
    plot_final_state_distribution,
    plot_mean_std_over_runs,
)
from polymer_sim.recording.summary import RunSummary


def plot_summary_pipeline(
    summaries_or_paths: list[RunSummary] | list[PathLike],
    *,
    species_index: int = 0,
    species_indices: list[int] | np.ndarray | None = None,
    bins: int = 20,
    title_prefix: str | None = None,
):
    """Run the standard repeated-run summary plot pipeline."""

    prefix = "" if title_prefix is None else f"{title_prefix} "
    figures = {
        "final_state_distribution": plot_final_state_distribution(
            summaries_or_paths,
            species_index=int(species_index),
            bins=int(bins),
            title=f"{prefix}Final State Distribution (species {int(species_index)})" if prefix else None,
        ),
        "event_time_distribution": plot_event_time_distribution(
            summaries_or_paths,
            bins=int(bins),
            title=f"{prefix}Event Time Distribution" if prefix else None,
        ),
        "mean_std_over_runs": plot_mean_std_over_runs(
            summaries_or_paths,
            species_indices=species_indices,
            title=f"{prefix}Mean +/- Std Over Runs" if prefix else None,
        ),
    }
    return figures


__all__ = [
    "plot_event_time_distribution",
    "plot_final_state_distribution",
    "plot_mean_std_over_runs",
    "plot_summary_pipeline",
]
