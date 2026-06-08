"""recording 层公共导出。

本包同时支持两条记录路径：

1. 默认轻量 summary 路径。
2. 显式启用的完整 trajectory 路径。

并提供独立于模拟器的离线保存、读取、统计和绘图模块。
"""

from polymer_sim.recording.base import BaseRecorder, BaseRunSummary, BaseTrajectoryRecord
from polymer_sim.recording.plot_single_run import (
    animate_reaction_network_state_tree,
    plot_channel_propensity_time_series,
    plot_event_time_distribution,
    plot_final_state_distribution,
    plot_mean_std_over_runs,
    plot_reaction_interval_bar,
    plot_reaction_interval_wave,
    plot_reaction_frequency_over_time,
    plot_reaction_network_state_tree,
    plot_reaction_trigger_frequency,
    plot_species_with_outflow,
    plot_time_series,
)
from polymer_sim.recording.plot_summary import plot_summary_pipeline
from polymer_sim.recording.summary import RunSummary, SummaryRecorder, load_summary, save_summary
from polymer_sim.recording.timing import TimingRecorder, TimingSummary, load_timing_summary, save_timing_summary
from polymer_sim.recording.trajectory import (
    TrajectoryRecord,
    TrajectoryRecorder,
    load_trajectory_record,
    save_trajectory_record,
)

__all__ = [
    "animate_reaction_network_state_tree",
    "BaseRecorder",
    "BaseRunSummary",
    "BaseTrajectoryRecord",
    "RunSummary",
    "SummaryRecorder",
    "TimingRecorder",
    "TimingSummary",
    "TrajectoryRecord",
    "TrajectoryRecorder",
    "load_summary",
    "load_timing_summary",
    "load_trajectory_record",
    "plot_event_time_distribution",
    "plot_final_state_distribution",
    "plot_mean_std_over_runs",
    "plot_channel_propensity_time_series",
    "plot_reaction_interval_bar",
    "plot_reaction_interval_wave",
    "plot_reaction_frequency_over_time",
    "plot_reaction_network_state_tree",
    "plot_reaction_trigger_frequency",
    "plot_summary_pipeline",
    "plot_species_with_outflow",
    "plot_time_series",
    "save_summary",
    "save_timing_summary",
    "save_trajectory_record",
]
