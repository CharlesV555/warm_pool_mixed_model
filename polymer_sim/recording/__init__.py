"""recording 层公共导出。

本包同时支持两条记录路径：

1. 默认轻量 summary 路径。
2. 显式启用的完整 trajectory 路径。

并提供独立于模拟器的离线保存、读取、统计和绘图模块。
"""

from polymer_sim.recording.base import BaseRecorder, BaseRunSummary, BaseTrajectoryRecord
from polymer_sim.recording.cle_sparsity_sampler import CLESparsitySample, CLESparsitySampler, save_cle_sparsity_plot
from polymer_sim.recording.fast_network_report import FastNetworkReportRecorder
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
from polymer_sim.recording.summary import RunSummary, SummaryRecorder, format_stepper_info, load_summary, save_summary
from polymer_sim.recording.timing import (
    RunTimingReport,
    TimingRecorder,
    TimingSummary,
    load_timing_summary,
    save_run_timing_report,
    save_timing_summary,
)
from polymer_sim.recording.trajectory import (
    DTStatistics,
    TrajectoryRecord,
    TrajectoryRecorder,
    load_trajectory_record,
    save_trajectory_record,
    trajectory_dt_statistics,
    trajectory_final_time,
)

__all__ = [
    "animate_reaction_network_state_tree",
    "BaseRecorder",
    "BaseRunSummary",
    "BaseTrajectoryRecord",
    "CLESparsitySample",
    "CLESparsitySampler",
    "DTStatistics",
    "FastNetworkReportRecorder",
    "RunSummary",
    "RunTimingReport",
    "SummaryRecorder",
    "TimingRecorder",
    "TimingSummary",
    "TrajectoryRecord",
    "TrajectoryRecorder",
    "format_stepper_info",
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
    "save_cle_sparsity_plot",
    "save_run_timing_report",
    "save_timing_summary",
    "save_trajectory_record",
    "trajectory_dt_statistics",
    "trajectory_final_time",
]
