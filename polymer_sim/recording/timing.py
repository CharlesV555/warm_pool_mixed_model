"""低侵入耗时监控工具。

本模块用于记录模拟不同阶段的耗时和调用次数，重点是：

1. 支持 context manager 风格计时。
2. 支持多模块累计统计。
3. 不强耦合具体 stepper 或 recorder 实现。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import log
from pathlib import Path
from time import perf_counter
import json
import numpy as np
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


@dataclass(slots=True)
class RunTimingReport:
    seed: int
    stepper: str
    final_time: float
    n_steps: int
    n_events: int
    stop_reason: str
    total_wall_seconds: float
    runner_setup_wall_seconds: float
    simulation_loop_wall_seconds: float
    finalize_wall_seconds: float
    step_wall_seconds: float
    restriction_wall_seconds: float
    recording_wall_seconds: float
    event_interval: int
    event_timing_samples: list[dict[str, float | int]]
    simulation_clock_interval: float
    simulation_clock_samples: list[dict[str, float | int]]
    network_build_wall_seconds: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


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


def save_run_timing_report(
    output_dir: PathLike,
    report: RunTimingReport,
    *,
    name: str | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = _unique_stem(output_path, name or f"timing_seed_{report.seed}")
    json_path = output_path / f"{stem}.json"
    plot_path = output_path / f"{stem}_events.png"
    simulation_clock_plot_path = output_path / f"{stem}_simulation_clock.png"
    payload = _run_timing_report_payload(report)
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    _save_event_timing_plot(plot_path, report)
    _save_simulation_clock_plot(simulation_clock_plot_path, report)
    return {"json": json_path, "event_plot": plot_path, "simulation_clock_plot": simulation_clock_plot_path}


def _run_timing_report_payload(report: RunTimingReport) -> dict[str, object]:
    network_build_note = None
    if report.network_build_wall_seconds is None:
        network_build_note = (
            "network construction happens before ExperimentRunner.run_one; "
            "pass network_build_elapsed_seconds to include it in this report"
        )
    return {
        "seed": int(report.seed),
        "stepper": str(report.stepper),
        "final_time": float(report.final_time),
        "n_steps": int(report.n_steps),
        "n_events": int(report.n_events),
        "stop_reason": str(report.stop_reason),
        "total_wall_seconds": float(report.total_wall_seconds),
        "network_build_wall_seconds": (
            None if report.network_build_wall_seconds is None else float(report.network_build_wall_seconds)
        ),
        "network_build_note": network_build_note,
        "runner_setup_wall_seconds": float(report.runner_setup_wall_seconds),
        "simulation_loop_wall_seconds": float(report.simulation_loop_wall_seconds),
        "finalize_wall_seconds": float(report.finalize_wall_seconds),
        "step_wall_seconds": float(report.step_wall_seconds),
        "restriction_wall_seconds": float(report.restriction_wall_seconds),
        "recording_wall_seconds": float(report.recording_wall_seconds),
        "event_interval": int(report.event_interval),
        "event_timing_samples": [dict(item) for item in report.event_timing_samples],
        "simulation_clock_interval": float(report.simulation_clock_interval),
        "simulation_clock_samples": [dict(item) for item in report.simulation_clock_samples],
        "metadata": dict(report.metadata),
    }


def _save_event_timing_plot(path: Path, report: RunTimingReport) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = report.event_timing_samples
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if samples:
        events = [int(item["event_count"]) for item in samples]
        elapsed = [float(item["wall_elapsed_seconds"]) for item in samples]
        ax.plot(events, elapsed, marker="o", linewidth=1.5, label="cumulative wall time")
        ax.set_xlabel("Discrete reaction event count")
        ax.set_ylabel("Wall time (s)")
        ax.legend(loc="best")  # 或 "upper left" 等
        ax.grid(True, alpha=0.3)
        if len(samples) > 1:
            ax2 = ax.twinx()
            ax2.legend(loc="best")  
            interval_events = events[1:]
            interval_elapsed = [float(item["interval_wall_seconds"]) for item in samples[1:]]
            ax2.plot(interval_events, interval_elapsed, marker="x", color="tab:orange", linewidth=1.0, label="interval wall time")
            ax2.set_ylabel("Interval wall time (s)")
        ax.set_title(f"Runtime vs events: {report.stepper}, seed={report.seed}")
    else:
        ax.text(0.5, 0.5, "No discrete events recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _unique_stem(output_dir: Path, stem: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(stem))
    candidate = safe
    index = 1
    while (output_dir / f"{candidate}.json").exists() or (output_dir / f"{candidate}_events.png").exists():
        candidate = f"{safe}_{index:03d}"
        index += 1
    return candidate


def _save_event_timing_plot(path: Path, report: RunTimingReport, use_log=False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples = report.event_timing_samples
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if samples:
        events = [int(item["event_count"]) for item in samples]

        elapsed = [float(item["wall_elapsed_seconds"]) for item in samples]
        ax.plot(events, elapsed, marker="o", linewidth=1.5, label="cumulative wall time")
        if use_log:
            ax.set_xscale('log')  # 这一行就够了，数据用原始 events
            ax.set_xlabel("Discrete reaction event count (log scale)")
        else:
            ax.set_xlabel("Discrete reaction event count")
        ax.set_ylabel("Wall time (s)")
        ax.grid(True, alpha=0.3)
        if len(samples) > 1:
            ax2 = ax.twinx()
            interval_events = events[1:]
            interval_elapsed = [float(item["interval_wall_seconds"]) for item in samples[1:]]
            ax2.plot(interval_events, interval_elapsed, marker="x", color="tab:orange", linewidth=1.0, label="interval wall time")
            ax2.set_ylabel("Interval wall time (s)")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax.legend(loc="best")
        ax.set_title(f"Runtime vs events: {report.stepper}, seed={report.seed}")
    else:
        ax.text(0.5, 0.5, "No discrete events recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_simulation_clock_plot(path: Path, report: RunTimingReport, use_log = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    samples = report.simulation_clock_samples
    fig, ax = plt.subplots(figsize=(9, 5))
    if samples:
        x = [0.5 * (float(item["t_start"]) + float(item["t_end"])) for item in samples]
        
        # 获取所有密度数据
        actual_density = [float(item["actual_ssa_event_density"]) for item in samples]
        expected_total = [float(item["expected_total_event_density"]) for item in samples]
        expected_jump = [float(item["expected_jump_event_density"]) for item in samples]
        expected_cle = [float(item["expected_cle_absorbed_event_density"]) for item in samples]
        
        # 绘制数据
        ax.plot(
            x,
            actual_density,
            marker="o",
            linewidth=1.3,
            label="actual SSA/jump events",
        )
        ax.plot(
            x,
            expected_total,
            linewidth=1.3,
            label="expected total SSA-equivalent events",
        )
        ax.plot(
            x,
            expected_jump,
            linewidth=1.3,
            label="expected discrete jump events",
        )
        ax.plot(
            x,
            expected_cle,
            linewidth=1.3,
            label="expected CLE-absorbed events",
        )
        
        # 根据 use_log 设置 y 轴为对数刻度
        if use_log:
            ax.set_yscale('log')
            ax.set_ylabel("Event density per simulation-time unit (log scale)")
        else:
            ax.set_ylabel("Event density per simulation-time unit")
        
        ax.set_xlabel("Simulation time")
        ax.set_title(
            f"Simulation-clock event density: {report.stepper}, seed={report.seed}, "
            f"interval={report.simulation_clock_interval:g}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No simulation-clock samples recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _unique_stem(output_dir: Path, stem: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(stem))
    candidate = safe
    index = 1
    while (
        (output_dir / f"{candidate}.json").exists()
        or (output_dir / f"{candidate}_events.png").exists()
        or (output_dir / f"{candidate}_simulation_clock.png").exists()
    ):
        candidate = f"{safe}_{index:03d}"
        index += 1
    return candidate


def load_timing_summary(path: PathLike) -> TimingSummary:
    """读取耗时摘要。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TimingSummary(
        elapsed={str(k): float(v) for k, v in payload.get("elapsed", {}).items()},
        counts={str(k): int(v) for k, v in payload.get("counts", {}).items()},
    )
