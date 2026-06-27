from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState
from polymer_sim.partition.strategies import BlendingStrategy, PartitionStrategy
from polymer_sim.recording.base import BaseRecorder, PathLike
from polymer_sim.recording.summary import RunSummary, SummaryRecorder
from polymer_sim.recording.timing import RunTimingReport, save_run_timing_report
from polymer_sim.simulation.restriction import BaseRestriction, RestrictionContext
from polymer_sim.simulation.stepper import BaseStepper, StepperContext, StepResult


@dataclass(slots=True)
class RunResult:
    seed: int
    state: SystemState
    recorder: BaseRecorder | None
    summary: RunSummary


class ExperimentRunner:
    def run_one(
        self,
        network: ReactionNetworkData,
        stepper: BaseStepper,
        *,
        t_end: float,
        seed: int,
        x0: np.ndarray | None = None,
        dt: float | None = None,
        recorder: BaseRecorder | None = None,
        restriction: BaseRestriction | None = None,
        partition_strategy: PartitionStrategy | None = None,
        blending_strategy: BlendingStrategy | None = None,
        max_steps: int = 100_000,
        max_runtime_seconds: float | None = None,
        timing_report: bool = False,
        timing_report_dir: PathLike | None = None,
        timing_report_interval_events: int = 1000,
        timing_report_sim_interval: float = 0.01,
        timing_report_name: str | None = None,
        network_build_elapsed_seconds: float | None = None,
    ) -> RunResult:
        run_started_at = perf_counter()
        timing_enabled = bool(timing_report)
        if timing_enabled and int(timing_report_interval_events) <= 0:
            raise ValueError("timing_report_interval_events must be > 0")
        if timing_enabled and float(timing_report_sim_interval) <= 0.0:
            raise ValueError("timing_report_sim_interval must be > 0")
        timing_step_elapsed = 0.0
        timing_restriction_elapsed = 0.0
        timing_recording_elapsed = 0.0
        timing_finalize_elapsed = 0.0
        timing_event_samples: list[dict[str, float | int]] = []
        timing_simulation_clock_samples: list[dict[str, float | int]] = []
        timing_next_event_sample = int(timing_report_interval_events)
        timing_next_sim_sample_index = 0
        timing_last_sample_wall = 0.0

        rng = np.random.default_rng(int(seed))
        state = SystemState.from_x0(network.x0 if x0 is None else x0)
        active_recorder = recorder or SummaryRecorder()
        started_at = perf_counter()
        stop_reason = "reached_t_end"
        context = StepperContext(
            network=network,
            rng=rng,
            partition_strategy=partition_strategy,
            blending_strategy=blending_strategy,
        )
        restriction_context = RestrictionContext(network=network, rng=rng)

        active_recorder.initialize(
            species_names=list(network.species_names),
            initial_state=state.x,
            metadata={
                "seed": int(seed),
                "n_channels": int(network.n_channels),
                "channel_labels": [network.describe_channel(channel_id) for channel_id in range(network.n_channels)],
            },
        )

        timing_loop_started_at = perf_counter()
        timing_setup_elapsed = timing_loop_started_at - run_started_at

        while state.t < t_end and state.step_count < max_steps:
            if max_runtime_seconds is not None and (perf_counter() - started_at) >= float(max_runtime_seconds):
                stop_reason = "max_runtime_seconds"
                break
            remaining = float(t_end - state.t)
            step_dt = remaining if dt is None else min(float(dt), remaining)
            timing_step_start_sim_time = float(state.t)
            timing_step_started_at = perf_counter()
            result = stepper.step(state, step_dt, context)
            if timing_enabled:
                timing_step_elapsed += perf_counter() - timing_step_started_at
                timing_next_sim_sample_index = _accumulate_simulation_clock_timing(
                    timing_simulation_clock_samples,
                    interval=float(timing_report_sim_interval),
                    start_time=timing_step_start_sim_time,
                    end_time=float(state.t),
                    result=result,
                    next_sample_index=timing_next_sim_sample_index,
                )
            if restriction is not None:
                timing_restriction_started_at = perf_counter()
                restriction.apply(state, float(result.advanced_time), restriction_context, result)
                if timing_enabled:
                    timing_restriction_elapsed += perf_counter() - timing_restriction_started_at
            step_metadata = {"seed": int(seed)}
            if result.channel_id is not None:
                step_metadata["channel_id"] = int(result.channel_id)
            if result.details and "continuous_channel_abs_increments" in result.details:
                step_metadata["continuous_channel_abs_increments"] = result.details["continuous_channel_abs_increments"]
            if restriction is not None:
                step_metadata.update(restriction.metadata())
            timing_recording_started_at = perf_counter()
            active_recorder.record_step(
                time=float(state.t),
                state=state.x,
                step_count=state.step_count,
                event_count=state.event_count,
                event_time=float(state.t) if result.event_occurred else None,
                metadata=step_metadata,
            )
            if timing_enabled:
                timing_recording_elapsed += perf_counter() - timing_recording_started_at
                if state.event_count >= timing_next_event_sample:
                    sample_wall = perf_counter() - timing_loop_started_at
                    while timing_next_event_sample <= state.event_count:
                        timing_event_samples.append(
                            {
                                "event_count": int(timing_next_event_sample),
                                "step_count": int(state.step_count),
                                "simulation_time": float(state.t),
                                "wall_elapsed_seconds": float(sample_wall),
                                "interval_wall_seconds": float(sample_wall - timing_last_sample_wall),
                            }
                        )
                        timing_last_sample_wall = sample_wall
                        timing_next_event_sample += int(timing_report_interval_events)
            if result.advanced_time <= 0.0 and not result.event_occurred:
                stop_reason = "no_progress"
                break

        timing_loop_finished_at = perf_counter()
        timing_loop_elapsed = timing_loop_finished_at - timing_loop_started_at
        if timing_enabled and state.event_count > 0:
            final_sample_wall = timing_loop_finished_at - timing_loop_started_at
            if not timing_event_samples or int(timing_event_samples[-1]["event_count"]) != int(state.event_count):
                timing_event_samples.append(
                    {
                        "event_count": int(state.event_count),
                        "step_count": int(state.step_count),
                        "simulation_time": float(state.t),
                        "wall_elapsed_seconds": float(final_sample_wall),
                        "interval_wall_seconds": float(final_sample_wall - timing_last_sample_wall),
                    }
                )

        if stop_reason == "reached_t_end":
            if state.step_count >= max_steps and state.t < t_end:
                stop_reason = "max_steps"
            elif state.t < t_end and max_runtime_seconds is not None and (perf_counter() - started_at) >= float(max_runtime_seconds):
                stop_reason = "max_runtime_seconds"

        timing_finalize_started_at = perf_counter()
        recorded = active_recorder.finalize()
        if timing_enabled:
            timing_finalize_elapsed = perf_counter() - timing_finalize_started_at
        if isinstance(recorded, RunSummary):
            recorded.metadata["seed"] = int(seed)
            recorded.metadata["stop_reason"] = stop_reason
            summary = recorded
        else:
            summary = RunSummary(
                final_time=float(state.t),
                final_state=np.array(state.x, dtype=float, copy=True),
                n_steps=int(state.step_count),
                n_events=int(state.event_count),
                metadata={"seed": int(seed), "stop_reason": stop_reason},
                species_names=list(network.species_names),
            )
        if timing_enabled:
            report = RunTimingReport(
                seed=int(seed),
                stepper=stepper.__class__.__name__,
                final_time=float(summary.final_time),
                n_steps=int(summary.n_steps),
                n_events=int(summary.n_events),
                stop_reason=str(summary.metadata.get("stop_reason", stop_reason)),
                total_wall_seconds=float(perf_counter() - run_started_at),
                runner_setup_wall_seconds=float(timing_setup_elapsed),
                simulation_loop_wall_seconds=float(timing_loop_elapsed),
                finalize_wall_seconds=float(timing_finalize_elapsed),
                step_wall_seconds=float(timing_step_elapsed),
                restriction_wall_seconds=float(timing_restriction_elapsed),
                recording_wall_seconds=float(timing_recording_elapsed),
                event_interval=int(timing_report_interval_events),
                event_timing_samples=list(timing_event_samples),
                simulation_clock_interval=float(timing_report_sim_interval),
                simulation_clock_samples=_finalize_simulation_clock_samples(
                    timing_simulation_clock_samples,
                    float(timing_report_sim_interval),
                ),
                network_build_wall_seconds=(
                    None if network_build_elapsed_seconds is None else float(network_build_elapsed_seconds)
                ),
                metadata={
                    "t_end": float(t_end),
                    "dt": None if dt is None else float(dt),
                    "max_steps": int(max_steps),
                    "max_runtime_seconds": None if max_runtime_seconds is None else float(max_runtime_seconds),
                    "n_species": int(network.n_species),
                    "n_channels": int(network.n_channels),
                    "simulation_clock_sampling": "point_propensity_times_interval",
                },
            )
            report_paths = save_run_timing_report(
                timing_report_dir or "timing_reports",
                report,
                name=timing_report_name,
            )
            summary.metadata["timing_report_paths"] = {key: str(value) for key, value in report_paths.items()}
        return RunResult(seed=int(seed), state=state, recorder=active_recorder, summary=summary)

    def run_many(
        self,
        network: ReactionNetworkData,
        stepper: BaseStepper,
        *,
        t_end: float,
        seeds: Iterable[int],
        x0: np.ndarray | None = None,
        dt: float | None = None,
        restriction: BaseRestriction | None = None,
        partition_strategy: PartitionStrategy | None = None,
        blending_strategy: BlendingStrategy | None = None,
        max_steps: int = 100_000,
        max_runtime_seconds: float | None = None,
        timing_report: bool = False,
        timing_report_dir: PathLike | None = None,
        timing_report_interval_events: int = 1000,
        timing_report_sim_interval: float = 0.01,
        network_build_elapsed_seconds: float | None = None,
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for seed in seeds:
            results.append(
                self.run_one(
                    network,
                    stepper,
                    t_end=t_end,
                    seed=int(seed),
                    x0=x0,
                    dt=dt,
                    recorder=None,
                    restriction=restriction,
                    partition_strategy=partition_strategy,
                    blending_strategy=blending_strategy,
                    max_steps=max_steps,
                    max_runtime_seconds=max_runtime_seconds,
                    timing_report=timing_report,
                    timing_report_dir=timing_report_dir,
                    timing_report_interval_events=timing_report_interval_events,
                    timing_report_sim_interval=timing_report_sim_interval,
                    network_build_elapsed_seconds=network_build_elapsed_seconds,
                )
            )
        return results


def _accumulate_simulation_clock_timing(
    samples: list[dict[str, float | int]],
    *,
    interval: float,
    start_time: float,
    end_time: float,
    result: StepResult,
    next_sample_index: int,
) -> int:
    duration = max(float(end_time) - float(start_time), 0.0)
    sample_index = int(next_sample_index)
    total_propensity, jump_propensity, cle_propensity = _timing_propensity_split(result)

    if duration > 0.0:
        # Point-sample the propensity at fixed simulation-clock nodes.  The
        # expected event count for each bucket is estimated as
        # propensity(sample_time) * interval; we intentionally do not integrate
        # over all step fragments inside the bucket.
        sample_time = sample_index * float(interval)
        eps = max(float(interval), 1.0) * 1e-12
        while sample_time < float(start_time) - eps:
            sample_index += 1
            sample_time = sample_index * float(interval)
        while sample_time >= float(start_time) - eps and sample_time < float(end_time) - eps:
            bucket = _ensure_simulation_clock_bucket(samples, sample_index, interval)
            bucket["sample_time"] = float(sample_time)
            bucket["sample_count"] = int(bucket["sample_count"]) + 1
            bucket["covered_simulation_time"] = float(interval)
            bucket["total_propensity_sample"] = total_propensity
            bucket["jump_propensity_sample"] = jump_propensity
            bucket["cle_propensity_sample"] = cle_propensity
            bucket["expected_total_events"] = total_propensity * float(interval)
            bucket["expected_jump_events"] = jump_propensity * float(interval)
            bucket["expected_cle_absorbed_events"] = cle_propensity * float(interval)
            sample_index += 1
            sample_time = sample_index * float(interval)

    if result.event_occurred:
        event_time = float(end_time)
        event_bucket_index = int(np.floor(event_time / interval))
        event_bucket = _ensure_simulation_clock_bucket(samples, event_bucket_index, interval)
        event_bucket["actual_ssa_events"] = int(event_bucket["actual_ssa_events"]) + 1
    return sample_index


def _timing_propensity_split(result: StepResult) -> tuple[float, float, float]:
    details = result.details or {}
    if "total_jump_propensity" in details or "total_cle_propensity" in details:
        jump = max(float(details.get("total_jump_propensity", 0.0)), 0.0)
        cle = max(float(details.get("total_cle_propensity", 0.0)), 0.0)
        return jump + cle, jump, cle

    total = max(float(result.propensity_sum), 0.0)
    mode = str(details.get("mode", "")).lower()
    if mode in {"cle", "cle_empty"}:
        return total, 0.0, total
    return total, total, 0.0


def _ensure_simulation_clock_bucket(
    samples: list[dict[str, float | int]],
    bucket_index: int,
    interval: float,
) -> dict[str, float | int]:
    while len(samples) <= int(bucket_index):
        index = len(samples)
        t_start = index * interval
        samples.append(
            {
                "interval_index": int(index),
                "t_start": float(t_start),
                "t_end": float(t_start + interval),
                "sample_time": float(t_start),
                "sample_count": 0,
                "covered_simulation_time": 0.0,
                "actual_ssa_events": 0,
                "total_propensity_sample": 0.0,
                "jump_propensity_sample": 0.0,
                "cle_propensity_sample": 0.0,
                "expected_total_events": 0.0,
                "expected_jump_events": 0.0,
                "expected_cle_absorbed_events": 0.0,
            }
        )
    return samples[int(bucket_index)]


def _finalize_simulation_clock_samples(
    samples: list[dict[str, float | int]],
    interval: float,
) -> list[dict[str, float | int]]:
    finalized: list[dict[str, float | int]] = []
    for sample in samples:
        width = float(sample.get("covered_simulation_time", 0.0))
        density_width = width if width > 0.0 else float(interval)
        item = dict(sample)
        item["actual_ssa_event_density"] = float(item["actual_ssa_events"]) / density_width
        item["expected_total_event_density"] = float(item["expected_total_events"]) / density_width
        item["expected_jump_event_density"] = float(item["expected_jump_events"]) / density_width
        item["expected_cle_absorbed_event_density"] = float(item["expected_cle_absorbed_events"]) / density_width
        finalized.append(item)
    return finalized
