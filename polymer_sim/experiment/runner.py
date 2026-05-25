from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState
from polymer_sim.partition.strategies import BlendingStrategy, PartitionStrategy
from polymer_sim.recording.base import BaseRecorder
from polymer_sim.recording.summary import RunSummary, SummaryRecorder
from polymer_sim.simulation.restriction import BaseRestriction, RestrictionContext
from polymer_sim.simulation.stepper import BaseStepper, StepperContext


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
    ) -> RunResult:
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

        while state.t < t_end and state.step_count < max_steps:
            if max_runtime_seconds is not None and (perf_counter() - started_at) >= float(max_runtime_seconds):
                stop_reason = "max_runtime_seconds"
                break
            remaining = float(t_end - state.t)
            step_dt = remaining if dt is None else min(float(dt), remaining)
            result = stepper.step(state, step_dt, context)
            if restriction is not None:
                restriction.apply(state, float(result.advanced_time), restriction_context, result)
            step_metadata = {"seed": int(seed)}
            if result.channel_id is not None:
                step_metadata["channel_id"] = int(result.channel_id)
            if restriction is not None:
                step_metadata.update(restriction.metadata())
            active_recorder.record_step(
                time=float(state.t),
                state=state.x,
                step_count=state.step_count,
                event_count=state.event_count,
                event_time=float(state.t) if result.event_occurred else None,
                metadata=step_metadata,
            )
            if result.advanced_time <= 0.0 and not result.event_occurred:
                stop_reason = "no_progress"
                break

        if stop_reason == "reached_t_end":
            if state.step_count >= max_steps and state.t < t_end:
                stop_reason = "max_steps"
            elif state.t < t_end and max_runtime_seconds is not None and (perf_counter() - started_at) >= float(max_runtime_seconds):
                stop_reason = "max_runtime_seconds"

        recorded = active_recorder.finalize()
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
                )
            )
        return results
