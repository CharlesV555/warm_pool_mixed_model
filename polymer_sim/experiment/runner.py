from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState
from polymer_sim.partition.strategies import BlendingStrategy, PartitionStrategy
from polymer_sim.recording.base import BaseRecorder
from polymer_sim.recording.summary import RunSummary, SummaryRecorder
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
        partition_strategy: PartitionStrategy | None = None,
        blending_strategy: BlendingStrategy | None = None,
        max_steps: int = 100_000,
    ) -> RunResult:
        rng = np.random.default_rng(int(seed))
        state = SystemState.from_x0(network.x0 if x0 is None else x0)
        active_recorder = recorder or SummaryRecorder()
        context = StepperContext(
            network=network,
            rng=rng,
            partition_strategy=partition_strategy,
            blending_strategy=blending_strategy,
        )

        active_recorder.initialize(
            species_names=list(network.species_names),
            initial_state=state.x,
            metadata={"seed": int(seed)},
        )

        while state.t < t_end and state.step_count < max_steps:
            remaining = float(t_end - state.t)
            step_dt = remaining if dt is None else min(float(dt), remaining)
            result = stepper.step(state, step_dt, context)
            active_recorder.record_step(
                time=float(state.t),
                state=state.x,
                step_count=state.step_count,
                event_count=state.event_count,
                event_time=float(state.t) if result.event_occurred else None,
                metadata={"seed": int(seed)},
            )
            if result.advanced_time <= 0.0 and not result.event_occurred:
                break

        recorded = active_recorder.finalize()
        if isinstance(recorded, RunSummary):
            summary = recorded
        else:
            summary = RunSummary(
                final_time=float(state.t),
                final_state=np.array(state.x, dtype=float, copy=True),
                n_steps=int(state.step_count),
                n_events=int(state.event_count),
                metadata={"seed": int(seed)},
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
        partition_strategy: PartitionStrategy | None = None,
        blending_strategy: BlendingStrategy | None = None,
        max_steps: int = 100_000,
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
                    partition_strategy=partition_strategy,
                    blending_strategy=blending_strategy,
                    max_steps=max_steps,
                )
            )
        return results
