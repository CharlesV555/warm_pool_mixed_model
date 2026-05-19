from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState
from polymer_sim.partition.strategies import BlendingStrategy, FixedPartitionStrategy, NoBlendingStrategy, PartitionStrategy


@dataclass(slots=True)
class StepperContext:
    network: ReactionNetworkData
    rng: np.random.Generator
    partition_strategy: PartitionStrategy | None = None
    blending_strategy: BlendingStrategy | None = None
    fast_channels: np.ndarray | None = None


@dataclass(slots=True)
class StepResult:
    advanced_time: float
    event_occurred: bool
    channel_id: int | None = None
    propensity_sum: float = 0.0
    tau: float | None = None
    details: dict[str, Any] | None = None


class BaseStepper(ABC):
    @abstractmethod
    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        raise NotImplementedError


class SSAStepper(BaseStepper):
    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        network = context.network
        propensities = network.compute_all_propensities(state)
        return self._step_from_channels(state, dt, context, np.arange(network.n_channels), propensities)

    def step_restricted(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        channels: np.ndarray,
        propensities: np.ndarray | None = None,
    ) -> StepResult:
        network = context.network
        all_propensities = network.compute_all_propensities(state) if propensities is None else propensities
        return self._step_from_channels(state, dt, context, channels, all_propensities)

    def _step_from_channels(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        channels: np.ndarray,
        propensities: np.ndarray,
    ) -> StepResult:
        network = context.network
        rng = context.rng
        selected_channels = np.asarray(channels, dtype=np.int64)
        if selected_channels.size == 0:
            state.t += float(dt)
            state.step_count += 1
            return StepResult(advanced_time=float(dt), event_occurred=False, propensity_sum=0.0)

        selected_prop = propensities[selected_channels]
        total = float(np.sum(selected_prop))
        if total <= 0.0:
            state.t += float(dt)
            state.step_count += 1
            return StepResult(advanced_time=float(dt), event_occurred=False, propensity_sum=0.0)

        tau = float(rng.exponential(1.0 / total))
        if tau > dt:
            state.t += float(dt)
            state.step_count += 1
            return StepResult(advanced_time=float(dt), event_occurred=False, propensity_sum=total, tau=tau)

        threshold = float(rng.random() * total)
        cumulative = 0.0
        chosen = int(selected_channels[-1])
        for channel_id, propensity in zip(selected_channels, selected_prop):
            cumulative += float(propensity)
            if cumulative >= threshold:
                chosen = int(channel_id)
                break

        network.apply_channel_update(state, chosen)
        state.t += tau
        state.step_count += 1
        state.event_count += 1
        return StepResult(
            advanced_time=tau,
            event_occurred=True,
            channel_id=chosen,
            propensity_sum=total,
            tau=tau,
        )


class CLEStepper(BaseStepper):
    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        channels = self._selected_fast_channels(state, context)
        if channels.size == 0:
            state.t += float(dt)
            state.step_count += 1
            return StepResult(advanced_time=float(dt), event_occurred=False, details={"mode": "cle_empty"})

        self._apply_cle_increment(state, dt, context, channels)
        state.t += float(dt)
        state.step_count += 1
        return StepResult(
            advanced_time=float(dt),
            event_occurred=False,
            details={"mode": "cle", "n_fast_channels": int(channels.size)},
        )

    def _selected_fast_channels(self, state: SystemState, context: StepperContext) -> np.ndarray:
        if context.fast_channels is not None:
            return np.asarray(context.fast_channels, dtype=np.int64)
        if context.partition_strategy is not None:
            return context.partition_strategy.partition(context.network, state).fast_channels
        return np.arange(context.network.n_channels, dtype=np.int64)

    def _apply_cle_increment(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        channels: np.ndarray,
    ) -> None:
        network = context.network
        rng = context.rng
        for channel_id in channels:
            a = network.compute_propensity(int(channel_id), state)
            mean = a * float(dt)
            if mean <= 0.0:
                continue
            amount = mean + np.sqrt(mean) * float(rng.normal())
            network.apply_channel_delta(state.x, int(channel_id), amount)
        np.maximum(state.x, 0.0, out=state.x)


class HybridStepper(BaseStepper):
    def __init__(
        self,
        partition_strategy: PartitionStrategy | None = None,
        blending_strategy: BlendingStrategy | None = None,
    ):
        self.partition_strategy = partition_strategy
        self.blending_strategy = blending_strategy or NoBlendingStrategy()
        self._ssa = SSAStepper()
        self._cle = CLEStepper()

    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        network = context.network
        propensities = network.compute_all_propensities(state)
        partition_strategy = context.partition_strategy or self.partition_strategy
        if partition_strategy is None:
            partition_strategy = FixedPartitionStrategy([])
        partition = partition_strategy.partition(network, state, propensities)
        blending_strategy = context.blending_strategy or self.blending_strategy
        weights = blending_strategy.weights(network, state, propensities)

        slow_channels = partition.slow_channels
        slow_total = float(np.sum(propensities[slow_channels])) if slow_channels.size else 0.0
        if slow_total <= 0.0:
            self._advance_fast(state, dt, context, partition.fast_channels)
            return StepResult(
                advanced_time=float(dt),
                event_occurred=False,
                propensity_sum=0.0,
                details={"mode": "hybrid", "n_fast_channels": int(partition.fast_channels.size)},
            )

        tau = float(context.rng.exponential(1.0 / slow_total))
        if tau > dt:
            self._advance_fast(state, dt, context, partition.fast_channels)
            return StepResult(
                advanced_time=float(dt),
                event_occurred=False,
                propensity_sum=slow_total,
                tau=tau,
                details={"mode": "hybrid", "n_fast_channels": int(partition.fast_channels.size)},
            )

        self._advance_fast(state, tau, context, partition.fast_channels)
        post_propensities = network.compute_all_propensities(state)
        post_slow_total = float(np.sum(post_propensities[slow_channels]))
        if post_slow_total <= 0.0:
            return StepResult(
                advanced_time=tau,
                event_occurred=False,
                propensity_sum=0.0,
                tau=tau,
                details={"mode": "hybrid", "n_fast_channels": int(partition.fast_channels.size)},
            )

        chosen = _sample_channel(slow_channels, post_propensities[slow_channels], post_slow_total, context.rng)
        network.apply_channel_update(state, chosen)
        state.event_count += 1
        return StepResult(
            advanced_time=tau,
            event_occurred=True,
            channel_id=chosen,
            propensity_sum=post_slow_total,
            tau=tau,
            details={
            "mode": "hybrid",
            "n_fast_channels": int(partition.fast_channels.size),
            "weights_shape": tuple(weights.shape),
            },
        )

    def _advance_fast(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        fast_channels: np.ndarray,
    ) -> None:
        if fast_channels.size:
            self._cle._apply_cle_increment(state, dt, context, fast_channels)
        state.t += float(dt)
        state.step_count += 1


def _sample_channel(
    channels: np.ndarray,
    propensities: np.ndarray,
    total: float,
    rng: np.random.Generator,
) -> int:
    threshold = float(rng.random() * total)
    cumulative = 0.0
    chosen = int(channels[-1])
    for channel_id, propensity in zip(channels, propensities):
        cumulative += float(propensity)
        if cumulative >= threshold:
            chosen = int(channel_id)
            break
    return chosen
