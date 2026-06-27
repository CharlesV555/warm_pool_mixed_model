from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from polymer_sim.core.enums import ChannelBlock
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


@dataclass(slots=True)
class BlendedHybridConfig:
    i1: float = 10.0
    i2: float = 30.0
    dt_cle: float = 0.01
    dt_macro: float | None = None
    beta_tol: float = 1e-12
    round_mode: str = "nearest"
    clip_negative: bool = True
    beta_species_mode: str = "reactants"
    use_reaction_interval_dt: bool = False
    reaction_interval_update_steps: int = 100
    reaction_interval_scale: float = 1.0

    def __post_init__(self) -> None:
        self.i1 = float(self.i1)
        self.i2 = float(self.i2)
        self.dt_cle = float(self.dt_cle)
        self.dt_macro = None if self.dt_macro is None else float(self.dt_macro)
        self.beta_tol = float(self.beta_tol)
        self.reaction_interval_update_steps = int(self.reaction_interval_update_steps)
        self.reaction_interval_scale = float(self.reaction_interval_scale)
        if self.i1 >= self.i2:
            raise ValueError("i1 must be < i2")
        if self.dt_cle <= 0.0:
            raise ValueError("dt_cle must be > 0")
        if self.dt_macro is not None and self.dt_macro <= 0.0:
            raise ValueError("dt_macro must be > 0 when provided")
        if self.beta_tol < 0.0:
            raise ValueError("beta_tol must be >= 0")
        if self.round_mode not in {"nearest", "floor", "ceil"}:
            raise ValueError("round_mode must be 'nearest', 'floor', or 'ceil'")
        if self.beta_species_mode != "reactants":
            raise ValueError("beta_species_mode currently only supports 'reactants'")
        if self.reaction_interval_update_steps <= 0:
            raise ValueError("reaction_interval_update_steps must be > 0")
        if self.reaction_interval_scale <= 0.0:
            raise ValueError("reaction_interval_scale must be > 0")

    @property
    def effective_dt_macro(self) -> float:
        return self.dt_cle if self.dt_macro is None else self.dt_macro


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
            return StepResult(
                advanced_time=float(dt),
                event_occurred=False,
                details={
                    "mode": "cle_empty",
                    "continuous_channel_abs_increments": np.zeros(context.network.n_channels, dtype=float),
                },
            )

        continuous_abs = self._apply_cle_increment(state, dt, context, channels)
        state.t += float(dt)
        state.step_count += 1
        return StepResult(
            advanced_time=float(dt),
            event_occurred=False,
            details={
                "mode": "cle",
                "n_fast_channels": int(channels.size),
                "continuous_channel_abs_increments": continuous_abs,
            },
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
    ) -> np.ndarray:
        network = context.network
        rng = context.rng
        continuous_abs = np.zeros(network.n_channels, dtype=float)
        for channel_id in channels:
            a = network.compute_propensity(int(channel_id), state)
            mean = a * float(dt)
            if mean <= 0.0:
                continue
            amount = mean + np.sqrt(mean) * float(rng.normal())
            continuous_abs[int(channel_id)] += abs(float(amount))
            network.apply_channel_delta(state.x, int(channel_id), amount)
        np.maximum(state.x, 0.0, out=state.x)
        return continuous_abs


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
            continuous_abs = self._advance_fast(state, dt, context, partition.fast_channels)
            return StepResult(
                advanced_time=float(dt),
                event_occurred=False,
                propensity_sum=0.0,
                details={
                    "mode": "hybrid",
                    "n_fast_channels": int(partition.fast_channels.size),
                    "continuous_channel_abs_increments": continuous_abs,
                },
            )

        tau = float(context.rng.exponential(1.0 / slow_total))
        if tau > dt:
            continuous_abs = self._advance_fast(state, dt, context, partition.fast_channels)
            return StepResult(
                advanced_time=float(dt),
                event_occurred=False,
                propensity_sum=slow_total,
                tau=tau,
                details={
                    "mode": "hybrid",
                    "n_fast_channels": int(partition.fast_channels.size),
                    "continuous_channel_abs_increments": continuous_abs,
                },
            )

        continuous_abs = self._advance_fast(state, tau, context, partition.fast_channels)
        post_propensities = network.compute_all_propensities(state)
        post_slow_total = float(np.sum(post_propensities[slow_channels]))
        if post_slow_total <= 0.0:
            return StepResult(
                advanced_time=tau,
                event_occurred=False,
                propensity_sum=0.0,
                tau=tau,
                details={
                    "mode": "hybrid",
                    "n_fast_channels": int(partition.fast_channels.size),
                    "continuous_channel_abs_increments": continuous_abs,
                },
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
            "continuous_channel_abs_increments": continuous_abs,
            },
        )

    def _advance_fast(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        fast_channels: np.ndarray,
    ) -> np.ndarray:
        continuous_abs = np.zeros(context.network.n_channels, dtype=float)
        if fast_channels.size:
            continuous_abs = self._cle._apply_cle_increment(state, dt, context, fast_channels)
        state.t += float(dt)
        state.step_count += 1
        return continuous_abs


class BlendedHybridStepper(BaseStepper):
    def __init__(self, config: BlendedHybridConfig | None = None):
        self.config = config or BlendedHybridConfig()
        self._nu_cache: dict[int, np.ndarray] = {}
        self._last_n_clipped = 0
        self._last_total_cle_propensity = 0.0
        self._last_continuous_channel_abs_increments = np.empty(0, dtype=float)
        self._reaction_interval_dt: float | None = None

    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        if dt <= 0.0:
            return StepResult(advanced_time=0.0, event_occurred=False, details={"mode": "blended_no_dt"})

        network = context.network
        self._maybe_update_reaction_interval_dt(network, state)
        x_float = self._float_nonnegative(state.x)
        beta = self._channel_betas(network, x_float)
        beta_min = float(np.min(beta)) if beta.size else 0.0
        beta_max = float(np.max(beta)) if beta.size else 0.0

        if beta_max <= self.config.beta_tol:
            return self._pure_cle_step(state, float(dt), context, beta, beta_min, beta_max)
        if beta_min >= 1.0 - self.config.beta_tol:
            return self._pure_ssa_step(state, float(dt), context, beta_min, beta_max)
        return self._mixed_step(state, float(dt), context, beta, beta_min, beta_max)

    def _pure_cle_step(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        beta: np.ndarray,
        beta_min: float,
        beta_max: float,
    ) -> StepResult:
        duration = min(self._current_dt_macro(), dt)
        state.x[:] = self._cle_increment(context.network, state.x, beta, duration, context.rng)
        state.t += duration
        state.step_count += 1
        return StepResult(
            advanced_time=duration,
            event_occurred=False,
            propensity_sum=self._last_total_cle_propensity,
            details={
                "mode": "cle",
                "fired_channel": None,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "total_jump_propensity": 0.0,
                "total_cle_propensity": self._last_total_cle_propensity,
                "n_clipped": self._last_n_clipped,
                "stepper_dt": duration,
                "reaction_interval_dt": self._reaction_interval_dt,
                "continuous_channel_abs_increments": self._last_continuous_channel_abs_increments.copy(),
            },
        )

    def _pure_ssa_step(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        beta_min: float,
        beta_max: float,
    ) -> StepResult:
        network = context.network
        duration = min(self._current_dt_macro(), dt)
        observed = self._rounded_nonnegative(state.x)
        propensities = self._propensities_for_x(network, observed, state.t)
        propensities = self._clean_propensities(propensities, "jump propensities")
        total = float(np.sum(propensities))
        state.x[:] = observed

        details = {
            "mode": "ssa",
            "fired_channel": None,
            "beta_min": beta_min,
            "beta_max": beta_max,
            "total_jump_propensity": total,
            "total_cle_propensity": 0.0,
            "n_clipped": 0,
            "stepper_dt": duration,
            "reaction_interval_dt": self._reaction_interval_dt,
        }
        if total <= 0.0:
            state.t += duration
            state.step_count += 1
            return StepResult(advanced_time=duration, event_occurred=False, propensity_sum=0.0, details=details)

        tau = float(context.rng.exponential(1.0 / total))
        if tau > duration:
            state.t += duration
            state.step_count += 1
            return StepResult(
                advanced_time=duration,
                event_occurred=False,
                propensity_sum=total,
                tau=tau,
                details=details,
            )

        channel = _sample_channel(np.arange(network.n_channels, dtype=np.int64), propensities, total, context.rng)
        applied = self._apply_jump_safely(network, state.x, channel)
        state.t += tau
        state.step_count += 1
        details["fired_channel"] = int(channel) if applied else None
        details["invalid_jump_skipped"] = not applied
        if applied:
            state.event_count += 1
        return StepResult(
            advanced_time=tau,
            event_occurred=applied,
            channel_id=channel if applied else None,
            propensity_sum=total,
            tau=tau,
            details=details,
        )

    def _mixed_step(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        beta: np.ndarray,
        beta_min: float,
        beta_max: float,
    ) -> StepResult:
        network = context.network
        duration = min(self._current_dt_cle(), dt)
        observed = self._rounded_nonnegative(state.x)
        base_jump = self._propensities_for_x(network, observed, state.t)
        base_jump = self._clean_propensities(base_jump, "jump propensities")
        lambda_jump = self._clean_propensities(beta * base_jump, "split jump propensities")
        total_jump = float(np.sum(lambda_jump))

        tau = float("inf")
        sampled_channel: int | None = None
        if total_jump > 0.0:
            tau = float(context.rng.exponential(1.0 / total_jump))
            sampled_channel = _sample_channel(
                np.arange(network.n_channels, dtype=np.int64),
                lambda_jump,
                total_jump,
                context.rng,
            )

        if tau < duration and sampled_channel is not None:
            state.x[:] = self._cle_increment(network, state.x, beta, tau, context.rng)
            applied = self._apply_jump_safely(network, state.x, sampled_channel)
            state.t += tau
            state.step_count += 1
            if applied:
                state.event_count += 1
            return StepResult(
                advanced_time=tau,
                event_occurred=applied,
                channel_id=sampled_channel if applied else None,
                propensity_sum=total_jump,
                tau=tau,
                details={
                    "mode": "mixed_jump",
                    "fired_channel": int(sampled_channel) if applied else None,
                    "beta_min": beta_min,
                    "beta_max": beta_max,
                    "total_jump_propensity": total_jump,
                    "total_cle_propensity": self._last_total_cle_propensity,
                    "n_clipped": self._last_n_clipped,
                    "stepper_dt": tau,
                    "reaction_interval_dt": self._reaction_interval_dt,
                    "invalid_jump_skipped": not applied,
                    "continuous_channel_abs_increments": self._last_continuous_channel_abs_increments.copy(),
                },
            )

        state.x[:] = self._cle_increment(network, state.x, beta, duration, context.rng)
        state.t += duration
        state.step_count += 1
        return StepResult(
            advanced_time=duration,
            event_occurred=False,
            propensity_sum=total_jump,
            tau=None if np.isinf(tau) else tau,
            details={
                "mode": "mixed_cle",
                "fired_channel": None,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "total_jump_propensity": total_jump,
                "total_cle_propensity": self._last_total_cle_propensity,
                "n_clipped": self._last_n_clipped,
                "stepper_dt": duration,
                "reaction_interval_dt": self._reaction_interval_dt,
                "continuous_channel_abs_increments": self._last_continuous_channel_abs_increments.copy(),
            },
        )

    def _cle_increment(
        self,
        network: ReactionNetworkData,
        x_float: np.ndarray,
        beta: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if dt <= 0.0:
            self._last_n_clipped = 0
            self._last_total_cle_propensity = 0.0
            self._last_continuous_channel_abs_increments = np.zeros(network.n_channels, dtype=float)
            return self._float_nonnegative(x_float)

        x0 = self._float_nonnegative(x_float)
        prop = self._propensities_for_x(network, x0, 0.0)
        prop = self._clean_propensities(prop, "CLE propensities")
        prop_cle = self._clean_propensities((1.0 - beta) * prop, "split CLE propensities")
        self._last_total_cle_propensity = float(np.sum(prop_cle))

        means = prop_cle * float(dt)
        amounts = means + np.sqrt(np.maximum(means, 0.0)) * rng.normal(size=network.n_channels)
        self._last_continuous_channel_abs_increments = np.abs(amounts).astype(float, copy=False)
        increment = amounts @ self._stoichiometry_matrix(network)
        x_new = x0 + increment
        if not np.all(np.isfinite(x_new)):
            raise ValueError("CLE increment produced NaN or inf state values")

        negative = x_new < 0.0
        self._last_n_clipped = int(np.count_nonzero(negative))
        if np.any(negative):
            if not self.config.clip_negative:
                raise ValueError("CLE increment produced negative state values")
            x_new = np.maximum(x_new, 0.0)
        return x_new

    def _channel_betas(self, network: ReactionNetworkData, x: np.ndarray) -> np.ndarray:
        beta = np.zeros(network.n_channels, dtype=float)
        for channel_id in range(network.n_channels):
            if network.get_channel_block(channel_id) == ChannelBlock.INFLOW:
                beta[channel_id] = 1.0
                continue
            relevant_species = _channel_relevant_species(network, channel_id)
            if not relevant_species:
                beta[channel_id] = 0.0
                continue
            beta[channel_id] = max(
                _species_beta(float(x[int(sid)]), self.config.i1, self.config.i2)
                for sid in relevant_species
            )
        return beta

    def _stoichiometry_matrix(self, network: ReactionNetworkData) -> np.ndarray:
        key = id(network)
        cached = self._nu_cache.get(key)
        if cached is not None and cached.shape == (network.n_channels, network.n_species):
            return cached

        nu = np.zeros((network.n_channels, network.n_species), dtype=float)
        for channel_id in range(network.n_channels):
            before = np.zeros(network.n_species, dtype=float)
            after = before.copy()
            network.apply_channel_delta(after, channel_id, 1.0)
            nu[channel_id, :] = after - before
        self._nu_cache[key] = nu
        return nu

    def _rounded_nonnegative(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if self.config.round_mode == "nearest":
            rounded = np.rint(values)
        elif self.config.round_mode == "floor":
            rounded = np.floor(values)
        else:
            rounded = np.ceil(values)
        return np.maximum(rounded, 0.0)

    def _float_nonnegative(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("state contains NaN or inf values")
        return np.maximum(values, 0.0)

    def _propensities_for_x(self, network: ReactionNetworkData, x: np.ndarray, t: float) -> np.ndarray:
        return network.compute_all_propensities(SystemState(t=float(t), x=np.asarray(x, dtype=float)))

    def _clean_propensities(self, propensities: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(propensities, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contain NaN or inf values")
        return np.maximum(values, 0.0)

    def _apply_jump_safely(self, network: ReactionNetworkData, x: np.ndarray, channel_id: int) -> bool:
        candidate = np.asarray(x, dtype=float).copy()
        network.apply_channel_delta(candidate, int(channel_id), 1.0)
        if not np.all(np.isfinite(candidate)):
            raise ValueError("discrete jump produced NaN or inf state values")
        if np.any(candidate < -self.config.beta_tol):
            return False
        if self.config.clip_negative:
            candidate = np.maximum(candidate, 0.0)
        x[:] = candidate
        return True

    def _maybe_update_reaction_interval_dt(self, network: ReactionNetworkData, state: SystemState) -> None:
        if not self.config.use_reaction_interval_dt:
            return
        should_update = (
            self._reaction_interval_dt is None
            or int(state.step_count) % self.config.reaction_interval_update_steps == 0
        )
        if not should_update:
            return
        interval = estimate_mean_reaction_interval(
            network,
            SystemState(t=float(state.t), x=self._float_nonnegative(state.x)),
        )
        if np.isfinite(interval) and interval > 0.0:
            self._reaction_interval_dt = float(interval * self.config.reaction_interval_scale)

    def _current_dt_cle(self) -> float:
        if self.config.use_reaction_interval_dt and self._reaction_interval_dt is not None:
            return self._reaction_interval_dt
        return self.config.dt_cle

    def _current_dt_macro(self) -> float:
        if self.config.use_reaction_interval_dt and self._reaction_interval_dt is not None:
            return self._reaction_interval_dt
        return self.config.effective_dt_macro


def estimate_mean_reaction_interval(network: ReactionNetworkData, state: SystemState) -> float:
    propensities = network.compute_all_propensities(state)
    if not np.all(np.isfinite(propensities)):
        raise ValueError("propensities contain NaN or inf values")
    total = float(np.sum(np.maximum(propensities, 0.0)))
    if total <= 0.0:
        return float("inf")
    return 1.0 / total


def _species_beta(x: float, i1: float, i2: float) -> float:
    value = float(x)
    if value <= float(i1):
        return 1.0
    if value >= float(i2):
        return 0.0
    return float((float(i2) - value) / (float(i2) - float(i1)))


def _channel_relevant_species(network: ReactionNetworkData, channel_id: int) -> list[int]:
    return [int(sid) for sid in network.get_channel_reactants(int(channel_id))]


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
