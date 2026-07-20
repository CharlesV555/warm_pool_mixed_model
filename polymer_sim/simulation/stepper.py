from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq
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
class OptimizedNRMConfig:
    use_dependency_graph: bool = True
    fallback_full_recompute: bool = True
    validate_nonnegative: bool = True
    propensity_tol: float = 0.0
    heap_rebuild_factor: float = 4.0
    diagnostics: bool = True

    def __post_init__(self) -> None:
        self.use_dependency_graph = bool(self.use_dependency_graph)
        self.fallback_full_recompute = bool(self.fallback_full_recompute)
        self.validate_nonnegative = bool(self.validate_nonnegative)
        self.propensity_tol = float(self.propensity_tol)
        self.heap_rebuild_factor = float(self.heap_rebuild_factor)
        self.diagnostics = bool(self.diagnostics)
        if self.propensity_tol < 0.0:
            raise ValueError("propensity_tol must be >= 0")
        if self.heap_rebuild_factor < 1.0:
            raise ValueError("heap_rebuild_factor must be >= 1")


@dataclass(slots=True)
class BlendedHybridConfig:
    i1: float = 10.0
    i2: float = 30.0
    dt_cle: float = 0.01
    dt_macro: float | None = None
    beta_tol: float = 1e-12
    round_mode: str = "nearest"
    clip_negative: bool = True
    beta_species_mode: str = "reactants_products"
    round_low_counts_after_cle: bool = True
    use_reaction_interval_dt: bool = False
    reaction_interval_update_steps: int = 100
    reaction_interval_scale: float = 1.0

    def __post_init__(self) -> None:
        self.i1 = float(self.i1)
        self.i2 = float(self.i2)
        self.dt_cle = float(self.dt_cle)
        self.dt_macro = None if self.dt_macro is None else float(self.dt_macro)
        self.beta_tol = float(self.beta_tol)
        self.beta_species_mode = str(self.beta_species_mode).lower()
        self.round_low_counts_after_cle = bool(self.round_low_counts_after_cle)
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
        if self.beta_species_mode not in {"reactants", "products", "reactants_products"}:
            raise ValueError("beta_species_mode must be 'reactants', 'products', or 'reactants_products'")
        if self.reaction_interval_update_steps <= 0:
            raise ValueError("reaction_interval_update_steps must be > 0")
        if self.reaction_interval_scale <= 0.0:
            raise ValueError("reaction_interval_scale must be > 0")

    @property
    def effective_dt_macro(self) -> float:
        return self.dt_cle if self.dt_macro is None else self.dt_macro


class BaseStepper(ABC):
    """所有模拟步进器的最小统一接口。

    `ExperimentRunner` 只依赖这个接口：给定当前 `SystemState`、本次最多
    推进的时间 `dt` 和 `StepperContext`，原地修改 state 并返回 `StepResult`。
    """

    @abstractmethod
    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        raise NotImplementedError


class SSAStepper(BaseStepper):
    """Gillespie Direct SSA 步进器。

    第一版采样仍使用线性扫描；propensity 计算路径已经支持缓存和 dependency
    graph 局部更新，因此单次反应后只重算受 changed species 影响的 channels。
    """

    def __init__(self, *, use_local_propensity_updates: bool = True):
        self.use_local_propensity_updates = bool(use_local_propensity_updates)
        self._propensity_cache: np.ndarray | None = None
        self._cache_network_id: int | None = None
        self._cache_state_array_id: int | None = None

    def invalidate_cache(self) -> None:
        """Discard cached propensities after external state or network changes."""

        self._propensity_cache = None
        self._cache_network_id = None
        self._cache_state_array_id = None

    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        network = context.network
        propensities = self._get_propensities(network, state)
        return self._step_from_channels(
            state,
            dt,
            context,
            np.arange(network.n_channels),
            propensities,
            update_propensity_cache=True,
        )

    def step_restricted(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        channels: np.ndarray,
        propensities: np.ndarray | None = None,
    ) -> StepResult:
        network = context.network
        all_propensities = self._get_propensities(network, state) if propensities is None else propensities
        return self._step_from_channels(
            state,
            dt,
            context,
            channels,
            all_propensities,
            update_propensity_cache=all_propensities is self._propensity_cache,
        )

    def _step_from_channels(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        channels: np.ndarray,
        propensities: np.ndarray,
        *,
        update_propensity_cache: bool = False,
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

        changed_species = network.get_channel_changed_species(chosen)
        network.apply_channel_update(state, chosen)
        state.t += tau
        state.step_count += 1
        state.event_count += 1
        if update_propensity_cache:
            self._update_cached_propensities(network, state, changed_species)
        return StepResult(
            advanced_time=tau,
            event_occurred=True,
            channel_id=chosen,
            propensity_sum=total,
            tau=tau,
        )

    def _get_propensities(self, network: ReactionNetworkData, state: SystemState) -> np.ndarray:
        if not self.use_local_propensity_updates:
            self.invalidate_cache()
            return network.compute_all_propensities(state)
        if not self._cache_matches(network, state):
            self._propensity_cache = network.compute_all_propensities(state)
            self._cache_network_id = id(network)
            self._cache_state_array_id = id(state.x)
        return self._propensity_cache

    def _cache_matches(self, network: ReactionNetworkData, state: SystemState) -> bool:
        return (
            self._propensity_cache is not None
            and self._cache_network_id == id(network)
            and self._cache_state_array_id == id(state.x)
            and self._propensity_cache.shape == (network.n_channels,)
        )

    def _update_cached_propensities(
        self,
        network: ReactionNetworkData,
        state: SystemState,
        changed_species: np.ndarray,
    ) -> None:
        if not self.use_local_propensity_updates or not self._cache_matches(network, state):
            return
        network.update_propensities_for_species(self._propensity_cache, state, changed_species)


class OptimizedNRMStepper(BaseStepper):
    """优化版 Next Reaction Method 步进器。

    每个 channel 维护一个真实时间轴上的下一次触发时间，并用优先队列取最近
    事件。反应触发后只重排自身和 dependency graph 中受 changed species
    影响的 channels；旧 heap entry 通过 version lazy deletion 跳过。
    """

    def __init__(self, config: OptimizedNRMConfig | None = None):
        self.config = config or OptimizedNRMConfig()
        self._propensities: np.ndarray | None = None
        self._scheduled_times: np.ndarray | None = None
        self._versions: np.ndarray | None = None
        self._heap: list[tuple[float, int, int]] = []
        self._propensity_sum = 0.0
        self._cache_network_id: int | None = None
        self._cache_state_array_id: int | None = None
        self._stale_pops = 0
        self._last_full_recompute = False
        self._last_dependency_graph_used = False

    def invalidate_cache(self) -> None:
        """Discard scheduled events after an external state or network change."""

        self._propensities = None
        self._scheduled_times = None
        self._versions = None
        self._heap = []
        self._propensity_sum = 0.0
        self._cache_network_id = None
        self._cache_state_array_id = None
        self._stale_pops = 0
        self._last_full_recompute = False
        self._last_dependency_graph_used = False

    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        network = context.network
        rng = context.rng
        self._ensure_initialized(network, state, rng)

        total_before = self._propensity_sum
        next_item = self._peek_next_valid()
        if total_before <= self.config.propensity_tol or next_item is None:
            state.t += float(dt)
            state.step_count += 1
            return StepResult(
                advanced_time=float(dt),
                event_occurred=False,
                propensity_sum=max(float(total_before), 0.0),
                details=self._details(n_affected_channels=0),
            )

        fire_time, chosen = next_item
        tau = max(float(fire_time) - float(state.t), 0.0)
        if tau > float(dt):
            state.t += float(dt)
            state.step_count += 1
            return StepResult(
                advanced_time=float(dt),
                event_occurred=False,
                propensity_sum=max(float(total_before), 0.0),
                tau=tau,
                details=self._details(n_affected_channels=0),
            )

        heapq.heappop(self._heap)
        old_time = float(state.t)
        state.t = float(fire_time)
        changed_species = network.get_channel_changed_species(chosen)
        network.apply_channel_update(state, chosen)
        if self.config.validate_nonnegative and np.any(state.x < -1e-12):
            raise ValueError("optimized NRM produced negative species counts")
        if self.config.validate_nonnegative:
            np.maximum(state.x, 0.0, out=state.x)

        affected = self._affected_channels(network, changed_species, chosen)
        self._reschedule_affected(network, state, rng, affected, fired_channel=chosen)
        self._maybe_rebuild_heap()

        state.step_count += 1
        state.event_count += 1
        return StepResult(
            advanced_time=float(state.t - old_time),
            event_occurred=True,
            channel_id=chosen,
            propensity_sum=max(float(total_before), 0.0),
            tau=float(state.t - old_time),
            details=self._details(n_affected_channels=int(affected.size)),
        )

    def _ensure_initialized(
        self,
        network: ReactionNetworkData,
        state: SystemState,
        rng: np.random.Generator,
    ) -> None:
        if self._cache_matches(network, state):
            return

        propensities = network.compute_all_propensities(state)
        propensities = self._clean_propensities(propensities)
        scheduled = np.full(network.n_channels, np.inf, dtype=float)
        versions = np.zeros(network.n_channels, dtype=np.int64)
        heap: list[tuple[float, int, int]] = []
        for channel_id, propensity in enumerate(propensities):
            if propensity <= self.config.propensity_tol:
                continue
            fire_time = self._sample_fire_time(float(state.t), float(propensity), rng)
            scheduled[channel_id] = fire_time
            heapq.heappush(heap, (fire_time, int(channel_id), int(versions[channel_id])))

        self._propensities = propensities
        self._scheduled_times = scheduled
        self._versions = versions
        self._heap = heap
        self._propensity_sum = float(np.sum(propensities))
        self._cache_network_id = id(network)
        self._cache_state_array_id = id(state.x)
        self._stale_pops = 0
        self._last_full_recompute = True
        self._last_dependency_graph_used = False

    def _cache_matches(self, network: ReactionNetworkData, state: SystemState) -> bool:
        return (
            self._propensities is not None
            and self._scheduled_times is not None
            and self._versions is not None
            and self._cache_network_id == id(network)
            and self._cache_state_array_id == id(state.x)
            and self._propensities.shape == (network.n_channels,)
            and self._scheduled_times.shape == (network.n_channels,)
            and self._versions.shape == (network.n_channels,)
        )

    def _peek_next_valid(self) -> tuple[float, int] | None:
        scheduled = self._scheduled_times
        versions = self._versions
        if scheduled is None or versions is None:
            return None
        while self._heap:
            fire_time, channel_id, version = self._heap[0]
            if (
                int(version) == int(versions[int(channel_id)])
                and np.isfinite(fire_time)
                and float(fire_time) == float(scheduled[int(channel_id)])
            ):
                return float(fire_time), int(channel_id)
            heapq.heappop(self._heap)
            self._stale_pops += 1
        return None

    def _affected_channels(
        self,
        network: ReactionNetworkData,
        changed_species: np.ndarray,
        fired_channel: int,
    ) -> np.ndarray:
        self._last_full_recompute = False
        self._last_dependency_graph_used = False
        if self.config.use_dependency_graph:
            try:
                affected = network.affected_channels_for_species(changed_species)
                self._last_dependency_graph_used = True
            except Exception:
                if not self.config.fallback_full_recompute:
                    raise
                affected = np.arange(network.n_channels, dtype=np.int64)
                self._last_full_recompute = True
        elif self.config.fallback_full_recompute:
            affected = np.arange(network.n_channels, dtype=np.int64)
            self._last_full_recompute = True
        else:
            affected = np.empty(0, dtype=np.int64)

        if affected.size == 0:
            return np.asarray([int(fired_channel)], dtype=np.int64)
        if np.any(affected == int(fired_channel)):
            return np.asarray(affected, dtype=np.int64)
        return np.unique(np.concatenate((affected, np.asarray([int(fired_channel)], dtype=np.int64)))).astype(
            np.int64,
            copy=False,
        )

    def _reschedule_affected(
        self,
        network: ReactionNetworkData,
        state: SystemState,
        rng: np.random.Generator,
        affected: np.ndarray,
        *,
        fired_channel: int,
    ) -> None:
        propensities = self._require_propensities()
        scheduled = self._require_scheduled_times()
        versions = self._require_versions()
        channels = np.asarray(affected, dtype=np.int64)
        old_propensities = propensities[channels].copy()
        old_scheduled = scheduled[channels].copy()
        new_propensities = self._clean_propensities(network.compute_propensities_for_channels(channels, state))
        propensities[channels] = new_propensities
        self._propensity_sum += float(np.sum(new_propensities - old_propensities))
        if abs(self._propensity_sum) <= 1e-12:
            self._propensity_sum = 0.0

        now = float(state.t)
        for index, channel_id in enumerate(channels):
            cid = int(channel_id)
            old_a = float(old_propensities[index])
            new_a = float(new_propensities[index])
            old_time = float(old_scheduled[index])
            versions[cid] += 1
            if cid == int(fired_channel):
                next_time = self._sample_fire_time_or_inf(now, new_a, rng)
            else:
                next_time = self._transformed_fire_time(now, old_time, old_a, new_a, rng)
            scheduled[cid] = next_time
            if np.isfinite(next_time):
                heapq.heappush(self._heap, (float(next_time), cid, int(versions[cid])))

    def _transformed_fire_time(
        self,
        now: float,
        old_time: float,
        old_propensity: float,
        new_propensity: float,
        rng: np.random.Generator,
    ) -> float:
        if new_propensity <= self.config.propensity_tol:
            return float("inf")
        if old_propensity <= self.config.propensity_tol or not np.isfinite(old_time):
            return self._sample_fire_time(now, new_propensity, rng)
        remaining = max(float(old_time) - float(now), 0.0)
        return float(now) + (old_propensity / new_propensity) * remaining

    def _sample_fire_time_or_inf(
        self,
        now: float,
        propensity: float,
        rng: np.random.Generator,
    ) -> float:
        if propensity <= self.config.propensity_tol:
            return float("inf")
        return self._sample_fire_time(now, propensity, rng)

    def _sample_fire_time(self, now: float, propensity: float, rng: np.random.Generator) -> float:
        return float(now) + float(rng.exponential(1.0 / float(propensity)))

    def _maybe_rebuild_heap(self) -> None:
        scheduled = self._scheduled_times
        versions = self._versions
        if scheduled is None or versions is None:
            return
        limit = max(int(np.ceil(self.config.heap_rebuild_factor * scheduled.size)), scheduled.size + 1)
        if len(self._heap) <= limit:
            return
        self._heap = [
            (float(fire_time), int(channel_id), int(versions[int(channel_id)]))
            for channel_id, fire_time in enumerate(scheduled)
            if np.isfinite(fire_time)
        ]
        heapq.heapify(self._heap)
        self._stale_pops = 0

    def _clean_propensities(self, propensities: np.ndarray) -> np.ndarray:
        values = np.asarray(propensities, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("optimized NRM propensities contain NaN or inf values")
        return np.maximum(values, 0.0)

    def _details(self, *, n_affected_channels: int) -> dict[str, Any] | None:
        if not self.config.diagnostics:
            return None
        return {
            "mode": "optimized_nrm",
            "n_affected_channels": int(n_affected_channels),
            "heap_entries": int(len(self._heap)),
            "stale_heap_pops": int(self._stale_pops),
            "dependency_graph_used": bool(self._last_dependency_graph_used),
            "full_recompute": bool(self._last_full_recompute),
        }

    def _require_propensities(self) -> np.ndarray:
        if self._propensities is None:
            raise RuntimeError("optimized NRM is not initialized")
        return self._propensities

    def _require_scheduled_times(self) -> np.ndarray:
        if self._scheduled_times is None:
            raise RuntimeError("optimized NRM is not initialized")
        return self._scheduled_times

    def _require_versions(self) -> np.ndarray:
        if self._versions is None:
            raise RuntimeError("optimized NRM is not initialized")
        return self._versions


class CLEStepper(BaseStepper):
    """Chemical Langevin Equation 的最小步进器。

    它只推进被 partition 标为 fast 的 channels。当前实现是显式 Euler-Maruyama
    风格的正态近似增量，并在状态出现负数时裁剪到 0。
    """

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
    """固定 fast/slow split 的基础 hybrid 步进器。

    fast channels 用 CLE 连续推进，slow channels 用 Direct SSA 采样一个离散
    事件。这个类主要是接口骨架；更细的平滑分裂逻辑在 BlendedHybridStepper 中。
    """

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
    """基于 beta 权重的 SSA/CLE blending 步进器。

    beta=1 表示该 channel 完全按离散 jump 处理，beta=0 表示完全按 CLE 连续
    处理，中间值会把 propensity 拆成 jump 与 CLE 两部分。当前离散部分每个
    step 至多采样一个 Direct SSA 事件。
    """

    def __init__(self, config: BlendedHybridConfig | None = None):
        self.config = config or BlendedHybridConfig()
        self._nu_cache: dict[int, np.ndarray] = {}
        self._last_n_clipped = 0
        self._last_n_low_count_rounded = 0
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
                "n_low_count_rounded": self._last_n_low_count_rounded,
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
            "n_low_count_rounded": 0,
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
                    "n_low_count_rounded": self._last_n_low_count_rounded,
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
                "n_low_count_rounded": self._last_n_low_count_rounded,
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
            self._last_n_low_count_rounded = 0
            self._last_total_cle_propensity = 0.0
            self._last_continuous_channel_abs_increments = np.zeros(network.n_channels, dtype=float)
            return self._float_nonnegative(x_float)

        self._last_n_low_count_rounded = 0
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
        if self.config.round_low_counts_after_cle:
            x_new = self._round_low_count_changed_species(x_new, increment)
        return x_new

    def _channel_betas(self, network: ReactionNetworkData, x: np.ndarray) -> np.ndarray:
        beta = np.zeros(network.n_channels, dtype=float)
        for channel_id in range(network.n_channels):
            if network.get_channel_block(channel_id) == ChannelBlock.INFLOW:
                beta[channel_id] = 0.0
                continue
            relevant_species = _channel_relevant_species(network, channel_id, self.config.beta_species_mode)
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

    def _round_low_count_changed_species(self, x: np.ndarray, increment: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        changed = np.abs(np.asarray(increment, dtype=float)) > self.config.beta_tol
        low_count = values <= self.config.i1 + self.config.beta_tol
        should_round = changed & low_count
        self._last_n_low_count_rounded = int(np.count_nonzero(should_round))
        if not np.any(should_round):
            return values
        rounded = values.copy()
        rounded[should_round] = self._rounded_nonnegative(rounded[should_round])
        return rounded

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


class NRMBlendedHybridStepper(BlendedHybridStepper):
    """基于 beta 权重的 NRM/CLE blending 步进器。

    使用与 ``BlendedHybridStepper`` 相同的 beta 定义：

    - 所有 beta == 0：走大步 CLE macro step；
    - 所有 beta == 1：委托给 ``OptimizedNRMStepper``；
    - mixed beta：把 propensity 拆为 ``beta * a`` 的 NRM jump 部分和
      ``(1 - beta) * a`` 的 CLE 连续部分。在一个 ``dt_cle`` 窗口内，
      jump 部分用 NRM priority queue 排事件时间，CLE 部分在事件间隔上推进。

    mixed 分支是一阶 operator splitting 近似：当前 ``dt_cle`` 内 beta 固定，
    jump schedule 来自 jump 子系统。它是高吞吐 hybrid 骨架，不是 pure NRM 的
    严格等价替代。
    """

    def __init__(
        self,
        config: BlendedHybridConfig | None = None,
        nrm_config: OptimizedNRMConfig | None = None,
    ):
        super().__init__(config)
        self.nrm_config = nrm_config or OptimizedNRMConfig()
        self._nrm = OptimizedNRMStepper(self.nrm_config)

    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        if dt <= 0.0:
            return StepResult(advanced_time=0.0, event_occurred=False, details={"mode": "nrm_blended_no_dt"})

        network = context.network
        self._maybe_update_reaction_interval_dt(network, state)
        x_float = self._float_nonnegative(state.x)
        beta = self._channel_betas(network, x_float)
        beta_min = float(np.min(beta)) if beta.size else 0.0
        beta_max = float(np.max(beta)) if beta.size else 0.0

        if beta_max <= self.config.beta_tol:
            return self._pure_cle_step(state, float(dt), context, beta, beta_min, beta_max)
        if beta_min >= 1.0 - self.config.beta_tol:
            return self._pure_nrm_step(state, float(dt), context, beta_min, beta_max)
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
        result = super()._pure_cle_step(state, dt, context, beta, beta_min, beta_max)
        self._nrm.invalidate_cache()
        if result.details is not None:
            result.details["mode"] = "nrm_blended_cle"
        return result

    def _pure_nrm_step(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        beta_min: float,
        beta_max: float,
    ) -> StepResult:
        duration = min(self._current_dt_macro(), dt)
        state.x[:] = self._rounded_nonnegative(state.x)
        result = self._nrm.step(state, duration, context)
        details = dict(result.details or {})
        details.update(
            {
                "mode": "nrm_blended_nrm",
                "fired_channel": result.channel_id,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "total_jump_propensity": result.propensity_sum,
                "total_cle_propensity": 0.0,
                "stepper_dt": duration,
                "reaction_interval_dt": self._reaction_interval_dt,
            }
        )
        return StepResult(
            advanced_time=result.advanced_time,
            event_occurred=result.event_occurred,
            channel_id=result.channel_id,
            propensity_sum=result.propensity_sum,
            tau=result.tau,
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
        rng = context.rng
        duration = min(self._current_dt_cle(), dt)
        start_time = float(state.t)
        end_time = start_time + duration
        x_work = self._float_nonnegative(state.x).copy()

        base_initial = self._propensities_for_x(network, self._rounded_nonnegative(x_work), start_time)
        base_initial = self._clean_propensities(base_initial, "initial mixed propensities")
        total_jump_initial = float(np.sum(self._clean_propensities(beta * base_initial, "initial jump propensities")))
        total_cle_initial = float(
            np.sum(self._clean_propensities((1.0 - beta) * base_initial, "initial CLE propensities"))
        )

        scheduled_events, schedule_details = self._schedule_mixed_nrm_events(
            network,
            x_work,
            start_time,
            duration,
            beta,
            rng,
        )

        continuous_abs_total = np.zeros(network.n_channels, dtype=float)
        applied_event_ids: list[int] = []
        applied_event_times: list[float] = []
        invalid_jumps = 0
        n_clipped_total = 0
        n_low_count_rounded_total = 0
        current_time = start_time

        for event_time, channel_id in scheduled_events:
            event_t = float(event_time)
            segment_dt = max(event_t - current_time, 0.0)
            if segment_dt > 0.0:
                x_work = self._cle_increment(network, x_work, beta, segment_dt, rng)
                continuous_abs_total += self._last_continuous_channel_abs_increments
                n_clipped_total += self._last_n_clipped
                n_low_count_rounded_total += self._last_n_low_count_rounded
            applied = self._apply_jump_safely(network, x_work, int(channel_id))
            if applied:
                applied_event_ids.append(int(channel_id))
                applied_event_times.append(event_t)
            else:
                invalid_jumps += 1
            current_time = event_t

        tail_dt = max(end_time - current_time, 0.0)
        if tail_dt > 0.0 or not scheduled_events:
            x_work = self._cle_increment(network, x_work, beta, tail_dt, rng)
            continuous_abs_total += self._last_continuous_channel_abs_increments
            n_clipped_total += self._last_n_clipped
            n_low_count_rounded_total += self._last_n_low_count_rounded

        state.x[:] = self._float_nonnegative(x_work)
        state.t = end_time
        state.step_count += 1
        state.event_count += len(applied_event_ids)
        self._nrm.invalidate_cache()

        self._last_n_clipped = int(n_clipped_total)
        self._last_n_low_count_rounded = int(n_low_count_rounded_total)
        self._last_total_cle_propensity = total_cle_initial
        self._last_continuous_channel_abs_increments = continuous_abs_total

        first_tau = None if not applied_event_times else float(applied_event_times[0] - start_time)
        first_channel = None if not applied_event_ids else int(applied_event_ids[0])
        return StepResult(
            advanced_time=duration,
            event_occurred=bool(applied_event_ids),
            channel_id=first_channel,
            propensity_sum=total_jump_initial,
            tau=first_tau,
            details={
                "mode": "nrm_blended_mixed",
                "fired_channel": first_channel,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "total_jump_propensity": total_jump_initial,
                "total_cle_propensity": total_cle_initial,
                "n_clipped": self._last_n_clipped,
                "n_low_count_rounded": self._last_n_low_count_rounded,
                "stepper_dt": duration,
                "reaction_interval_dt": self._reaction_interval_dt,
                "n_scheduled_discrete_events": int(len(scheduled_events)),
                "n_applied_discrete_events": int(len(applied_event_ids)),
                "n_invalid_jump_skipped": int(invalid_jumps),
                "discrete_event_ids": list(applied_event_ids),
                "discrete_event_times": list(applied_event_times),
                "continuous_channel_abs_increments": continuous_abs_total.copy(),
                **schedule_details,
            },
        )

    def _schedule_mixed_nrm_events(
        self,
        network: ReactionNetworkData,
        x: np.ndarray,
        start_time: float,
        duration: float,
        beta: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[list[tuple[float, int]], dict[str, Any]]:
        if duration <= 0.0:
            return [], {"nrm_dependency_graph_used": False, "nrm_full_recompute": False, "nrm_affected_updates": 0}

        now = float(start_time)
        end_time = now + float(duration)
        x_jump = self._rounded_nonnegative(x).copy()
        jump_state = SystemState(t=now, x=x_jump)
        propensities = self._clean_propensities(
            beta * network.compute_all_propensities(jump_state),
            "mixed NRM jump propensities",
        )
        scheduled = np.full(network.n_channels, np.inf, dtype=float)
        versions = np.zeros(network.n_channels, dtype=np.int64)
        heap: list[tuple[float, int, int]] = []
        for channel_id, propensity in enumerate(propensities):
            if float(propensity) <= self.nrm_config.propensity_tol:
                continue
            fire_time = self._sample_split_nrm_fire_time(now, float(propensity), rng)
            scheduled[channel_id] = fire_time
            heapq.heappush(heap, (fire_time, int(channel_id), int(versions[channel_id])))

        events: list[tuple[float, int]] = []
        stale_pops = 0
        affected_updates = 0
        dependency_graph_used = False
        full_recompute = False

        while True:
            item, stale_delta = self._pop_next_split_nrm_event(heap, scheduled, versions)
            stale_pops += stale_delta
            if item is None:
                break
            fire_time, channel_id = item
            if float(fire_time) > end_time + self.config.beta_tol:
                break

            now = float(fire_time)
            jump_state = SystemState(t=now, x=x_jump)
            applied = self._apply_jump_safely(network, x_jump, int(channel_id))
            if applied:
                events.append((now, int(channel_id)))
                changed_species = network.get_channel_changed_species(int(channel_id))
            else:
                changed_species = np.empty(0, dtype=np.int64)

            affected = self._split_nrm_affected_channels(network, changed_species, int(channel_id))
            affected_updates += int(affected.size)
            if affected.size == network.n_channels:
                full_recompute = True
            elif affected.size > 1:
                dependency_graph_used = True
            self._reschedule_split_nrm_channels(
                network,
                jump_state,
                rng,
                beta,
                affected,
                fired_channel=int(channel_id),
                propensities=propensities,
                scheduled=scheduled,
                versions=versions,
                heap=heap,
                now=now,
            )

        return events, {
            "nrm_dependency_graph_used": bool(dependency_graph_used),
            "nrm_full_recompute": bool(full_recompute),
            "nrm_stale_heap_pops": int(stale_pops),
            "nrm_affected_updates": int(affected_updates),
        }

    def _pop_next_split_nrm_event(
        self,
        heap: list[tuple[float, int, int]],
        scheduled: np.ndarray,
        versions: np.ndarray,
    ) -> tuple[tuple[float, int] | None, int]:
        stale_pops = 0
        while heap:
            fire_time, channel_id, version = heapq.heappop(heap)
            cid = int(channel_id)
            if (
                int(version) == int(versions[cid])
                and np.isfinite(fire_time)
                and float(fire_time) == float(scheduled[cid])
            ):
                return (float(fire_time), cid), stale_pops
            stale_pops += 1
        return None, stale_pops

    def _split_nrm_affected_channels(
        self,
        network: ReactionNetworkData,
        changed_species: np.ndarray,
        fired_channel: int,
    ) -> np.ndarray:
        if changed_species.size == 0:
            return np.asarray([int(fired_channel)], dtype=np.int64)
        if self.nrm_config.use_dependency_graph:
            try:
                affected = network.affected_channels_for_species(changed_species)
            except Exception:
                if not self.nrm_config.fallback_full_recompute:
                    raise
                affected = np.arange(network.n_channels, dtype=np.int64)
        elif self.nrm_config.fallback_full_recompute:
            affected = np.arange(network.n_channels, dtype=np.int64)
        else:
            affected = np.asarray([int(fired_channel)], dtype=np.int64)

        if np.any(affected == int(fired_channel)):
            return np.asarray(affected, dtype=np.int64)
        return np.unique(np.concatenate((affected, np.asarray([int(fired_channel)], dtype=np.int64)))).astype(
            np.int64,
            copy=False,
        )

    def _reschedule_split_nrm_channels(
        self,
        network: ReactionNetworkData,
        state: SystemState,
        rng: np.random.Generator,
        beta: np.ndarray,
        channels: np.ndarray,
        *,
        fired_channel: int,
        propensities: np.ndarray,
        scheduled: np.ndarray,
        versions: np.ndarray,
        heap: list[tuple[float, int, int]],
        now: float,
    ) -> None:
        affected = np.asarray(channels, dtype=np.int64)
        old_propensities = propensities[affected].copy()
        old_scheduled = scheduled[affected].copy()
        new_base = network.compute_propensities_for_channels(affected, state)
        new_propensities = self._clean_propensities(beta[affected] * new_base, "updated mixed NRM propensities")
        propensities[affected] = new_propensities

        for index, channel_id in enumerate(affected):
            cid = int(channel_id)
            old_a = float(old_propensities[index])
            new_a = float(new_propensities[index])
            old_time = float(old_scheduled[index])
            versions[cid] += 1
            if cid == int(fired_channel):
                next_time = self._sample_split_nrm_fire_time_or_inf(now, new_a, rng)
            else:
                next_time = self._transformed_split_nrm_fire_time(now, old_time, old_a, new_a, rng)
            scheduled[cid] = next_time
            if np.isfinite(next_time):
                heapq.heappush(heap, (float(next_time), cid, int(versions[cid])))

    def _transformed_split_nrm_fire_time(
        self,
        now: float,
        old_time: float,
        old_propensity: float,
        new_propensity: float,
        rng: np.random.Generator,
    ) -> float:
        if new_propensity <= self.nrm_config.propensity_tol:
            return float("inf")
        if old_propensity <= self.nrm_config.propensity_tol or not np.isfinite(old_time):
            return self._sample_split_nrm_fire_time(now, new_propensity, rng)
        remaining = max(float(old_time) - float(now), 0.0)
        return float(now) + (old_propensity / new_propensity) * remaining

    def _sample_split_nrm_fire_time_or_inf(
        self,
        now: float,
        propensity: float,
        rng: np.random.Generator,
    ) -> float:
        if propensity <= self.nrm_config.propensity_tol:
            return float("inf")
        return self._sample_split_nrm_fire_time(now, propensity, rng)

    def _sample_split_nrm_fire_time(self, now: float, propensity: float, rng: np.random.Generator) -> float:
        return float(now) + float(rng.exponential(1.0 / float(propensity)))


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


def _channel_relevant_species(network: ReactionNetworkData, channel_id: int, mode: str = "reactants_products") -> list[int]:
    if mode == "reactants":
        species = network.get_channel_reactants(int(channel_id))
    elif mode == "products":
        species = network.get_channel_products(int(channel_id))
    elif mode == "reactants_products":
        species = (*network.get_channel_reactants(int(channel_id)), *network.get_channel_products(int(channel_id)))
    else:
        raise ValueError("mode must be 'reactants', 'products', or 'reactants_products'")
    return [int(sid) for sid in sorted(set(int(sid) for sid in species))]


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
