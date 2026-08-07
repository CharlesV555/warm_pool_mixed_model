from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq
from time import perf_counter
from typing import Any

from line_profiler import profile
import numpy as np
from scipy.sparse import csr_matrix

from polymer_sim.core.elementary import ElementaryMassActionNetwork
from polymer_sim.core.enums import ChannelBlock
from polymer_sim.core.kernels import (
    NUMBA_AVAILABLE,
    collect_affected_channels_numba,
    rebuild_elementary_gillespie_cache_numba,
    refresh_elementary_gillespie_cache_numba,
)
from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState
from polymer_sim.partition.pdmp import (
    FixedPDMPPartitionStrategy,
    LinearCatalysisScalingPDMPPartitionStrategy,
    PDMPPartitionResult,
    PDMPPartitionStrategy,
    ScalingPDMPConfig,
    ScalingPDMPPartitionStrategy,
)
from polymer_sim.partition.strategies import BlendingStrategy, FixedPartitionStrategy, NoBlendingStrategy, PartitionStrategy


@dataclass(slots=True)
class StepperContext:
    network: ReactionNetworkData | ElementaryMassActionNetwork
    rng: np.random.Generator
    partition_strategy: PartitionStrategy | PDMPPartitionStrategy | None = None
    blending_strategy: BlendingStrategy | None = None
    fast_channels: np.ndarray | None = None
    wall_deadline: float | None = None


@dataclass(slots=True)
class StepResult:
    advanced_time: float
    event_occurred: bool
    channel_id: int | None = None
    propensity_sum: float = 0.0
    tau: float | None = None
    details: dict[str, Any] | None = None


@dataclass(slots=True)
class _AdaptiveCLEResult:
    x: np.ndarray
    continuous_abs: np.ndarray
    dt: float
    requested_dt: float
    rejected_attempts: int
    n_clipped: int
    n_low_count_rounded: int
    total_cle_propensity: float
    dt_after: float
    min_dt_reached: bool


@dataclass(slots=True)
class _FastCLEAdvanceResult:
    continuous_abs: np.ndarray
    advanced_time: float
    requested_dt: float
    rejected_attempts: int
    n_clipped: int
    dt_after: float | None
    min_dt_reached: bool


@dataclass(slots=True)
class _ChannelBetaLookup:
    relevant_species: np.ndarray
    relevant_mask: np.ndarray
    species_to_channels_indptr: np.ndarray
    species_to_channels_indices: np.ndarray
    propensity_extra_species_to_channels_indptr: np.ndarray
    propensity_extra_species_to_channels_indices: np.ndarray
    catalyst_species_mask: np.ndarray
    catalyst_species_to_channels_indptr: np.ndarray
    catalyst_species_to_channels_indices: np.ndarray
    n_channels: int
    n_species: int
    mode: str


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
class PDMPConfig:
    """Configuration for the adaptive PDMP stepper.

    The current implementation follows the paper's Algorithm 1 structure with a
    lightweight Euler integrator for continuous channels and integrated hazards
    for discrete channels.  A later implementation can replace the Euler loop
    with an ODE solver plus event root finding without changing the runner
    interface.
    """

    ode_step: float = 0.01
    adaptive: bool = True
    repartition_on_event: bool = True
    repartition_on_bounds: bool = True
    validate_nonnegative: bool = True
    hazard_tol: float = 1e-12
    diagnostics: bool = True
    discrete_event_method: str = "auto"
    use_discrete_event_heap: bool = False
    use_local_propensity_updates: bool = True
    local_propensity_full_recompute_fraction: float = 0.5
    heap_rebuild_factor: float = 4.0
    kernel_backend: str = "auto"

    def __post_init__(self) -> None:
        self.ode_step = float(self.ode_step)
        self.adaptive = bool(self.adaptive)
        self.repartition_on_event = bool(self.repartition_on_event)
        self.repartition_on_bounds = bool(self.repartition_on_bounds)
        self.validate_nonnegative = bool(self.validate_nonnegative)
        self.hazard_tol = float(self.hazard_tol)
        self.diagnostics = bool(self.diagnostics)
        self.discrete_event_method = _normalize_pdmp_discrete_event_method(
            self.discrete_event_method,
            legacy_use_heap=bool(self.use_discrete_event_heap),
        )
        # Keep the historical boolean aligned for old call sites and metadata.
        self.use_discrete_event_heap = self.discrete_event_method == "nrm_heap"
        self.use_discrete_event_heap = bool(self.use_discrete_event_heap)
        self.use_local_propensity_updates = bool(self.use_local_propensity_updates)
        self.local_propensity_full_recompute_fraction = float(self.local_propensity_full_recompute_fraction)
        self.heap_rebuild_factor = float(self.heap_rebuild_factor)
        self.kernel_backend = str(self.kernel_backend).lower()
        if self.ode_step <= 0.0:
            raise ValueError("ode_step must be > 0")
        if self.hazard_tol < 0.0:
            raise ValueError("hazard_tol must be >= 0")
        if not (0.0 <= self.local_propensity_full_recompute_fraction <= 1.0):
            raise ValueError("local_propensity_full_recompute_fraction must be in [0, 1]")
        if self.heap_rebuild_factor < 1.0:
            raise ValueError("heap_rebuild_factor must be >= 1")
        if self.kernel_backend not in {"auto", "python", "numba"}:
            raise ValueError("kernel_backend must be 'auto', 'python', or 'numba'")
        if self.kernel_backend == "numba" and not NUMBA_AVAILABLE:
            raise ValueError("kernel_backend='numba' requires the numba package")


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
    beta_compute_mode: str = "beta_fully_compute"
    round_low_counts_after_cle: bool = True
    strict_int_for_CLE: bool = False
    use_reaction_interval_dt: bool = False
    reaction_interval_update_steps: int = 100
    reaction_interval_scale: float = 1.0
    adaptive_cle_dt: bool = True
    cle_dt_min: float = 1e-12
    cle_dt_max: float | None = None
    cle_dt_shrink_factor: float = 0.5
    cle_dt_growth_factor: float = 2.0
    cle_dt_max_retries: int = 20

    def __post_init__(self) -> None:
        self.i1 = float(self.i1)
        self.i2 = float(self.i2)
        self.dt_cle = float(self.dt_cle)
        self.dt_macro = None if self.dt_macro is None else float(self.dt_macro)
        self.beta_tol = float(self.beta_tol)
        self.clip_negative = bool(self.clip_negative)
        self.use_reaction_interval_dt = bool(self.use_reaction_interval_dt)
        self.beta_species_mode = str(self.beta_species_mode).lower()
        self.beta_compute_mode = str(self.beta_compute_mode).lower()
        self.round_low_counts_after_cle = bool(self.round_low_counts_after_cle)
        self.strict_int_for_CLE = bool(self.strict_int_for_CLE)
        self.reaction_interval_update_steps = int(self.reaction_interval_update_steps)
        self.reaction_interval_scale = float(self.reaction_interval_scale)
        self.adaptive_cle_dt = bool(self.adaptive_cle_dt)
        self.cle_dt_min = float(self.cle_dt_min)
        self.cle_dt_max = None if self.cle_dt_max is None else float(self.cle_dt_max)
        self.cle_dt_shrink_factor = float(self.cle_dt_shrink_factor)
        self.cle_dt_growth_factor = float(self.cle_dt_growth_factor)
        self.cle_dt_max_retries = int(self.cle_dt_max_retries)
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
        if self.beta_compute_mode not in {"beta_fully_compute", "beta_compute_by_state_difference"}:
            raise ValueError(
                "beta_compute_mode must be 'beta_fully_compute' or 'beta_compute_by_state_difference'"
            )
        if self.reaction_interval_update_steps <= 0:
            raise ValueError("reaction_interval_update_steps must be > 0")
        if self.reaction_interval_scale <= 0.0:
            raise ValueError("reaction_interval_scale must be > 0")
        if self.cle_dt_min <= 0.0:
            raise ValueError("cle_dt_min must be > 0")
        if self.cle_dt_max is not None and self.cle_dt_max < self.cle_dt_min:
            raise ValueError("cle_dt_max must be >= cle_dt_min when provided")
        if not (0.0 < self.cle_dt_shrink_factor < 1.0):
            raise ValueError("cle_dt_shrink_factor must be in (0, 1)")
        if self.cle_dt_growth_factor < 1.0:
            raise ValueError("cle_dt_growth_factor must be >= 1")
        if self.cle_dt_max_retries < 0:
            raise ValueError("cle_dt_max_retries must be >= 0")

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
        # for channel_id, propensity in zip(selected_channels, selected_prop):
        #     cumulative += float(propensity)
        #     if cumulative >= threshold:
        #         chosen = int(channel_id)
        #         break
        # 这里尝试加速。有效果哦
        cum = np.cumsum(selected_prop)
        chosen_idx = np.searchsorted(cum, threshold)
        chosen = selected_channels[chosen_idx]

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


def _pdmp_partition_strategy_from_method(
    partition_method: str,
    partition_config: ScalingPDMPConfig | None,
) -> PDMPPartitionStrategy:
    method = str(partition_method).lower()
    config = partition_config or ScalingPDMPConfig()
    if method in {"scaling", "scaling_lp", "algorithm3"}:
        return ScalingPDMPPartitionStrategy(config)
    if method in {"linear_catalysis", "linear_catalysis_scaling", "linear_catalysis_scaling_lp"}:
        return LinearCatalysisScalingPDMPPartitionStrategy(config)
    raise ValueError(
        "partition_method must be one of "
        "'scaling', 'scaling_lp', 'algorithm3', "
        "'linear_catalysis', 'linear_catalysis_scaling', or 'linear_catalysis_scaling_lp'"
    )


class PDMPStepper(BaseStepper):
    """Adaptive PDMP stepper for mass-action and direct-catalysis network views.

    This implements the Algorithm-1 execution pattern:
    - continuous channels contribute deterministic drift;
    - discrete channels keep independent exponential firing thresholds;
    - discrete hazards are accumulated while the continuous state evolves;
    - adaptive repartitioning can run after jumps or when scaling bounds are left.

    The default scaling partition follows the elementary mass-action view.  Set
    ``partition_method="linear_catalysis_scaling"`` to run the direct
    effective-catalysis scaling partition on ``ReactionNetworkData``.
    """

    def __init__(
        self,
        partition_strategy: PDMPPartitionStrategy | PartitionStrategy | None = None,
        partition_method: str = "scaling",
        partition_config: ScalingPDMPConfig | None = None,
        config: PDMPConfig | None = None,
    ):
        self.partition_method = str(partition_method)
        self.partition_config = partition_config
        self.partition_strategy = partition_strategy or _pdmp_partition_strategy_from_method(
            self.partition_method,
            self.partition_config,
        )
        self.config = config or PDMPConfig()
        self._invalidated = False

    def invalidate_cache(self) -> None:
        self._invalidated = True

    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        network = self._pdmp_network(context.network)
        duration = float(dt)
        if not np.isfinite(duration):
            # A wall-clock-limited run often passes t_end=inf and dt=None.
            # PDMP must still return control to the runner frequently so the
            # runner can enforce max_runtime_seconds and record progress.
            duration = float(self.config.ode_step)
        if duration <= 0.0:
            return StepResult(advanced_time=0.0, event_occurred=False, details={"mode": "pdmp_no_dt"})

        rng = context.rng
        store = self._ensure_runtime_store(state, network)
        repartitions = 0
        # Algorithm 2: initial adaptation.
        # Pseudocode:
        #   call adaptation, get RC, RD, SC, SD, RQ and bounds
        # Current implementation:
        #   _compute_partition(...) calls the configured PDMPPartitionStrategy.
        #   For Algorithm 3 this is ScalingPDMPPartitionStrategy.partition(...).
        #   The result stores RC as continuous_channels, RD as discrete_channels,
        #   SC/SD as continuous_species/discrete_species, and scale bounds.
        #   RQ is stored as rq_channels: discrete jumps that change a
        #   continuous species and therefore force immediate adaptation.
        if self._invalidated or store.get("partition") is None:
            # External invalidation means either the caller changed the state or
            # the network/stepper configuration changed.  Do not trust any
            # previously cached propensity array in that case.
            if self._invalidated:
                self._mark_propensities_dirty(store, "external invalidation")
            propensities = self._get_propensities(network, state, store, reason="initial adaptation")
            partition = self._compute_partition(network, state, propensities, context)
            self._install_partition(store, partition, rng, current_time=float(state.t), propensities=propensities, reset_all=True)
            self._invalidated = False
            repartitions += 1

        if self._uses_discrete_event_gillespie():
            return self._step_gillespie_integrated_hazard(
                state,
                duration,
                context,
                network,
                store,
                rng,
                initial_repartitions=repartitions,
            )

        start_time = float(state.t)
        end_time = start_time + duration
        continuous_abs_total = np.zeros(network.n_channels, dtype=float)
        last_total_discrete_propensity = 0.0
        last_total_continuous_propensity = 0.0
        wall_deadline_reached = False

        # Algorithm 2 outer loop.
        # Pseudocode:
        #   while t < tf:
        # Here one PDMPStepper.step(...) call advances at most dt simulation time.
        # ExperimentRunner repeatedly calls this method until global t_end.
        while float(state.t) < end_time - self.config.hazard_tol:
            if self._wall_deadline_reached(context):
                wall_deadline_reached = True
                break
            remaining = max(end_time - float(state.t), 0.0)
            if remaining <= 0.0:
                break
            step_dt = min(float(self.config.ode_step), remaining)
            partition = self._require_partition(store)
            # Use cached propensities when the state has not changed since the
            # previous calculation.  A continuous Euler update or discrete jump
            # marks the cache dirty below; repartition alone does not.
            propensities = self._get_propensities(network, state, store, reason="micro-step start")
            continuous_channels = partition.continuous_channels
            discrete_channels = partition.discrete_channels
            last_total_continuous_propensity = (
                float(np.sum(propensities[continuous_channels])) if continuous_channels.size else 0.0
            )
            last_total_discrete_propensity = (
                float(np.sum(propensities[discrete_channels])) if discrete_channels.size else 0.0
            )

            # Algorithm 2 step 1: try one continuous integration step.
            # Pseudocode:
            #   tN = min(t + dt, tf)
            #   integrate dz/dt = sum_{k in RC} lambda_k(z) xi_k
            #             dw/dt = sum_{k in RD} lambda_k(z)
            # Current implementation is a lightweight approximation:
            #   - lambda_k is frozen at the start of this micro-step;
            #   - _advance_continuous(...) applies explicit Euler for z;
            #   - _advance_hazards(...) applies explicit Euler for discrete
            #     hazard accumulation;
            #   - there is no dense output object.
            #
            # Discrete event selection is configurable:
            #   - "gillespie" uses the Direct SSA rule on RD for this frozen
            #     micro-step: tau ~ Exp(sum lambda_k), then channel sampled by
            #     lambda_k / sum lambda_k.
            #   - "nrm_scan" keeps per-channel internal hazard/threshold arrays
            #     and scans for the first threshold crossing.
            #   - "nrm_heap" keeps scheduled real firing times in a priority
            #     queue and lazily refreshes affected channels.
            tau, fired_channel = self._next_discrete_event(
                store,
                network,
                state,
                discrete_channels,
                propensities,
                rng=rng,
                current_time=float(state.t),
                horizon=step_dt,
            )
            if fired_channel is not None:
                # Algorithm 2 step 2: handle a discrete event inside the ODE step.
                # Pseudocode:
                #   find tR where w(tR) = u
                #   interpolate zR, wR
                #   sample r from lambda_k(x_pre) / sum lambda_m(x_pre)
                #   if r in RQ: break and repartition
                #   else: accumulate Delta z and continue inside the same ODE step
                #
                # Current implementation:
                #   - _next_discrete_event(...) computes tR using the selected
                #     discrete_event_method;
                #   - the jump is applied immediately to state.x after the
                #     continuous drift over event_dt;
                #   - this method returns after one event.  The runner calls
                #     step(...) again for additional events, so the inner
                #     "while wN >= u" loop is distributed across runner steps;
                #   - RQ is not separated yet.  repartition_on_event=True treats
                #     every discrete event as a possible repartition point.
                event_dt = max(float(tau), 0.0)
                continuous_abs_total += self._advance_continuous(network, state, propensities, continuous_channels, event_dt)
                self._advance_hazards(store, discrete_channels, propensities, event_dt)
                state.t += event_dt
                changed_species = self._changed_species_for_channels(network, continuous_channels) if event_dt > 0.0 else np.empty(0, dtype=np.int64)
                if self._uses_discrete_event_gillespie():
                    sampled_channel, event_total = self._sample_gillespie_channel_at_current_state(
                        network,
                        state,
                        discrete_channels,
                        rng,
                        store=store,
                        propensities=propensities,
                    )
                    if sampled_channel is None:
                        store["gillespie_disabled_event_candidates"] = int(
                            store.get("gillespie_disabled_event_candidates", 0)
                        ) + 1
                        self._mark_propensities_dirty(
                            store,
                            "gillespie event candidate disabled after continuous drift",
                            changed_species,
                        )
                        if (
                            self.config.adaptive
                            and self.config.repartition_on_bounds
                            and not self._require_partition(store).is_within_bounds(state.x)
                        ):
                            propensities = self._get_propensities(network, state, store, reason="bounds repartition")
                            partition = self._compute_partition(network, state, propensities, context)
                            self._install_partition(
                                store,
                                partition,
                                rng,
                                current_time=float(state.t),
                                propensities=propensities,
                                reset_all=False,
                            )
                            repartitions += 1
                        continue
                    fired_channel = int(sampled_channel)
                    last_total_discrete_propensity = float(event_total)
                elif not self._channel_has_available_reactants(network, state, int(fired_channel)):
                    store["disabled_discrete_event_candidates"] = int(
                        store.get("disabled_discrete_event_candidates", 0)
                    ) + 1
                    if self._uses_discrete_event_heap():
                        self._refresh_propensities_after_state_change(
                            network,
                            state,
                            store,
                            changed_species,
                            rng,
                            reason="nrm event candidate disabled after continuous drift",
                            fired_channel=int(fired_channel),
                        )
                    else:
                        self._mark_propensities_dirty(
                            store,
                            "nrm event candidate disabled after continuous drift",
                            changed_species,
                        )
                        if self._uses_discrete_event_hazards():
                            self._reset_channel_threshold(store, int(fired_channel), rng)
                    continue
                jump_changed_species = network.get_channel_changed_species(int(fired_channel))
                network.apply_channel_update(state, int(fired_channel))
                self._validate_or_clip_state(state)
                changed_species = self._merge_species_ids(changed_species, jump_changed_species)
                # The continuous drift over event_dt and the discrete jump both
                # change state.x.  In heap mode we update the cached propensity
                # vector immediately and reschedule only affected discrete
                # channels.  In scan mode we keep the older hazard/threshold
                # implementation and let the next read recompute the full vector.
                if self._uses_discrete_event_heap():
                    self._refresh_propensities_after_state_change(
                        network,
                        state,
                        store,
                        changed_species,
                        rng,
                        reason="discrete event changed state",
                        fired_channel=int(fired_channel),
                    )
                else:
                    self._mark_propensities_dirty(store, "discrete event changed state", changed_species)
                    if self._uses_discrete_event_hazards():
                        self._reset_channel_threshold(store, int(fired_channel), rng)

                if self.config.adaptive and self.config.repartition_on_event:
                    # Repartition requires propensities at the new state.  The
                    # freshly computed array remains valid after _install_partition
                    # because partition installation only changes thresholds and
                    # masks, not state.x.
                    propensities = self._get_propensities(network, state, store, reason="event repartition")
                    partition = self._compute_partition(network, state, propensities, context)
                    self._install_partition(store, partition, rng, current_time=float(state.t), propensities=propensities, reset_all=False)
                    repartitions += 1

                state.step_count += 1
                state.event_count += 1
                details = self._details(
                    store,
                    total_discrete_propensity=last_total_discrete_propensity,
                    total_continuous_propensity=last_total_continuous_propensity,
                    continuous_abs_total=continuous_abs_total,
                    repartitions=repartitions,
                )
                if details is not None:
                    details["wall_deadline_reached"] = bool(self._wall_deadline_reached(context))
                return StepResult(
                    advanced_time=float(state.t - start_time),
                    event_occurred=True,
                    channel_id=int(fired_channel),
                    propensity_sum=last_total_discrete_propensity,
                    tau=float(state.t - start_time),
                    details=details,
                )

            # Algorithm 2 step 3: accept the current ODE/hazard step.
            # Pseudocode:
            #   t = tN
            #   z = zN + Delta z
            #   w = wN
            # Here Delta z is always applied immediately when an event fires.
            # If no event fires inside this micro-step, we accept the Euler
            # continuous update and hazard accumulation over step_dt.
            continuous_abs_total += self._advance_continuous(network, state, propensities, continuous_channels, step_dt)
            self._advance_hazards(store, discrete_channels, propensities, step_dt)
            state.t += step_dt
            if continuous_channels.size:
                # The Euler drift may change continuous species by fractional
                # amounts.  Any later propensity evaluation must be based on the
                # updated state, so the cache is invalid after accepting the
                # continuous part of the step.  If RC is empty, only time changed
                # and propensities remain valid because current rates are time
                # homogeneous.
                changed_species = self._changed_species_for_channels(network, continuous_channels)
                if self._uses_discrete_event_heap():
                    self._refresh_propensities_after_state_change(
                        network,
                        state,
                        store,
                        changed_species,
                        rng,
                        reason="continuous drift changed state",
                    )
                else:
                    self._mark_propensities_dirty(store, "continuous drift changed state", changed_species)

            # Algorithm 2 step 4: check whether the partition is invalid.
            # Pseudocode:
            #   if any species violates current scale bounds:
            #       adaptation
            #       w = 0
            #       u = Exp(1)
            # Current implementation checks PDMPPartitionResult.is_within_bounds.
            # _install_partition(..., reset_all=False) preserves thresholds for
            # channels that remain discrete and initializes only newly discrete
            # channels.  This differs from the pseudocode's global w/u reset.
            if (
                self.config.adaptive
                and self.config.repartition_on_bounds
                and not self._require_partition(store).is_within_bounds(state.x)
            ):
                # Bounds violation triggers adaptation.  It does not by itself
                # alter state.x; the recomputed propensities can therefore be
                # reused by the following micro-step unless a later drift/jump
                # marks the cache dirty.
                propensities = self._get_propensities(network, state, store, reason="bounds repartition")
                partition = self._compute_partition(network, state, propensities, context)
                self._install_partition(store, partition, rng, current_time=float(state.t), propensities=propensities, reset_all=False)
                repartitions += 1

        state.step_count += 1
        details = self._details(
            store,
            total_discrete_propensity=last_total_discrete_propensity,
            total_continuous_propensity=last_total_continuous_propensity,
            continuous_abs_total=continuous_abs_total,
            repartitions=repartitions,
        )
        if details is not None:
            details["wall_deadline_reached"] = bool(wall_deadline_reached or self._wall_deadline_reached(context))
        return StepResult(
            advanced_time=float(state.t - start_time),
            event_occurred=False,
            propensity_sum=last_total_discrete_propensity,
            details=details,
        )

    def _step_gillespie_integrated_hazard(
        self,
        state: SystemState,
        duration: float,
        context: StepperContext,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        store: dict[str, Any],
        rng: np.random.Generator,
        *,
        initial_repartitions: int,
    ) -> StepResult:
        """Algorithm-2 scalar discrete-hazard loop for RD.

        This is the paper-style handling for the Direct/Gillespie discrete
        part.  The stepper maintains a scalar integrated hazard ``w`` for the
        current discrete reaction set RD and an absolute threshold ``u``.  Over
        each Euler ODE segment we test whether ``w + a_D * dt`` crosses ``u``;
        at a crossing time the concrete channel is sampled from the current RD
        propensity distribution.

        The continuous ODE solve is still the existing explicit Euler segment,
        so this is not yet a dense-output ODE solver.  The stochastic event
        clock, threshold update, and adaptation reset semantics match the
        Algorithm-2 structure.
        """

        start_time = float(state.t)
        end_time = start_time + float(duration)
        continuous_abs_total = np.zeros(network.n_channels, dtype=float)
        last_total_discrete_propensity = 0.0
        last_total_continuous_propensity = 0.0
        repartitions = int(initial_repartitions)
        rq_event_count = 0
        disabled_event_candidates = 0
        applied_event_ids: list[int] = []
        event_times: list[float] = []
        wall_deadline_reached = False

        while float(state.t) < end_time - self.config.hazard_tol:
            if self._wall_deadline_reached(context):
                wall_deadline_reached = True
                break
            micro_end = min(float(state.t) + float(self.config.ode_step), end_time)
            stop_after_event_or_adaptation = False

            while float(state.t) < micro_end - self.config.hazard_tol:
                if self._wall_deadline_reached(context):
                    wall_deadline_reached = True
                    break
                segment_dt = max(float(micro_end) - float(state.t), 0.0)
                if segment_dt <= 0.0:
                    break

                partition = self._require_partition(store)
                propensities = self._get_propensities(network, state, store, reason="gillespie hazard segment")
                continuous_channels = partition.continuous_channels
                discrete_channels = partition.discrete_channels
                last_total_continuous_propensity = (
                    float(np.sum(propensities[continuous_channels])) if continuous_channels.size else 0.0
                )
                last_total_discrete_propensity = self._available_discrete_propensity_total(
                    network,
                    state,
                    discrete_channels,
                    propensities,
                    store=store,
                )

                if last_total_discrete_propensity <= self.config.hazard_tol:
                    continuous_abs_total += self._advance_continuous(
                        network,
                        state,
                        propensities,
                        continuous_channels,
                        segment_dt,
                    )
                    state.t += segment_dt
                    if continuous_channels.size:
                        changed_species = self._changed_species_for_channels(network, continuous_channels)
                        self._refresh_propensities_after_state_change(
                            network,
                            state,
                            store,
                            changed_species,
                            rng,
                            reason="continuous drift changed state",
                        )
                    break

                hazard = self._scalar_discrete_hazard(store)
                threshold = self._scalar_discrete_threshold(store, rng)
                remaining_hazard = max(float(threshold) - float(hazard), 0.0)
                event_dt = remaining_hazard / max(float(last_total_discrete_propensity), self.config.hazard_tol)

                if not np.isfinite(event_dt) or event_dt > segment_dt + self.config.hazard_tol:
                    continuous_abs_total += self._advance_continuous(
                        network,
                        state,
                        propensities,
                        continuous_channels,
                        segment_dt,
                    )
                    self._advance_scalar_discrete_hazard(store, last_total_discrete_propensity, segment_dt)
                    state.t += segment_dt
                    if continuous_channels.size:
                        changed_species = self._changed_species_for_channels(network, continuous_channels)
                        self._refresh_propensities_after_state_change(
                            network,
                            state,
                            store,
                            changed_species,
                            rng,
                            reason="continuous drift changed state",
                        )
                    break

                event_dt = max(float(event_dt), 0.0)
                continuous_abs_total += self._advance_continuous(
                    network,
                    state,
                    propensities,
                    continuous_channels,
                    event_dt,
                )
                state.t += event_dt
                store["discrete_total_hazard"] = float(threshold)
                changed_species = (
                    self._changed_species_for_channels(network, continuous_channels)
                    if continuous_channels.size and event_dt > 0.0
                    else np.empty(0, dtype=np.int64)
                )
                if changed_species.size:
                    self._refresh_propensities_after_state_change(
                        network,
                        state,
                        store,
                        changed_species,
                        rng,
                        reason="continuous drift before gillespie event",
                    )

                partition = self._require_partition(store)
                sampled_channel, event_total = self._sample_gillespie_channel_at_current_state(
                    network,
                    state,
                    partition.discrete_channels,
                    rng,
                    store=store,
                )
                last_total_discrete_propensity = float(event_total)
                self._advance_scalar_discrete_threshold(store, rng)

                if sampled_channel is None:
                    disabled_event_candidates += 1
                    store["gillespie_disabled_event_candidates"] = int(
                        store.get("gillespie_disabled_event_candidates", 0)
                    ) + 1
                    self._mark_propensities_dirty(
                        store,
                        "gillespie hazard crossing had no available event",
                        changed_species,
                    )
                    continue

                fired_channel = int(sampled_channel)
                jump_changed_species = network.get_channel_changed_species(fired_channel)
                network.apply_channel_update(state, fired_channel)
                self._validate_or_clip_state(state)
                changed_species = self._merge_species_ids(changed_species, jump_changed_species)
                applied_event_ids.append(fired_channel)
                event_times.append(float(state.t))

                self._refresh_propensities_after_state_change(
                    network,
                    state,
                    store,
                    changed_species,
                    rng,
                    reason="gillespie discrete event changed state",
                    fired_channel=fired_channel,
                )

                if self._wall_deadline_reached(context):
                    wall_deadline_reached = True
                    stop_after_event_or_adaptation = True
                    break

                force_adaptation = self._channel_is_rq(store, fired_channel)
                if force_adaptation:
                    rq_event_count += 1

                if (
                    force_adaptation
                    or (self.config.adaptive and self.config.repartition_on_event)
                    or (
                        self.config.adaptive
                        and self.config.repartition_on_bounds
                        and not self._require_partition(store).is_within_bounds(state.x)
                    )
                ):
                    propensities = self._get_propensities(network, state, store, reason="gillespie adaptation")
                    partition = self._compute_partition(network, state, propensities, context)
                    self._install_partition(
                        store,
                        partition,
                        rng,
                        current_time=float(state.t),
                        propensities=propensities,
                        reset_all=True,
                    )
                    repartitions += 1
                    stop_after_event_or_adaptation = True
                    break

            if stop_after_event_or_adaptation:
                break

            if (
                self.config.adaptive
                and self.config.repartition_on_bounds
                and not self._require_partition(store).is_within_bounds(state.x)
            ):
                propensities = self._get_propensities(network, state, store, reason="bounds repartition")
                partition = self._compute_partition(network, state, propensities, context)
                self._install_partition(
                    store,
                    partition,
                    rng,
                    current_time=float(state.t),
                    propensities=propensities,
                    reset_all=True,
                )
                repartitions += 1
                break

        state.step_count += 1
        state.event_count += len(applied_event_ids)
        details = self._details(
            store,
            total_discrete_propensity=last_total_discrete_propensity,
            total_continuous_propensity=last_total_continuous_propensity,
            continuous_abs_total=continuous_abs_total,
            repartitions=repartitions,
        )
        if details is not None:
            details["discrete_hazard_mode"] = "scalar_integrated"
            details["discrete_event_ids"] = np.asarray(applied_event_ids, dtype=np.int64)
            details["discrete_event_times"] = np.asarray(event_times, dtype=float)
            details["rq_event_count"] = int(rq_event_count)
            details["disabled_discrete_event_candidates"] = int(disabled_event_candidates)
            details["discrete_total_hazard"] = float(store.get("discrete_total_hazard", 0.0))
            details["discrete_hazard_threshold"] = float(store.get("discrete_hazard_threshold", np.inf))
            details["wall_deadline_reached"] = bool(wall_deadline_reached or self._wall_deadline_reached(context))
        first_tau = None if not event_times else float(event_times[0] - start_time)
        return StepResult(
            advanced_time=float(state.t - start_time),
            event_occurred=bool(applied_event_ids),
            channel_id=None if not applied_event_ids else int(applied_event_ids[-1]),
            propensity_sum=max(float(last_total_discrete_propensity), 0.0),
            tau=first_tau,
            details=details,
        )

    def _pdmp_network(self, network: ReactionNetworkData | ElementaryMassActionNetwork) -> ReactionNetworkData | ElementaryMassActionNetwork:
        if not isinstance(network, (ReactionNetworkData, ElementaryMassActionNetwork)):
            raise TypeError(
                "PDMPStepper requires ReactionNetworkData or ElementaryMassActionNetwork. "
                "Use partition_method='linear_catalysis_scaling' for direct effective-catalysis polymer networks, "
                "or build an ElementaryMassActionNetwork first for the default paper-style scaling partition."
            )
        return network

    def _wall_deadline_reached(self, context: StepperContext) -> bool:
        """Return True when the runner's wall-clock budget has expired.

        The runner normally checks max_runtime_seconds between stepper calls.
        PDMP can process many discrete hazard crossings inside one macro step,
        so it also observes the same absolute deadline at safe loop points.
        """

        deadline = context.wall_deadline
        return deadline is not None and perf_counter() >= float(deadline)

    def _ensure_runtime_store(self, state: SystemState, network: ReactionNetworkData | ElementaryMassActionNetwork) -> dict[str, Any]:
        if not isinstance(state.partition_state, dict):
            state.partition_state = {}
        store = state.partition_state.get("pdmp")
        if not isinstance(store, dict):
            store = {}
            state.partition_state["pdmp"] = store

        invalid = (
            store.get("network_id") != id(network)
            or store.get("n_channels") != network.n_channels
            or store.get("n_species") != network.n_species
        )
        if invalid:
            store.clear()
            store["network_id"] = id(network)
            store["n_channels"] = int(network.n_channels)
            store["n_species"] = int(network.n_species)
            store["thresholds"] = np.full(network.n_channels, np.inf, dtype=float)
            store["hazards"] = np.zeros(network.n_channels, dtype=float)
            store["discrete_mask"] = np.zeros(network.n_channels, dtype=bool)
            store["partition"] = None
            store["propensities"] = np.empty(network.n_channels, dtype=float)
            store["propensities_valid"] = False
            store["propensities_dirty_reason"] = "new runtime store"
            store["propensities_dirty_species"] = np.empty(0, dtype=np.int64)
            store["propensity_update_mode"] = "full"
            store["propensity_update_channels"] = int(network.n_channels)
            store["scheduled_times"] = np.full(network.n_channels, np.inf, dtype=float)
            store["heap_versions"] = np.zeros(network.n_channels, dtype=np.int64)
            store["event_heap"] = []
            store["heap_stale_pops"] = 0
            store["heap_rebuilds"] = 0
            store["rq_mask"] = np.zeros(network.n_channels, dtype=bool)
            store["discrete_total_hazard"] = 0.0
            store["discrete_hazard_threshold"] = np.inf
            # Gillespie-in-PDMP 热路径缓存：
            # - gillespie_available_propensities[c] 保存当前 RD 中、且反应物
            #   可用的通道 propensity；其它通道为 0。
            # - gillespie_discrete_total 是上面数组在 RD 上的和。
            # 状态或 partition 改变后只刷新受影响通道，避免每个 hazard
            # segment 扫描全部离散通道。
            store["gillespie_available_mask"] = np.zeros(network.n_channels, dtype=bool)
            store["gillespie_available_propensities"] = np.zeros(network.n_channels, dtype=float)
            store["gillespie_discrete_total"] = 0.0
            store["gillespie_cache_valid"] = False
            # scratch buffers：用于局部 propensity 更新和可用性过滤。
            # 这些数组属于当前 state/network 的运行缓存，不参与模型语义。
            store["affected_channel_marker"] = np.zeros(network.n_channels, dtype=bool)
            store["affected_channel_scratch"] = np.empty(network.n_channels, dtype=np.int64)
            store["propensity_subset_scratch"] = np.empty(network.n_channels, dtype=float)
            store["available_subset_scratch"] = np.empty(network.n_channels, dtype=bool)
            store["numba_kernel_used"] = False
            store["numba_local_refreshes"] = 0
            store["numba_cache_rebuilds"] = 0
            store["numba_affected_collects"] = 0
        return store

    def _get_propensities(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        store: dict[str, Any],
        *,
        reason: str,
    ) -> np.ndarray:
        """Return the current propensity vector, reusing it only when valid.

        Cache contract:
        - valid means the array was computed for the current ``state.x``;
        - continuous Euler drift and discrete jumps call
          ``_mark_propensities_dirty`` immediately after mutating ``state.x``;
        - partition installation does not mutate ``state.x``, so a vector
          computed for repartition can be reused by the next micro-step;
        - external callers can force a rebuild through ``invalidate_cache``.
        """

        cached = store.get("propensities")
        if (
            bool(store.get("propensities_valid", False))
            and isinstance(cached, np.ndarray)
            and cached.shape == (network.n_channels,)
        ):
            return cached

        if not isinstance(cached, np.ndarray) or cached.shape != (network.n_channels,):
            cached = np.empty(network.n_channels, dtype=float)
            store["propensities"] = cached
        network.compute_all_propensities(state, out=cached)
        self._clean_propensities(cached)
        store["propensities_valid"] = True
        store["propensities_dirty_reason"] = None
        store["propensities_dirty_species"] = np.empty(0, dtype=np.int64)
        store["propensities_last_compute_reason"] = str(reason)
        store["propensity_update_mode"] = "full"
        store["propensity_update_channels"] = int(network.n_channels)
        return cached

    def _mark_propensities_dirty(
        self,
        store: dict[str, Any],
        reason: str,
        changed_species: np.ndarray | None = None,
    ) -> None:
        """Mark cached propensities invalid after a state-changing operation."""

        store["propensities_valid"] = False
        store["propensities_dirty_reason"] = str(reason)
        self._invalidate_gillespie_discrete_cache(store)
        if changed_species is not None:
            previous = np.asarray(store.get("propensities_dirty_species", np.empty(0, dtype=np.int64)), dtype=np.int64)
            store["propensities_dirty_species"] = self._merge_species_ids(previous, changed_species)

    def _invalidate_gillespie_discrete_cache(self, store: dict[str, Any]) -> None:
        """Invalidate cached RD propensity totals for the Gillespie discrete path."""

        store["gillespie_cache_valid"] = False
        store["gillespie_discrete_total"] = 0.0

    def _ensure_runtime_array(
        self,
        store: dict[str, Any],
        name: str,
        *,
        size: int,
        dtype: type,
        fill_value: float | int | bool | None = None,
    ) -> np.ndarray:
        values = store.get(name)
        if not isinstance(values, np.ndarray) or values.shape != (int(size),) or values.dtype != np.dtype(dtype):
            if fill_value is None:
                values = np.empty(int(size), dtype=dtype)
            else:
                values = np.full(int(size), fill_value, dtype=dtype)
            store[name] = values
        return values

    def _numba_backend_enabled(self) -> bool:
        backend = str(getattr(self.config, "kernel_backend", "auto")).lower()
        if backend == "python":
            return False
        if backend == "numba":
            return True
        return bool(NUMBA_AVAILABLE)

    def _has_dependency_csr(self, network: ReactionNetworkData | ElementaryMassActionNetwork) -> bool:
        indptr = getattr(network, "species_to_channels_indptr", None)
        indices = getattr(network, "species_to_channels_indices", None)
        return (
            isinstance(indptr, np.ndarray)
            and isinstance(indices, np.ndarray)
            and indptr.shape == (network.n_species + 1,)
            and indices.ndim == 1
        )

    def _elementary_numba_kernel_supported(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
    ) -> bool:
        return (
            self._numba_backend_enabled()
            and isinstance(network, ElementaryMassActionNetwork)
            and self._has_dependency_csr(network)
            and self._has_precomputed_reactant_terms(network)
            and isinstance(getattr(network, "rate_constants", None), np.ndarray)
        )

    def _compute_partition(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        propensities: np.ndarray,
        context: StepperContext,
    ) -> PDMPPartitionResult:
        # Algorithm 2 adaptation hook.
        # This method is where the stepper delegates partition construction.
        # With ScalingPDMPPartitionStrategy it executes the Algorithm-3-style
        # scale analysis; with FixedPDMPPartitionStrategy it uses a manual split.
        strategy = context.partition_strategy or self.partition_strategy
        if strategy is None:
            strategy = _pdmp_partition_strategy_from_method(self.partition_method, self.partition_config)
        result = strategy.partition(network, state, propensities)
        if isinstance(result, PDMPPartitionResult):
            self._validate_partition(network, result)
            return result
        if hasattr(result, "fast_channels") and hasattr(result, "slow_channels"):
            converted = FixedPDMPPartitionStrategy(
                continuous_channels=np.asarray(result.fast_channels, dtype=np.int64),
            ).partition(network, state, propensities)
            self._validate_partition(network, converted)
            return converted
        raise TypeError("PDMP partition_strategy must return PDMPPartitionResult or a fast/slow PartitionResult")

    def _install_partition(
        self,
        store: dict[str, Any],
        partition: PDMPPartitionResult,
        rng: np.random.Generator,
        *,
        current_time: float,
        propensities: np.ndarray,
        reset_all: bool,
    ) -> None:
        thresholds = np.asarray(store["thresholds"], dtype=float)
        hazards = np.asarray(store["hazards"], dtype=float)
        old_mask = np.asarray(store.get("discrete_mask"), dtype=bool)
        new_mask = np.zeros_like(old_mask, dtype=bool)
        new_mask[partition.discrete_channels] = True

        if reset_all:
            initialize_mask = new_mask
        else:
            initialize_mask = new_mask & ~old_mask
        removed_mask = old_mask & ~new_mask
        if np.any(removed_mask):
            thresholds[removed_mask] = np.inf
            hazards[removed_mask] = 0.0
        if np.any(initialize_mask):
            thresholds[initialize_mask] = rng.exponential(1.0, size=int(np.sum(initialize_mask)))
            hazards[initialize_mask] = 0.0

        store["partition"] = partition
        store["discrete_mask"] = new_mask
        rq_mask = np.zeros_like(new_mask, dtype=bool)
        rq_channels = np.asarray(getattr(partition, "rq_channels", np.empty(0, dtype=np.int64)), dtype=np.int64)
        if rq_channels.size:
            rq_mask[rq_channels] = True
        store["rq_mask"] = rq_mask
        if self._uses_discrete_event_gillespie():
            self._invalidate_gillespie_discrete_cache(store)
            self._reset_scalar_discrete_hazard(store, rng)
        if self._uses_discrete_event_heap():
            self._install_discrete_event_heap_mask(
                store,
                current_time=float(current_time),
                propensities=propensities,
                rng=rng,
                old_mask=old_mask,
                new_mask=new_mask,
                reset_all=reset_all,
            )

    def _install_discrete_event_heap_mask(
        self,
        store: dict[str, Any],
        *,
        current_time: float,
        propensities: np.ndarray,
        rng: np.random.Generator,
        old_mask: np.ndarray,
        new_mask: np.ndarray,
        reset_all: bool,
    ) -> None:
        scheduled = self._ensure_heap_array(store, "scheduled_times", fill=np.inf)
        versions = self._ensure_heap_versions(store)
        heap = self._ensure_event_heap(store)

        if reset_all:
            scheduled[:] = np.inf
            versions[:] = 0
            heap.clear()
            initialize_mask = new_mask
        else:
            removed_mask = old_mask & ~new_mask
            if np.any(removed_mask):
                removed = np.flatnonzero(removed_mask).astype(np.int64, copy=False)
                scheduled[removed] = np.inf
                versions[removed] += 1
            initialize_mask = new_mask & ~old_mask

        initialize_channels = np.flatnonzero(initialize_mask).astype(np.int64, copy=False)
        for channel_id in initialize_channels:
            self._schedule_discrete_heap_channel(
                store,
                int(channel_id),
                now=float(current_time),
                propensity=float(propensities[int(channel_id)]),
                rng=rng,
                fresh=True,
            )
        self._maybe_rebuild_discrete_event_heap(store)

    def _ensure_heap_array(self, store: dict[str, Any], name: str, *, fill: float) -> np.ndarray:
        values = store.get(name)
        n_channels = int(store["n_channels"])
        if not isinstance(values, np.ndarray) or values.shape != (n_channels,):
            values = np.full(n_channels, float(fill), dtype=float)
            store[name] = values
        return values

    def _ensure_heap_versions(self, store: dict[str, Any]) -> np.ndarray:
        versions = store.get("heap_versions")
        n_channels = int(store["n_channels"])
        if not isinstance(versions, np.ndarray) or versions.shape != (n_channels,):
            versions = np.zeros(n_channels, dtype=np.int64)
            store["heap_versions"] = versions
        return versions

    def _ensure_event_heap(self, store: dict[str, Any]) -> list[tuple[float, int, int]]:
        heap = store.get("event_heap")
        if not isinstance(heap, list):
            heap = []
            store["event_heap"] = heap
        return heap

    def _schedule_discrete_heap_channel(
        self,
        store: dict[str, Any],
        channel_id: int,
        *,
        now: float,
        propensity: float,
        rng: np.random.Generator,
        fresh: bool,
    ) -> None:
        scheduled = self._ensure_heap_array(store, "scheduled_times", fill=np.inf)
        versions = self._ensure_heap_versions(store)
        heap = self._ensure_event_heap(store)
        cid = int(channel_id)
        if fresh:
            versions[cid] += 1
        propensity_value = max(float(propensity), 0.0)
        if propensity_value <= self.config.hazard_tol:
            scheduled[cid] = np.inf
            return
        fire_time = float(now) + float(rng.exponential(1.0 / propensity_value))
        scheduled[cid] = fire_time
        heapq.heappush(heap, (fire_time, cid, int(versions[cid])))

    def _rebuild_discrete_event_heap(self, store: dict[str, Any]) -> None:
        scheduled = self._ensure_heap_array(store, "scheduled_times", fill=np.inf)
        versions = self._ensure_heap_versions(store)
        mask = np.asarray(store.get("discrete_mask"), dtype=bool)
        channels = np.flatnonzero(mask & np.isfinite(scheduled)).astype(np.int64, copy=False)
        heap = [(float(scheduled[int(cid)]), int(cid), int(versions[int(cid)])) for cid in channels]
        heapq.heapify(heap)
        store["event_heap"] = heap
        store["heap_rebuilds"] = int(store.get("heap_rebuilds", 0)) + 1

    def _maybe_rebuild_discrete_event_heap(self, store: dict[str, Any]) -> None:
        heap = self._ensure_event_heap(store)
        scheduled = self._ensure_heap_array(store, "scheduled_times", fill=np.inf)
        live = int(np.sum(np.isfinite(scheduled) & np.asarray(store.get("discrete_mask"), dtype=bool)))
        limit = max(int(np.ceil(self.config.heap_rebuild_factor * max(live, 1))), live + 1)
        if len(heap) > limit:
            self._rebuild_discrete_event_heap(store)

    def _next_discrete_event(
        self,
        store: dict[str, Any],
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        discrete_channels: np.ndarray,
        propensities: np.ndarray,
        *,
        rng: np.random.Generator,
        current_time: float,
        horizon: float,
    ) -> tuple[float | None, int | None]:
        if self._uses_discrete_event_heap():
            return self._next_discrete_heap_event(store, current_time=float(current_time), horizon=float(horizon))
        if self._uses_discrete_event_gillespie():
            return self._next_discrete_gillespie_event(
                network,
                state,
                discrete_channels,
                propensities,
                rng=rng,
                current_time=float(current_time),
                horizon=float(horizon),
            )

        # Discrete-event locator for the current micro-step.
        # Pseudocode uses a scalar condition wN >= u and then solves w(tR) = u.
        # Here each discrete channel has its own threshold and hazard.  Under
        # frozen propensities for this micro-step, the next firing time is
        # (threshold_r - hazard_r) / propensity_r, and the smallest such time is
        # the event inside the horizon.
        channels = np.asarray(discrete_channels, dtype=np.int64)
        if channels.size == 0:
            return None, None
        props = np.maximum(propensities[channels], 0.0)
        active = props > self.config.hazard_tol
        if not np.any(active):
            return None, None

        thresholds = np.asarray(store["thresholds"], dtype=float)
        hazards = np.asarray(store["hazards"], dtype=float)
        remaining_hazard = np.maximum(thresholds[channels] - hazards[channels], 0.0)
        taus = np.full(channels.shape, np.inf, dtype=float)
        taus[active] = remaining_hazard[active] / props[active]
        local = int(np.argmin(taus))
        tau = float(taus[local])
        if not np.isfinite(tau) or tau > float(horizon) + self.config.hazard_tol:
            return None, None
        return max(tau, 0.0), int(channels[local])

    def _next_discrete_gillespie_event(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        discrete_channels: np.ndarray,
        propensities: np.ndarray,
        *,
        rng: np.random.Generator,
        current_time: float,
        horizon: float,
    ) -> tuple[float | None, int | None]:
        channels = np.asarray(discrete_channels, dtype=np.int64)
        if channels.size == 0:
            return None, None
        props = np.maximum(np.asarray(propensities, dtype=float)[channels], 0.0)
        self._zero_unavailable_gillespie_props(network, state, channels, props)
        total = float(np.sum(props))
        if total <= self.config.hazard_tol:
            return None, None
        tau = float(rng.exponential(1.0 / total))
        if not np.isfinite(tau) or tau > float(horizon) + self.config.hazard_tol:
            return None, None
        # Only the event time is decided here.  The actual channel is sampled
        # after continuous drift has advanced to the event state, using current
        # RD propensities and an explicit reactant-availability check.
        return max(tau, 0.0), -1

    def _zero_unavailable_gillespie_props(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        channels: np.ndarray,
        props: np.ndarray,
    ) -> None:
        channels_array = np.asarray(channels, dtype=np.int64)
        if channels_array.size == 0:
            return
        available = self._available_channel_mask(network, state, channels_array)
        props[~available] = 0.0

    def _available_discrete_propensity_total(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        discrete_channels: np.ndarray,
        propensities: np.ndarray,
        *,
        store: dict[str, Any] | None = None,
    ) -> float:
        if store is not None and self._uses_discrete_event_gillespie():
            self._ensure_gillespie_discrete_cache(network, state, store, propensities)
            return max(float(store.get("gillespie_discrete_total", 0.0)), 0.0)

        channels = np.asarray(discrete_channels, dtype=np.int64)
        if channels.size == 0:
            return 0.0
        props = np.maximum(np.asarray(propensities, dtype=float)[channels], 0.0).copy()
        self._zero_unavailable_gillespie_props(network, state, channels, props)
        return max(float(np.sum(props)), 0.0)

    def _sample_gillespie_channel_at_current_state(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        discrete_channels: np.ndarray,
        rng: np.random.Generator,
        *,
        store: dict[str, Any] | None = None,
        propensities: np.ndarray | None = None,
    ) -> tuple[int | None, float]:
        channels = np.asarray(discrete_channels, dtype=np.int64)
        if channels.size == 0:
            return None, 0.0

        if store is not None:
            cached_props = propensities
            if cached_props is None:
                cached_props = self._get_propensities(
                    network,
                    state,
                    store,
                    reason="gillespie channel sample",
                )
            self._ensure_gillespie_discrete_cache(network, state, store, cached_props)
            props = np.asarray(store["gillespie_available_propensities"], dtype=float)[channels]
            total = float(store.get("gillespie_discrete_total", 0.0))
        else:
            props = network.compute_propensities_for_channels(channels, state)
            self._clean_propensities(props)
            self._zero_unavailable_gillespie_props(network, state, channels, props)
            total = float(np.sum(props))
        if total <= self.config.hazard_tol:
            return None, 0.0
        threshold = float(rng.random() * total)
        cumulative = np.cumsum(props)
        local = int(np.searchsorted(cumulative, threshold, side="right"))
        if local >= channels.size:
            local = int(channels.size - 1)
        return int(channels[local]), total

    def _ensure_gillespie_discrete_cache(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        store: dict[str, Any],
        propensities: np.ndarray,
    ) -> None:
        """Build the cached available RD propensity vector when invalid.

        P0 优化点：Algorithm-2 的离散危险度每个 segment 都需要
        ``sum_{r in RD} lambda_r``。旧实现每次都复制 RD propensity 并检查
        反应物是否可用；这里在缓存有效时直接返回标量总和。缓存失效发生在
        状态改变、partition 改变或 propensity 全量失效时。
        """

        available_props = self._ensure_runtime_array(
            store,
            "gillespie_available_propensities",
            size=network.n_channels,
            dtype=float,
            fill_value=0.0,
        )
        available_mask = self._ensure_runtime_array(
            store,
            "gillespie_available_mask",
            size=network.n_channels,
            dtype=bool,
            fill_value=False,
        )
        if bool(store.get("gillespie_cache_valid", False)):
            return

        available_props[:] = 0.0
        available_mask[:] = False
        discrete_mask = np.asarray(store.get("discrete_mask"), dtype=bool)
        if discrete_mask.shape != (network.n_channels,):
            store["gillespie_discrete_total"] = 0.0
            store["gillespie_cache_valid"] = True
            return
        if self._try_numba_elementary_gillespie_cache_rebuild(
            network,
            state,
            store,
            propensities,
            discrete_mask,
            available_props,
            available_mask,
        ):
            return
        channels = np.flatnonzero(discrete_mask).astype(np.int64, copy=False)
        if channels.size == 0:
            store["gillespie_discrete_total"] = 0.0
            store["gillespie_cache_valid"] = True
            return

        scratch_props = self._ensure_runtime_array(
            store,
            "propensity_subset_scratch",
            size=network.n_channels,
            dtype=float,
        )[: channels.size]
        scratch_available = self._ensure_runtime_array(
            store,
            "available_subset_scratch",
            size=network.n_channels,
            dtype=bool,
        )[: channels.size]
        scratch_props[:] = np.asarray(propensities, dtype=float)[channels]
        np.maximum(scratch_props, 0.0, out=scratch_props)
        self._fill_available_channel_mask(network, state, channels, scratch_available)
        scratch_props[~scratch_available] = 0.0

        available_props[channels] = scratch_props
        active = scratch_props > self.config.hazard_tol
        if np.any(active):
            available_mask[channels[active]] = True
        store["gillespie_discrete_total"] = max(float(np.sum(scratch_props)), 0.0)
        store["gillespie_cache_valid"] = True

    def _rebuild_gillespie_discrete_cache_after_full_compute(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        store: dict[str, Any],
        propensities: np.ndarray,
    ) -> None:
        """Rebuild Gillespie RD cache after a known-current full propensity pass."""

        if not self._uses_discrete_event_gillespie():
            return
        self._invalidate_gillespie_discrete_cache(store)
        self._ensure_gillespie_discrete_cache(network, state, store, propensities)

    def _try_numba_elementary_gillespie_cache_rebuild(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        store: dict[str, Any],
        propensities: np.ndarray,
        discrete_mask: np.ndarray,
        available_props: np.ndarray,
        available_mask: np.ndarray,
    ) -> bool:
        if not self._elementary_numba_kernel_supported(network):
            return False
        total, bad_count = rebuild_elementary_gillespie_cache_numba(
            np.asarray(state.x, dtype=float),
            np.asarray(network.rate_constants, dtype=float),
            np.asarray(network.reaction_order, dtype=np.int8),
            np.asarray(network.reactant1, dtype=np.int64),
            np.asarray(network.reactant2, dtype=np.int64),
            np.asarray(network.homo_second_order, dtype=bool),
            np.asarray(discrete_mask, dtype=bool),
            np.asarray(propensities, dtype=float),
            available_props,
            available_mask,
            float(self.config.hazard_tol),
        )
        if int(bad_count) > 0:
            raise ValueError("PDMP numba Gillespie cache rebuild produced NaN or inf propensities")
        store["gillespie_discrete_total"] = max(float(total), 0.0)
        store["gillespie_cache_valid"] = True
        store["numba_kernel_used"] = True
        store["numba_cache_rebuilds"] = int(store.get("numba_cache_rebuilds", 0)) + 1
        return True

    def _update_gillespie_discrete_cache_for_channels(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        store: dict[str, Any],
        channels: np.ndarray,
        propensities: np.ndarray,
    ) -> None:
        """Refresh cached Gillespie weights for changed channels only.

        该函数在 ``_refresh_propensities_after_state_change`` 完成局部
        propensity 更新后调用。若缓存尚未建立，保持 invalid，由下一次
        hazard total 查询一次性重建。
        """

        if not self._uses_discrete_event_gillespie() or not bool(store.get("gillespie_cache_valid", False)):
            return
        channel_array = np.asarray(channels, dtype=np.int64)
        if channel_array.size == 0:
            return
        discrete_mask = np.asarray(store.get("discrete_mask"), dtype=bool)
        if discrete_mask.shape != (network.n_channels,):
            self._invalidate_gillespie_discrete_cache(store)
            return
        valid = (channel_array >= 0) & (channel_array < network.n_channels)
        if not np.any(valid):
            return
        valid_indices = np.flatnonzero(valid)
        discrete_indices = valid_indices[discrete_mask[channel_array[valid_indices]]]
        if discrete_indices.size == 0:
            return
        cids = channel_array[discrete_indices]

        available_props = self._ensure_runtime_array(
            store,
            "gillespie_available_propensities",
            size=network.n_channels,
            dtype=float,
            fill_value=0.0,
        )
        available_mask = self._ensure_runtime_array(
            store,
            "gillespie_available_mask",
            size=network.n_channels,
            dtype=bool,
            fill_value=False,
        )
        total = max(float(store.get("gillespie_discrete_total", 0.0)), 0.0)
        total -= float(np.sum(available_props[cids]))

        scratch_props = self._ensure_runtime_array(
            store,
            "propensity_subset_scratch",
            size=network.n_channels,
            dtype=float,
        )[: cids.size]
        scratch_available = self._ensure_runtime_array(
            store,
            "available_subset_scratch",
            size=network.n_channels,
            dtype=bool,
        )[: cids.size]
        scratch_props[:] = np.asarray(propensities, dtype=float)[cids]
        np.maximum(scratch_props, 0.0, out=scratch_props)
        self._fill_available_channel_mask(network, state, cids, scratch_available)
        scratch_props[~scratch_available] = 0.0

        available_props[cids] = scratch_props
        available_mask[cids] = scratch_props > self.config.hazard_tol
        total += float(np.sum(scratch_props))
        store["gillespie_discrete_total"] = max(float(total), 0.0)

    def _try_numba_elementary_gillespie_local_refresh(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        store: dict[str, Any],
        affected_channels: np.ndarray,
        cached_propensities: np.ndarray,
    ) -> bool:
        """Run the compiled elementary local refresh when it is safe to do so."""

        if (
            not self._uses_discrete_event_gillespie()
            or not bool(store.get("gillespie_cache_valid", False))
            or not self._elementary_numba_kernel_supported(network)
        ):
            return False
        affected = np.asarray(affected_channels, dtype=np.int64)
        if affected.size == 0:
            return True
        discrete_mask = np.asarray(store.get("discrete_mask"), dtype=bool)
        if discrete_mask.shape != (network.n_channels,):
            self._invalidate_gillespie_discrete_cache(store)
            return False
        available_props = self._ensure_runtime_array(
            store,
            "gillespie_available_propensities",
            size=network.n_channels,
            dtype=float,
            fill_value=0.0,
        )
        available_mask = self._ensure_runtime_array(
            store,
            "gillespie_available_mask",
            size=network.n_channels,
            dtype=bool,
            fill_value=False,
        )
        total, bad_count = refresh_elementary_gillespie_cache_numba(
            affected,
            np.asarray(state.x, dtype=float),
            np.asarray(network.rate_constants, dtype=float),
            np.asarray(network.reaction_order, dtype=np.int8),
            np.asarray(network.reactant1, dtype=np.int64),
            np.asarray(network.reactant2, dtype=np.int64),
            np.asarray(network.homo_second_order, dtype=bool),
            discrete_mask,
            cached_propensities,
            available_props,
            available_mask,
            float(store.get("gillespie_discrete_total", 0.0)),
            float(self.config.hazard_tol),
        )
        if int(bad_count) > 0:
            raise ValueError("PDMP numba local propensity refresh produced NaN or inf propensities")
        store["gillespie_discrete_total"] = max(float(total), 0.0)
        store["gillespie_cache_valid"] = True
        store["numba_kernel_used"] = True
        store["numba_local_refreshes"] = int(store.get("numba_local_refreshes", 0)) + 1
        return True

    def _channel_has_available_reactants(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        channel_id: int,
    ) -> bool:
        if self._has_precomputed_reactant_terms(network):
            return bool(self._available_channel_mask(network, state, np.asarray([int(channel_id)], dtype=np.int64))[0])
        reactants = network.get_channel_reactants(int(channel_id))
        if not reactants:
            return True
        required: dict[int, int] = {}
        for sid in reactants:
            key = int(sid)
            required[key] = required.get(key, 0) + 1
        x = np.asarray(state.x, dtype=float)
        tol = max(float(self.config.hazard_tol), 1e-12)
        return all(float(x[sid]) >= float(count) - tol for sid, count in required.items())

    def _available_channel_mask(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        channels: np.ndarray,
    ) -> np.ndarray:
        ids = np.asarray(channels, dtype=np.int64)
        available = np.ones(ids.shape, dtype=bool)
        return self._fill_available_channel_mask(network, state, ids, available)

    def _fill_available_channel_mask(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        channels: np.ndarray,
        out: np.ndarray,
    ) -> np.ndarray:
        ids = np.asarray(channels, dtype=np.int64)
        available = out
        if available.shape != ids.shape:
            raise ValueError(f"out must have shape {ids.shape}")
        available[...] = True
        if ids.size == 0:
            return available
        if not self._has_precomputed_reactant_terms(network):
            for position, channel_id in enumerate(ids):
                available[position] = self._channel_has_available_reactants(network, state, int(channel_id))
            return available

        order = np.asarray(getattr(network, "reaction_order"), dtype=np.int8)[ids]
        reactant1 = np.asarray(getattr(network, "reactant1"), dtype=np.int64)[ids]
        reactant2 = np.asarray(getattr(network, "reactant2"), dtype=np.int64)[ids]
        homo_second_order = np.asarray(getattr(network, "homo_second_order"), dtype=bool)[ids]
        x = np.asarray(state.x, dtype=float)
        tol = max(float(self.config.hazard_tol), 1e-12)

        first = order == 1
        if np.any(first):
            available[first] = x[reactant1[first]] >= 1.0 - tol

        second = order == 2
        if np.any(second):
            homo = second & homo_second_order
            if np.any(homo):
                available[homo] = x[reactant1[homo]] >= 2.0 - tol
            hetero = second & ~homo_second_order
            if np.any(hetero):
                available[hetero] = (
                    (x[reactant1[hetero]] >= 1.0 - tol)
                    & (x[reactant2[hetero]] >= 1.0 - tol)
                )

        higher_order = order > 2
        if np.any(higher_order):
            for position in np.flatnonzero(higher_order):
                reactants = network.get_channel_reactants(int(ids[int(position)]))
                required: dict[int, int] = {}
                for sid in reactants:
                    key = int(sid)
                    required[key] = required.get(key, 0) + 1
                available[int(position)] = all(
                    float(x[sid]) >= float(count) - tol
                    for sid, count in required.items()
                )
        return available

    def _has_precomputed_reactant_terms(self, network: ReactionNetworkData | ElementaryMassActionNetwork) -> bool:
        return all(
            hasattr(network, name)
            for name in ("reaction_order", "reactant1", "reactant2", "homo_second_order")
        )

    def _next_discrete_heap_event(
        self,
        store: dict[str, Any],
        *,
        current_time: float,
        horizon: float,
    ) -> tuple[float | None, int | None]:
        heap = self._ensure_event_heap(store)
        scheduled = self._ensure_heap_array(store, "scheduled_times", fill=np.inf)
        versions = self._ensure_heap_versions(store)
        discrete_mask = np.asarray(store.get("discrete_mask"), dtype=bool)
        deadline = float(current_time) + float(horizon) + self.config.hazard_tol

        while heap:
            fire_time, channel_id, version = heap[0]
            cid = int(channel_id)
            valid = (
                0 <= cid < scheduled.size
                and bool(discrete_mask[cid])
                and int(version) == int(versions[cid])
                and np.isfinite(fire_time)
                and float(fire_time) == float(scheduled[cid])
            )
            if not valid:
                heapq.heappop(heap)
                store["heap_stale_pops"] = int(store.get("heap_stale_pops", 0)) + 1
                continue
            if float(fire_time) > deadline:
                return None, None
            heapq.heappop(heap)
            scheduled[cid] = np.inf
            return max(float(fire_time) - float(current_time), 0.0), cid
        return None, None

    def _advance_continuous(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        propensities: np.ndarray,
        continuous_channels: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        # Continuous part of Algorithm 2 step 1.
        # This is explicit Euler for dz/dt = sum_{RC} lambda_k(z) xi_k using
        # propensities frozen at the start of the micro-step.  It is not yet a
        # dense-output ODE solve.
        continuous_abs = np.zeros(network.n_channels, dtype=float)
        if dt <= 0.0 or continuous_channels.size == 0:
            return continuous_abs
        channels = np.asarray(continuous_channels, dtype=np.int64)
        reaction_amounts = np.maximum(propensities[channels], 0.0) * float(dt)
        if reaction_amounts.size:
            state.x[:] = np.asarray(state.x, dtype=float) + reaction_amounts @ network.nu[channels]
            continuous_abs[channels] = np.abs(reaction_amounts)
        self._validate_or_clip_state(state)
        return continuous_abs

    def _advance_hazards(
        self,
        store: dict[str, Any],
        discrete_channels: np.ndarray,
        propensities: np.ndarray,
        dt: float,
    ) -> None:
        # Discrete hazard part of Algorithm 2 step 1.
        # This advances integrated hazards for RD over dt.  The implementation
        # uses one hazard per discrete channel instead of the pseudocode's
        # scalar w = integral sum_{RD} lambda_k(z) dt.
        if not self._uses_discrete_event_hazards():
            return
        if dt <= 0.0 or discrete_channels.size == 0:
            return
        channels = np.asarray(discrete_channels, dtype=np.int64)
        hazards = np.asarray(store["hazards"], dtype=float)
        hazards[channels] += np.maximum(propensities[channels], 0.0) * float(dt)

    def _reset_channel_threshold(
        self,
        store: dict[str, Any],
        channel_id: int,
        rng: np.random.Generator,
    ) -> None:
        cid = int(channel_id)
        thresholds = np.asarray(store["thresholds"], dtype=float)
        hazards = np.asarray(store["hazards"], dtype=float)
        thresholds[cid] = float(rng.exponential(1.0))
        hazards[cid] = 0.0

    def _reset_scalar_discrete_hazard(self, store: dict[str, Any], rng: np.random.Generator) -> None:
        """Reset Algorithm-2 scalar hazard after adaptation."""

        store["discrete_total_hazard"] = 0.0
        store["discrete_hazard_threshold"] = float(rng.exponential(1.0))

    def _scalar_discrete_hazard(self, store: dict[str, Any]) -> float:
        value = float(store.get("discrete_total_hazard", 0.0))
        if not np.isfinite(value) or value < 0.0:
            value = 0.0
            store["discrete_total_hazard"] = value
        return value

    def _scalar_discrete_threshold(self, store: dict[str, Any], rng: np.random.Generator) -> float:
        value = float(store.get("discrete_hazard_threshold", np.inf))
        if not np.isfinite(value):
            value = self._scalar_discrete_hazard(store) + float(rng.exponential(1.0))
            store["discrete_hazard_threshold"] = value
        return value

    def _advance_scalar_discrete_hazard(
        self,
        store: dict[str, Any],
        total_discrete_propensity: float,
        dt: float,
    ) -> None:
        if dt <= 0.0:
            return
        increment = max(float(total_discrete_propensity), 0.0) * float(dt)
        store["discrete_total_hazard"] = self._scalar_discrete_hazard(store) + increment

    def _advance_scalar_discrete_threshold(self, store: dict[str, Any], rng: np.random.Generator) -> None:
        store["discrete_hazard_threshold"] = self._scalar_discrete_threshold(store, rng) + float(rng.exponential(1.0))

    def _channel_is_rq(self, store: dict[str, Any], channel_id: int) -> bool:
        mask = store.get("rq_mask")
        cid = int(channel_id)
        return bool(isinstance(mask, np.ndarray) and 0 <= cid < mask.size and bool(mask[cid]))

    def _affected_channels_for_species_fast(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        store: dict[str, Any],
        changed_species: np.ndarray,
        *,
        fired_channel: int | None,
    ) -> np.ndarray:
        """Map changed species to affected channels with reusable scratch buffers.

        P1 优化点：旧路径委托给 ``affected_channels_for_species``，内部通常是
        ``concatenate + np.unique``。事件很多时这会成为热路径。这里直接读取
        network 的 ``species_to_channels`` 列表，用 marker 去重并把结果写入
        scratch buffer。返回值只在当前刷新调用内有效，调用方不要长期保存。
        """

        species = np.asarray(changed_species, dtype=np.int64)
        if species.ndim != 1:
            raise ValueError("changed_species must be a 1D array")
        if species.size and np.any((species < 0) | (species >= network.n_species)):
            raise IndexError("changed_species contains out-of-range species ids")

        if getattr(network, "dependency_indices_dirty", False) or len(getattr(network, "species_to_channels", [])) != network.n_species:
            rebuild = getattr(network, "rebuild_dependency_indices", None)
            if callable(rebuild):
                rebuild()

        species_to_channels = getattr(network, "species_to_channels", None)
        if not isinstance(species_to_channels, list) or len(species_to_channels) != network.n_species:
            if species.size:
                affected = network.affected_channels_for_species(species)
            else:
                affected = np.empty(0, dtype=np.int64)
            if fired_channel is None:
                return affected
            fired = np.asarray([int(fired_channel)], dtype=np.int64)
            return fired if affected.size == 0 else np.unique(np.concatenate((affected, fired))).astype(np.int64, copy=False)

        marker = self._ensure_runtime_array(
            store,
            "affected_channel_marker",
            size=network.n_channels,
            dtype=bool,
            fill_value=False,
        )
        scratch = self._ensure_runtime_array(
            store,
            "affected_channel_scratch",
            size=network.n_channels,
            dtype=np.int64,
        )
        if self._numba_backend_enabled() and self._has_dependency_csr(network):
            count = int(
                collect_affected_channels_numba(
                    species,
                    -1 if fired_channel is None else int(fired_channel),
                    np.asarray(network.species_to_channels_indptr, dtype=np.int64),
                    np.asarray(network.species_to_channels_indices, dtype=np.int64),
                    marker,
                    scratch,
                    int(network.n_species),
                    int(network.n_channels),
                )
            )
            store["numba_kernel_used"] = True
            store["numba_affected_collects"] = int(store.get("numba_affected_collects", 0)) + 1
            return scratch[:count]

        count = 0
        for sid_value in species:
            channels = np.asarray(species_to_channels[int(sid_value)], dtype=np.int64)
            if channels.size == 0:
                continue
            unmarked = channels[~marker[channels]]
            if unmarked.size == 0:
                continue
            marker[unmarked] = True
            scratch[count : count + unmarked.size] = unmarked
            count += int(unmarked.size)

        if fired_channel is not None:
            cid = int(fired_channel)
            if cid < 0 or cid >= network.n_channels:
                raise IndexError(f"fired_channel out of range: {cid}")
            if not bool(marker[cid]):
                marker[cid] = True
                scratch[count] = cid
                count += 1

        affected = scratch[:count]
        if count:
            marker[affected] = False
        return affected

    def _refresh_propensities_after_state_change(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        state: SystemState,
        store: dict[str, Any],
        changed_species: np.ndarray,
        rng: np.random.Generator,
        *,
        reason: str,
        fired_channel: int | None = None,
    ) -> None:
        cached = store.get("propensities")
        if not isinstance(cached, np.ndarray) or cached.shape != (network.n_channels,):
            cached = np.empty(network.n_channels, dtype=float)
            store["propensities"] = cached
            network.compute_all_propensities(state, out=cached)
            self._clean_propensities(cached)
            self._rebuild_heap_from_propensities(store, state, cached, rng)
            self._rebuild_gillespie_discrete_cache_after_full_compute(network, state, store, cached)
            store["propensities_valid"] = True
            store["propensities_dirty_reason"] = None
            store["propensities_dirty_species"] = np.empty(0, dtype=np.int64)
            store["propensity_update_mode"] = "full"
            store["propensity_update_channels"] = int(network.n_channels)
            return

        if not bool(store.get("propensities_valid", False)):
            network.compute_all_propensities(state, out=cached)
            self._clean_propensities(cached)
            self._rebuild_heap_from_propensities(store, state, cached, rng)
            self._rebuild_gillespie_discrete_cache_after_full_compute(network, state, store, cached)
            store["propensities_valid"] = True
            store["propensities_dirty_reason"] = None
            store["propensities_dirty_species"] = np.empty(0, dtype=np.int64)
            store["propensity_update_mode"] = "full"
            store["propensity_update_channels"] = int(network.n_channels)
            return

        species = np.asarray(changed_species, dtype=np.int64)
        if species.ndim != 1:
            raise ValueError("changed_species must be a 1D array")
        if species.size == 0 and fired_channel is None:
            return

        if self.config.use_local_propensity_updates:
            try:
                affected = self._affected_channels_for_species_fast(
                    network,
                    store,
                    species,
                    fired_channel=fired_channel,
                )
            except Exception:
                affected = np.arange(network.n_channels, dtype=np.int64)
        else:
            affected = np.arange(network.n_channels, dtype=np.int64)
        if affected.size == 0:
            return

        full_recompute = (
            not self.config.use_local_propensity_updates
            or affected.size >= int(np.ceil(self.config.local_propensity_full_recompute_fraction * network.n_channels))
        )
        if full_recompute:
            affected = np.arange(network.n_channels, dtype=np.int64)

        if self._uses_discrete_event_heap():
            old_propensities = np.asarray(cached[affected], dtype=float).copy()
            old_scheduled = np.asarray(store.get("scheduled_times", np.full(network.n_channels, np.inf, dtype=float)))[affected].copy()
        else:
            old_propensities = np.empty(0, dtype=float)
            old_scheduled = np.empty(0, dtype=float)
        if full_recompute:
            network.compute_all_propensities(state, out=cached)
            self._clean_propensities(cached)
            new_propensities = cached[affected]
            update_mode = "full"
            self._rebuild_gillespie_discrete_cache_after_full_compute(network, state, store, cached)
        elif self._try_numba_elementary_gillespie_local_refresh(network, state, store, affected, cached):
            new_propensities = cached[affected]
            update_mode = "local_numba"
        else:
            subset_scratch = self._ensure_runtime_array(
                store,
                "propensity_subset_scratch",
                size=network.n_channels,
                dtype=float,
            )[: affected.size]
            new_propensities = network.compute_propensities_for_channels(affected, state, out=subset_scratch)
            self._clean_propensities(new_propensities)
            cached[affected] = new_propensities
            update_mode = "local"
            self._update_gillespie_discrete_cache_for_channels(network, state, store, affected, cached)

        if self._uses_discrete_event_heap():
            self._reschedule_discrete_heap_channels(
                store,
                affected,
                old_propensities,
                old_scheduled,
                new_propensities,
                now=float(state.t),
                rng=rng,
                fired_channel=fired_channel,
            )
            self._maybe_rebuild_discrete_event_heap(store)

        store["propensities_valid"] = True
        store["propensities_dirty_reason"] = None
        store["propensities_dirty_species"] = np.empty(0, dtype=np.int64)
        store["propensities_last_compute_reason"] = str(reason)
        store["propensity_update_mode"] = update_mode
        store["propensity_update_channels"] = int(affected.size)

    def _rebuild_heap_from_propensities(
        self,
        store: dict[str, Any],
        state: SystemState,
        propensities: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        if not self._uses_discrete_event_heap():
            return
        discrete_mask = np.asarray(store.get("discrete_mask"), dtype=bool)
        scheduled = self._ensure_heap_array(store, "scheduled_times", fill=np.inf)
        versions = self._ensure_heap_versions(store)
        scheduled[:] = np.inf
        versions += 1
        heap: list[tuple[float, int, int]] = []
        for channel_id in np.flatnonzero(discrete_mask).astype(np.int64, copy=False):
            cid = int(channel_id)
            propensity = max(float(propensities[cid]), 0.0)
            if propensity <= self.config.hazard_tol:
                continue
            fire_time = float(state.t) + float(rng.exponential(1.0 / propensity))
            scheduled[cid] = fire_time
            heap.append((fire_time, cid, int(versions[cid])))
        heapq.heapify(heap)
        store["event_heap"] = heap
        store["heap_rebuilds"] = int(store.get("heap_rebuilds", 0)) + 1

    def _reschedule_discrete_heap_channels(
        self,
        store: dict[str, Any],
        channels: np.ndarray,
        old_propensities: np.ndarray,
        old_scheduled: np.ndarray,
        new_propensities: np.ndarray,
        *,
        now: float,
        rng: np.random.Generator,
        fired_channel: int | None,
    ) -> None:
        scheduled = self._ensure_heap_array(store, "scheduled_times", fill=np.inf)
        versions = self._ensure_heap_versions(store)
        heap = self._ensure_event_heap(store)
        discrete_mask = np.asarray(store.get("discrete_mask"), dtype=bool)

        channel_array = np.asarray(channels, dtype=np.int64)
        if channel_array.size == 0:
            return
        valid = (channel_array >= 0) & (channel_array < scheduled.size)
        if discrete_mask.shape == scheduled.shape:
            valid_indices = np.flatnonzero(valid)
            valid[valid_indices] = discrete_mask[channel_array[valid_indices]]
        if not np.any(valid):
            return

        cids = channel_array[valid]
        old_a = np.maximum(np.asarray(old_propensities, dtype=float)[valid], 0.0)
        new_a = np.maximum(np.asarray(new_propensities, dtype=float)[valid], 0.0)
        old_time = np.asarray(old_scheduled, dtype=float)[valid]

        versions[cids] += 1
        current_versions = np.asarray(versions[cids], dtype=np.int64)
        next_times = np.full(cids.shape, np.inf, dtype=float)

        new_active = new_a > self.config.hazard_tol
        if fired_channel is None:
            fired_mask = np.zeros(cids.shape, dtype=bool)
        else:
            fired_mask = cids == int(fired_channel)

        # Unfired channels keep their already sampled exponential threshold.
        # If lambda changes from old_a to new_a, the remaining real time scales
        # by old_a / new_a.  Fired channels, newly active channels, and channels
        # without a finite previous schedule draw a fresh exponential time.
        transform = new_active & ~fired_mask & (old_a > self.config.hazard_tol) & np.isfinite(old_time)
        if np.any(transform):
            remaining = np.maximum(old_time[transform] - float(now), 0.0)
            next_times[transform] = float(now) + (old_a[transform] / new_a[transform]) * remaining

        fresh = new_active & ~transform
        if np.any(fresh):
            next_times[fresh] = float(now) + rng.exponential(1.0 / new_a[fresh])

        scheduled[cids] = next_times
        finite = np.isfinite(next_times)
        if not np.any(finite):
            return

        entries = [
            (float(time), int(channel_id), int(version))
            for time, channel_id, version in zip(next_times[finite], cids[finite], current_versions[finite])
        ]
        if len(entries) > max(64, len(heap) // 4):
            heap.extend(entries)
            heapq.heapify(heap)
            store["heap_batch_heapifies"] = int(store.get("heap_batch_heapifies", 0)) + 1
        else:
            for entry in entries:
                heapq.heappush(heap, entry)

    def _transform_heap_fire_time(
        self,
        now: float,
        old_time: float,
        old_propensity: float,
        new_propensity: float,
        rng: np.random.Generator,
    ) -> float:
        if new_propensity <= self.config.hazard_tol:
            return float("inf")
        if old_propensity <= self.config.hazard_tol or not np.isfinite(old_time):
            return self._sample_heap_fire_time(float(now), float(new_propensity), rng)
        remaining = max(float(old_time) - float(now), 0.0)
        return float(now) + (float(old_propensity) / float(new_propensity)) * remaining

    def _sample_heap_fire_time_or_inf(
        self,
        now: float,
        propensity: float,
        rng: np.random.Generator,
    ) -> float:
        if propensity <= self.config.hazard_tol:
            return float("inf")
        return self._sample_heap_fire_time(now, propensity, rng)

    def _sample_heap_fire_time(self, now: float, propensity: float, rng: np.random.Generator) -> float:
        return float(now) + float(rng.exponential(1.0 / float(propensity)))

    def _changed_species_for_channels(
        self,
        network: ReactionNetworkData | ElementaryMassActionNetwork,
        channels: np.ndarray,
    ) -> np.ndarray:
        ids = np.asarray(channels, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(np.any(network.nu[ids] != 0.0, axis=0)).astype(np.int64, copy=False)

    def _merge_species_ids(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        a = np.asarray(first, dtype=np.int64)
        b = np.asarray(second, dtype=np.int64)
        if a.size == 0:
            return np.unique(b).astype(np.int64, copy=False)
        if b.size == 0:
            return np.unique(a).astype(np.int64, copy=False)
        return np.unique(np.concatenate((a, b))).astype(np.int64, copy=False)

    def _validate_or_clip_state(self, state: SystemState) -> None:
        if not np.all(np.isfinite(state.x)):
            raise ValueError("PDMP produced NaN or inf species counts")
        if self.config.validate_nonnegative and np.any(state.x < -self.config.hazard_tol):
            raise ValueError("PDMP produced negative species counts")
        np.maximum(state.x, 0.0, out=state.x)

    def _validate_partition(self, network: ReactionNetworkData | ElementaryMassActionNetwork, partition: PDMPPartitionResult) -> None:
        for name, values, size in (
            ("continuous_channels", partition.continuous_channels, network.n_channels),
            ("discrete_channels", partition.discrete_channels, network.n_channels),
            ("continuous_species", partition.continuous_species, network.n_species),
            ("discrete_species", partition.discrete_species, network.n_species),
            ("rq_channels", partition.rq_channels, network.n_channels),
        ):
            arr = np.asarray(values, dtype=np.int64)
            if np.any(arr < 0) or np.any(arr >= int(size)):
                raise IndexError(f"{name} contains out-of-range ids")

    def _require_partition(self, store: dict[str, Any]) -> PDMPPartitionResult:
        partition = store.get("partition")
        if not isinstance(partition, PDMPPartitionResult):
            raise RuntimeError("PDMP partition is not initialized")
        return partition

    def _clean_propensities(self, propensities: np.ndarray) -> np.ndarray:
        values = np.asarray(propensities, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("PDMP propensities contain NaN or inf values")
        np.maximum(values, 0.0, out=values)
        return values

    def _discrete_event_method(self) -> str:
        return str(getattr(self.config, "discrete_event_method", "nrm_heap" if self.config.use_discrete_event_heap else "nrm_scan"))

    def _uses_discrete_event_heap(self) -> bool:
        return self._discrete_event_method() == "nrm_heap"

    def _uses_discrete_event_hazards(self) -> bool:
        return self._discrete_event_method() == "nrm_scan"

    def _uses_discrete_event_gillespie(self) -> bool:
        return self._discrete_event_method() == "gillespie"

    def _discrete_event_locator_label(self) -> str:
        method = self._discrete_event_method()
        if method == "nrm_heap":
            return "heap"
        if method == "nrm_scan":
            return "scan"
        return "gillespie"

    def _details(
        self,
        store: dict[str, Any],
        *,
        total_discrete_propensity: float,
        total_continuous_propensity: float,
        continuous_abs_total: np.ndarray,
        repartitions: int,
    ) -> dict[str, Any] | None:
        if not self.config.diagnostics:
            return None
        partition = self._require_partition(store)
        metadata = dict(partition.metadata)
        return {
            "mode": "pdmp",
            "partition_method": metadata.get("method", "unknown"),
            "n_continuous_channels": int(partition.continuous_channels.size),
            "n_discrete_channels": int(partition.discrete_channels.size),
            "n_continuous_species": int(partition.continuous_species.size),
            "n_discrete_species": int(partition.discrete_species.size),
            "n_rq_channels": int(partition.rq_channels.size),
            "fast_subnetwork_count": int(len(partition.fast_subnetworks)),
            "n_repartitions": int(repartitions),
            "total_jump_propensity": max(float(total_discrete_propensity), 0.0),
            "total_cle_propensity": max(float(total_continuous_propensity), 0.0),
            "total_continuous_propensity": max(float(total_continuous_propensity), 0.0),
            "continuous_channel_abs_increments": np.asarray(continuous_abs_total, dtype=float).copy(),
            "discrete_event_method": self._discrete_event_method(),
            "discrete_event_locator": self._discrete_event_locator_label(),
            "propensity_update_mode": str(store.get("propensity_update_mode", "unknown")),
            "propensity_update_channels": int(store.get("propensity_update_channels", 0)),
            "heap_entries": int(len(store.get("event_heap", []))) if self._uses_discrete_event_heap() else 0,
            "heap_stale_pops": int(store.get("heap_stale_pops", 0)) if self._uses_discrete_event_heap() else 0,
            "heap_rebuilds": int(store.get("heap_rebuilds", 0)) if self._uses_discrete_event_heap() else 0,
            "heap_batch_heapifies": int(store.get("heap_batch_heapifies", 0)) if self._uses_discrete_event_heap() else 0,
            "gillespie_cache_valid": bool(store.get("gillespie_cache_valid", False)) if self._uses_discrete_event_gillespie() else False,
            "gillespie_cached_total": float(store.get("gillespie_discrete_total", 0.0)) if self._uses_discrete_event_gillespie() else 0.0,
            "kernel_backend": str(getattr(self.config, "kernel_backend", "auto")),
            "numba_available": bool(NUMBA_AVAILABLE),
            "numba_kernel_used": bool(store.get("numba_kernel_used", False)),
            "numba_local_refreshes": int(store.get("numba_local_refreshes", 0)),
            "numba_cache_rebuilds": int(store.get("numba_cache_rebuilds", 0)),
            "numba_affected_collects": int(store.get("numba_affected_collects", 0)),
            "partition_metadata": metadata,
        }


class CLEStepper(BaseStepper):
    """Chemical Langevin Equation 的最小步进器。

    它只推进被 partition 标为 fast 的 channels。当前实现是显式 Euler-Maruyama
    风格的正态近似增量，并在状态出现负数时裁剪到 0。
    """

    def __init__(
        self,
        *,
        adaptive_dt: bool = True,
        dt_min: float = 1e-12,
        dt_shrink_factor: float = 0.5,
        dt_growth_factor: float = 2.0,
        max_retries: int = 20,
    ):
        self.adaptive_dt = bool(adaptive_dt)
        self.dt_min = float(dt_min)
        self.dt_shrink_factor = float(dt_shrink_factor)
        self.dt_growth_factor = float(dt_growth_factor)
        self.max_retries = int(max_retries)
        if self.dt_min <= 0.0:
            raise ValueError("dt_min must be > 0")
        if not (0.0 < self.dt_shrink_factor < 1.0):
            raise ValueError("dt_shrink_factor must be in (0, 1)")
        if self.dt_growth_factor < 1.0:
            raise ValueError("dt_growth_factor must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._adaptive_dt: float | None = None
        self._last_n_clipped = 0

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

        advance = self._apply_cle_increment_adaptive(
            state,
            dt,
            context,
            channels,
            grow_on_success=True,
        )
        state.t += float(advance.advanced_time)
        state.step_count += 1
        return StepResult(
            advanced_time=float(advance.advanced_time),
            event_occurred=False,
            details={
                "mode": "cle",
                "n_fast_channels": int(channels.size),
                "continuous_channel_abs_increments": advance.continuous_abs,
                "cle_requested_dt": float(advance.requested_dt),
                "cle_accepted_dt": float(advance.advanced_time),
                "cle_rejected_attempts": int(advance.rejected_attempts),
                "cle_dt_after": None if advance.dt_after is None else float(advance.dt_after),
                "cle_dt_min_reached": bool(advance.min_dt_reached),
                "n_clipped": int(advance.n_clipped),
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
        x_new, continuous_abs, n_clipped = self._cle_increment_candidate(state, dt, context, channels)
        self._last_n_clipped = int(n_clipped)
        state.x[:] = x_new
        return continuous_abs

    def _apply_cle_increment_adaptive(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        channels: np.ndarray,
        *,
        grow_on_success: bool,
    ) -> _FastCLEAdvanceResult:
        requested_dt = max(float(dt), 0.0)
        if requested_dt <= 0.0:
            return _FastCLEAdvanceResult(
                continuous_abs=np.zeros(context.network.n_channels, dtype=float),
                advanced_time=0.0,
                requested_dt=0.0,
                rejected_attempts=0,
                n_clipped=0,
                dt_after=self._adaptive_dt,
                min_dt_reached=False,
            )

        if not self.adaptive_dt:
            continuous_abs = self._apply_cle_increment(state, requested_dt, context, channels)
            return _FastCLEAdvanceResult(
                continuous_abs=continuous_abs,
                advanced_time=requested_dt,
                requested_dt=requested_dt,
                rejected_attempts=0,
                n_clipped=self._last_n_clipped,
                dt_after=None,
                min_dt_reached=False,
            )

        if self._adaptive_dt is None:
            self._adaptive_dt = requested_dt
        current_dt = max(float(self._adaptive_dt), self.dt_min)
        attempt_dt = min(requested_dt, current_dt)
        rejected = 0

        while True:
            x_new, continuous_abs, n_clipped = self._cle_increment_candidate(state, attempt_dt, context, channels)
            if n_clipped == 0:
                state.x[:] = x_new
                self._last_n_clipped = 0
                if grow_on_success and rejected == 0 and attempt_dt >= current_dt - 1e-15:
                    self._adaptive_dt = max(self.dt_min, current_dt * self.dt_growth_factor)
                elif rejected:
                    self._adaptive_dt = attempt_dt
                return _FastCLEAdvanceResult(
                    continuous_abs=continuous_abs,
                    advanced_time=attempt_dt,
                    requested_dt=requested_dt,
                    rejected_attempts=rejected,
                    n_clipped=0,
                    dt_after=float(self._adaptive_dt),
                    min_dt_reached=False,
                )

            if rejected >= self.max_retries or attempt_dt <= self.dt_min * (1.0 + 1e-12):
                state.x[:] = x_new
                self._last_n_clipped = int(n_clipped)
                self._adaptive_dt = max(self.dt_min, attempt_dt)
                return _FastCLEAdvanceResult(
                    continuous_abs=continuous_abs,
                    advanced_time=attempt_dt,
                    requested_dt=requested_dt,
                    rejected_attempts=rejected,
                    n_clipped=int(n_clipped),
                    dt_after=float(self._adaptive_dt),
                    min_dt_reached=True,
                )

            rejected += 1
            attempt_dt = max(self.dt_min, attempt_dt * self.dt_shrink_factor)
            self._adaptive_dt = attempt_dt

    def _cle_increment_candidate(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        channels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        network = context.network
        rng = context.rng
        candidate = np.asarray(state.x, dtype=float).copy()
        trial_state = SystemState(
            t=float(state.t),
            x=candidate,
            step_count=int(state.step_count),
            event_count=int(state.event_count),
            partition_state=state.partition_state,
        )
        continuous_abs = np.zeros(network.n_channels, dtype=float)
        for channel_id in channels:
            a = network.compute_propensity(int(channel_id), trial_state)
            mean = a * float(dt)
            if mean <= 0.0:
                continue
            amount = mean + np.sqrt(mean) * float(rng.normal())
            continuous_abs[int(channel_id)] += abs(float(amount))
            network.apply_channel_delta(candidate, int(channel_id), amount)
        if not np.all(np.isfinite(candidate)):
            raise ValueError("CLE increment produced NaN or inf state values")
        negative = candidate < 0.0
        n_clipped = int(np.count_nonzero(negative))
        if n_clipped:
            candidate = np.maximum(candidate, 0.0)
        return candidate, continuous_abs, n_clipped


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
            fast = self._advance_fast(state, dt, context, partition.fast_channels, grow_on_success=True)
            return StepResult(
                advanced_time=float(fast.advanced_time),
                event_occurred=False,
                propensity_sum=0.0,
                details={
                    "mode": "hybrid",
                    "n_fast_channels": int(partition.fast_channels.size),
                    "continuous_channel_abs_increments": fast.continuous_abs,
                    **self._fast_cle_details(fast),
                },
            )

        tau = float(context.rng.exponential(1.0 / slow_total))
        if tau > dt:
            fast = self._advance_fast(state, dt, context, partition.fast_channels, grow_on_success=True)
            return StepResult(
                advanced_time=float(fast.advanced_time),
                event_occurred=False,
                propensity_sum=slow_total,
                tau=tau,
                details={
                    "mode": "hybrid",
                    "n_fast_channels": int(partition.fast_channels.size),
                    "continuous_channel_abs_increments": fast.continuous_abs,
                    **self._fast_cle_details(fast),
                },
            )

        fast = self._advance_fast(state, tau, context, partition.fast_channels, grow_on_success=False)
        if fast.advanced_time < tau - 1e-15 or fast.rejected_attempts > 0:
            return StepResult(
                advanced_time=float(fast.advanced_time),
                event_occurred=False,
                propensity_sum=slow_total,
                tau=tau,
                details={
                    "mode": "hybrid_cle_retry",
                    "n_fast_channels": int(partition.fast_channels.size),
                    "continuous_channel_abs_increments": fast.continuous_abs,
                    "scheduled_slow_event_deferred": True,
                    **self._fast_cle_details(fast),
                },
            )

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
                    "continuous_channel_abs_increments": fast.continuous_abs,
                    **self._fast_cle_details(fast),
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
            "continuous_channel_abs_increments": fast.continuous_abs,
            **self._fast_cle_details(fast),
            },
        )

    def _advance_fast(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        fast_channels: np.ndarray,
        *,
        grow_on_success: bool,
    ) -> _FastCLEAdvanceResult:
        continuous_abs = np.zeros(context.network.n_channels, dtype=float)
        if fast_channels.size:
            fast = self._cle._apply_cle_increment_adaptive(
                state,
                dt,
                context,
                fast_channels,
                grow_on_success=grow_on_success,
            )
        else:
            fast = _FastCLEAdvanceResult(
                continuous_abs=continuous_abs,
                advanced_time=float(dt),
                requested_dt=float(dt),
                rejected_attempts=0,
                n_clipped=0,
                dt_after=self._cle._adaptive_dt,
                min_dt_reached=False,
            )
        state.t += float(fast.advanced_time)
        state.step_count += 1
        return fast

    def _fast_cle_details(self, fast: _FastCLEAdvanceResult) -> dict[str, Any]:
        return {
            "cle_requested_dt": float(fast.requested_dt),
            "cle_accepted_dt": float(fast.advanced_time),
            "cle_rejected_attempts": int(fast.rejected_attempts),
            "cle_dt_after": None if fast.dt_after is None else float(fast.dt_after),
            "cle_dt_min_reached": bool(fast.min_dt_reached),
            "n_clipped": int(fast.n_clipped),
        }


class BlendedHybridStepper(BaseStepper):
    """基于 beta 权重的 SSA/CLE blending 步进器。

    beta=1 表示该 channel 完全按离散 jump 处理，beta=0 表示完全按 CLE 连续
    处理，中间值会把 propensity 拆成 jump 与 CLE 两部分。当前离散部分每个
    step 至多采样一个 Direct SSA 事件。
    """

    def __init__(self, config: BlendedHybridConfig | None = None):
        self.config = config or BlendedHybridConfig()
        self._nu_cache: dict[int, np.ndarray] = {}
        self._beta_lookup_cache: dict[tuple[int, str], _ChannelBetaLookup] = {}
        self._last_n_clipped = 0
        self._last_n_low_count_rounded = 0
        self._last_total_cle_propensity = 0.0
        self._last_continuous_channel_abs_increments = np.empty(0, dtype=float)
        self._reaction_interval_dt: float | None = None
        self._adaptive_dt_cle = self._clamp_cle_dt(self.config.dt_cle)
        self._adaptive_dt_macro = self._clamp_cle_dt(self.config.effective_dt_macro)
        self._channel_beta_cache: np.ndarray | None = None
        self._species_beta_cache: np.ndarray | None = None
        self._channel_beta_state_cache: np.ndarray | None = None
        self._channel_beta_cache_key: tuple[int, str, float, float, str] | None = None
        self._last_beta_full_recompute = False
        self._last_beta_affected_updates = 0
        self._last_beta_reuse_network_id: int | None = None
        self._last_beta_reuse_changed_species = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_affected_channels = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_affected_species = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_beta_channels = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_extra_channels = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_changed_catalyst_species = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_catalyst_channels = np.empty(0, dtype=np.int64)
        self._observed_propensity_cache: np.ndarray | None = None
        self._observed_propensity_state: np.ndarray | None = None
        self._observed_propensity_network_id: int | None = None
        self._last_observed_propensity_full_recompute = False
        self._last_observed_propensity_affected_updates = 0
        self._last_observed_propensity_update_path = "not_used"
        self._last_observed_propensity_reused_beta_affected = False
        self._last_observed_propensity_beta_affected_species = 0
        self._last_observed_propensity_beta_affected_channels = 0
        self._last_observed_propensity_beta_extra_channels = 0
        self._last_observed_propensity_changed_catalyst_species = 0
        self._last_observed_propensity_catalyst_affected_channels = 0
        self._stoich_sparsity_profile = None

    def invalidate_cache(self) -> None:
        self._invalidate_beta_cache()
        self._invalidate_observed_propensity_cache()

    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        if dt <= 0.0:
            return StepResult(advanced_time=0.0, event_occurred=False, details={"mode": "blended_no_dt"})

        network = context.network
        self._maybe_update_reaction_interval_dt(network, state) # 自适应dt,默认不开启
        x_float = self._float_nonnegative(state.x) # 返回一个浮点数数组，确保所有元素非负
        observed = self._rounded_nonnegative(x_float) # beta 和 mixed propensity 都基于同一个 rounded observed state
        beta = self._channel_betas(network, observed) # 根据x计算当前全局的beta，可能存在小数组问题（一个反应可能只涉及1、2、3个物质但要查表）
        beta_min = float(np.min(beta)) if beta.size else 0.0
        beta_max = float(np.max(beta)) if beta.size else 0.0

        if beta_max <= self.config.beta_tol: # 1e-12
            return self._pure_cle_step(state, float(dt), context, beta, beta_min, beta_max)
        if beta_min >= 1.0 - self.config.beta_tol:
            return self._pure_ssa_step(state, float(dt), context, beta_min, beta_max, observed=observed)
        return self._mixed_step(state, float(dt), context, beta, beta_min, beta_max, observed=observed) # 一般走这个路，检查一下

    def _pure_cle_step(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        beta: np.ndarray,
        beta_min: float,
        beta_max: float,
    ) -> StepResult:
        requested = min(self._current_dt_macro(), dt)
        cle = self._adaptive_cle_increment(
            context.network,
            state.x,
            beta,
            requested,
            context.rng,
            kind="macro",
            grow_on_success=True,
        )
        state.x[:] = cle.x
        state.t += cle.dt
        state.step_count += 1
        return StepResult(
            advanced_time=cle.dt,
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
                "stepper_dt": cle.dt,
                "reaction_interval_dt": self._reaction_interval_dt,
                "continuous_channel_abs_increments": cle.continuous_abs.copy(),
                **self._observed_propensity_details(),
                **self._adaptive_cle_details(cle),
            },
        )

    def _pure_ssa_step(
        self,
        state: SystemState,
        dt: float,
        context: StepperContext,
        beta_min: float,
        beta_max: float,
        *,
        observed: np.ndarray | None = None,
    ) -> StepResult:
        network = context.network
        duration = min(self._current_dt_macro(), dt)
        observed = self._rounded_nonnegative(state.x) if observed is None else np.asarray(observed, dtype=float)
        if self.config.strict_int_for_CLE:
            propensities = self._propensities_for_observed_cached(network, observed, state.t, "jump propensities")
        else:
            self._last_observed_propensity_update_path = "full_uncached"
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
            **self._observed_propensity_details(),
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
        *,
        observed: np.ndarray | None = None,
    ) -> StepResult:
        network = context.network
        duration = min(self._current_dt_cle(), dt)
        observed = self._rounded_nonnegative(state.x) if observed is None else np.asarray(observed, dtype=float) # 若没有传入rounded，现场做。
        if self.config.strict_int_for_CLE: # 采取共享propensity
            base_jump = self._propensities_for_observed_cached(network, observed, state.t, "jump propensities") # 第一次计算propensity
        else:
            self._last_observed_propensity_update_path = "full_uncached"
            base_jump = self._propensities_for_x(network, observed, state.t) # 耗时大头
            base_jump = self._clean_propensities(base_jump, "jump propensities") # 把负数换成0
        lambda_jump = self._clean_propensities(beta * base_jump, "split jump propensities (beta or propensity may be negative)") # 这里是全量beta*propensity,能否结合“改变的mask”来减少？
        total_jump = float(np.sum(lambda_jump))

        tau = float("inf")
        sampled_channel: int | None = None
        if total_jump > 0.0:
            tau = float(context.rng.exponential(1.0 / total_jump))
            sampled_channel = _sample_channel( # 确定离散通道发生的反应 耗时项
                np.arange(network.n_channels, dtype=np.int64),
                lambda_jump, # 乘上beta的propensity就得到了离散通道
                total_jump,
                context.rng,
            )

        if tau < duration and sampled_channel is not None: # 如果采样的tau小于duration,说明有离散事件发生
            cle = self._adaptive_cle_increment( # 这里的x会内部进行round,检查后续是否需要，如果不需要就删掉
                network,
                state.x,
                beta,
                tau,
                context.rng,
                kind="cle",
                grow_on_success=False,
                base_propensities=base_jump if self.config.strict_int_for_CLE else None,
            )
            state.x[:] = cle.x
            if cle.dt < tau - self.config.beta_tol or cle.rejected_attempts > 0: # 在离散通道预计发生反应之前，CLE通道出事
                state.t += cle.dt
                state.step_count += 1
                return StepResult(
                    advanced_time=cle.dt,
                    event_occurred=False,
                    propensity_sum=total_jump,
                    tau=tau,
                    details={
                        "mode": "mixed_cle_retry",
                        "fired_channel": None,
                        "beta_min": beta_min,
                        "beta_max": beta_max,
                        "total_jump_propensity": total_jump,
                        "total_cle_propensity": self._last_total_cle_propensity,
                        "n_clipped": self._last_n_clipped,
                        "n_low_count_rounded": self._last_n_low_count_rounded,
                        "stepper_dt": cle.dt,
                        "reaction_interval_dt": self._reaction_interval_dt,
                        "scheduled_channel": int(sampled_channel),
                        "scheduled_tau": float(tau),
                        "scheduled_discrete_event_deferred": True,
                        "continuous_channel_abs_increments": cle.continuous_abs.copy(),
                        **self._observed_propensity_details(),
                        **self._adaptive_cle_details(cle),
                    },
                )
            #正常得到CLE结果后
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
                    "continuous_channel_abs_increments": cle.continuous_abs.copy(),
                    **self._observed_propensity_details(),
                    **self._adaptive_cle_details(cle),
                },
            )
        # 如果没有离散事件发生
        cle = self._adaptive_cle_increment(
            network,
            state.x,
            beta,
            duration,
            context.rng,
            kind="cle",
            grow_on_success=True,
            base_propensities=base_jump if self.config.strict_int_for_CLE else None,
        )
        state.x[:] = cle.x
        state.t += cle.dt
        state.step_count += 1
        return StepResult(
            advanced_time=cle.dt,
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
                "stepper_dt": cle.dt,
                "reaction_interval_dt": self._reaction_interval_dt,
                "continuous_channel_abs_increments": cle.continuous_abs.copy(),
                **self._observed_propensity_details(),
                **self._adaptive_cle_details(cle),
            },
        )
        
    @profile
    def _cle_increment(
        self,
        network: ReactionNetworkData,
        x_float: np.ndarray,
        beta: np.ndarray,
        dt: float,
        rng: np.random.Generator,
        *,
        base_propensities: np.ndarray | None = None,
    ) -> np.ndarray:
        if dt <= 0.0:
            self._last_n_clipped = 0
            self._last_n_low_count_rounded = 0
            self._last_total_cle_propensity = 0.0
            self._last_continuous_channel_abs_increments = np.zeros(network.n_channels, dtype=float)
            return self._float_nonnegative(x_float)

        self._last_n_low_count_rounded = 0
        x0 = self._float_nonnegative(x_float) # 又来一次
        if base_propensities is None: # 如果没有传入缓存的propensity
            propensity_state = self._rounded_nonnegative(x0) if self.config.strict_int_for_CLE else x0
            if self.config.strict_int_for_CLE:
                prop = self._propensities_for_observed_cached(network, propensity_state, 0.0, "CLE propensities")
            else:
                self._last_observed_propensity_update_path = "full_uncached"
                prop = self._propensities_for_x(network, propensity_state, 0.0) # 耗时项
                prop = self._clean_propensities(prop, "CLE propensities")
        else: # 有propensity缓存就直接用，还是已经检查过non_negative的
            prop = np.asarray(base_propensities, dtype=float)
            if prop.shape != (network.n_channels,):
                raise ValueError(f"base_propensities must have shape ({network.n_channels},)")

        prop_cle = self._clean_propensities((1.0 - beta) * prop, "split CLE propensities") # 连续通道propensity
        self._last_total_cle_propensity = float(np.sum(prop_cle))

        # 具体计算CLE
        means = prop_cle * float(dt)
        amounts = means + np.sqrt(np.maximum(means, 0.0)) * rng.normal(size=network.n_channels)
        self._last_continuous_channel_abs_increments = np.abs(amounts).astype(float, copy=False)
        
        S = self._stoichiometry_matrix(network)
        ### 测试函数
        

        # # ---- sparsity profiling ----
        # if not hasattr(self, "_stoich_sparsity_profile"):
        #     nnz = np.count_nonzero(S)
        #     total = S.size
            
        #     self._stoich_sparsity_profile = {
        #         "n_reactions": S.shape[0],
        #         "n_species": S.shape[1],
        #         "matrix_size": total,
        #         "nnz": nnz,
        #         "density": nnz / total,
        #         "sparsity": 1 - nnz / total,
        #     }

        increment = amounts @ S
        ### 测试函数结束
        
        # increment = amounts @ self._stoichiometry_matrix(network) # 如何优化？
        #尝试1：自动转化为稀疏矩阵，并采取切片更新
        # S = csr_matrix(S)
        # # increment = amounts @ S
        # cle_mask = beta < 1

        # amounts_cle = amounts[cle_mask]

        # S_cle = S[cle_mask,:]

        # increment = amounts_cle @ S_cle
        
        
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

    def _adaptive_cle_increment( # 推进CLE到指定时间的核心函数，返回一个 _AdaptiveCLEResult 对象
        self,
        network: ReactionNetworkData,
        x_float: np.ndarray,
        beta: np.ndarray,
        requested_dt: float,
        rng: np.random.Generator,
        *,
        kind: str,
        grow_on_success: bool,
        base_propensities: np.ndarray | None = None,
    ) -> _AdaptiveCLEResult:
        requested = max(float(requested_dt), 0.0)
        if requested <= 0.0:
            x_new = self._cle_increment(
                network,
                x_float,
                beta,
                0.0,
                rng,
                base_propensities=base_propensities,
            )
            return _AdaptiveCLEResult(
                x=x_new,
                continuous_abs=self._last_continuous_channel_abs_increments.copy(),
                dt=0.0,
                requested_dt=0.0,
                rejected_attempts=0,
                n_clipped=0,
                n_low_count_rounded=0,
                total_cle_propensity=0.0,
                dt_after=self._get_adaptive_dt(kind),
                min_dt_reached=False,
            )

        if not self.config.adaptive_cle_dt: # True时会重试
            x_new = self._cle_increment(
                network,
                x_float,
                beta,
                requested,
                rng,
                base_propensities=base_propensities,
            )
            return _AdaptiveCLEResult(
                x=x_new,
                continuous_abs=self._last_continuous_channel_abs_increments.copy(),
                dt=requested,
                requested_dt=requested,
                rejected_attempts=0,
                n_clipped=self._last_n_clipped,
                n_low_count_rounded=self._last_n_low_count_rounded,
                total_cle_propensity=self._last_total_cle_propensity,
                dt_after=requested,
                min_dt_reached=False,
            )

        current_dt = self._get_adaptive_dt(kind)
        attempt_dt = min(requested, current_dt)
        rejected = 0

        while True:
            x_new = self._cle_increment(
                network,
                x_float,
                beta,
                attempt_dt,
                rng,
                base_propensities=base_propensities,
            )
            n_clipped = int(self._last_n_clipped)
            n_low_count_rounded = int(self._last_n_low_count_rounded)
            total_cle_propensity = float(self._last_total_cle_propensity)
            continuous_abs = self._last_continuous_channel_abs_increments.copy()

            if n_clipped == 0:
                if rejected:
                    self._set_adaptive_dt(kind, attempt_dt)
                elif grow_on_success and attempt_dt >= current_dt - self.config.beta_tol:
                    self._grow_adaptive_dt(kind, current_dt)
                return _AdaptiveCLEResult(
                    x=x_new,
                    continuous_abs=continuous_abs,
                    dt=attempt_dt,
                    requested_dt=requested,
                    rejected_attempts=rejected,
                    n_clipped=0,
                    n_low_count_rounded=n_low_count_rounded,
                    total_cle_propensity=total_cle_propensity,
                    dt_after=self._get_adaptive_dt(kind),
                    min_dt_reached=False,
                )

            if rejected >= self.config.cle_dt_max_retries or attempt_dt <= self.config.cle_dt_min * (1.0 + 1e-12):
                self._set_adaptive_dt(kind, attempt_dt)
                return _AdaptiveCLEResult(
                    x=x_new,
                    continuous_abs=continuous_abs,
                    dt=attempt_dt,
                    requested_dt=requested,
                    rejected_attempts=rejected,
                    n_clipped=n_clipped,
                    n_low_count_rounded=n_low_count_rounded,
                    total_cle_propensity=total_cle_propensity,
                    dt_after=self._get_adaptive_dt(kind),
                    min_dt_reached=True,
                )

            rejected += 1
            attempt_dt = self._shrink_adaptive_dt(kind, attempt_dt)

    def _adaptive_cle_details(self, result: _AdaptiveCLEResult) -> dict[str, Any]:
        return {
            "cle_requested_dt": float(result.requested_dt),
            "cle_accepted_dt": float(result.dt),
            "cle_rejected_attempts": int(result.rejected_attempts),
            "cle_dt_after": float(result.dt_after),
            "cle_dt_min_reached": bool(result.min_dt_reached),
        }

    def _channel_betas(self, network: ReactionNetworkData, x: np.ndarray) -> np.ndarray:
        # lookup 是拓扑缓存：记录 channel -> relevant species，以及反向的 species -> affected beta channels。
        # 它只依赖 network 和 beta_species_mode，不依赖当前 state.x。
        lookup = self._channel_beta_lookup(network) # 获得映射关系：每个 reaction/channel 应该看哪些 species 来算 beta
        if lookup.n_channels == 0 or lookup.relevant_species.shape[1] == 0:
            return np.zeros(lookup.n_channels, dtype=float)

        values = np.asarray(x, dtype=float)
        if values.shape[0] < lookup.n_species:
            raise ValueError("state has fewer species than the reaction network")

        # 数值缓存的 key 必须包含 network、beta 选择模式和阈值；阈值变了，旧 beta 缓存就无效。
        cache_key = self._beta_cache_key(network, lookup)
        beta_state = values[: lookup.n_species] # 只取前 n_species 个元素，避免 state.x 里有多余的元素，这里的beta_state其实是rounded_x_state

        if self.config.beta_compute_mode == "beta_fully_compute":
            # 全量模式：直接由当前 state 计算全量 species beta 和 channel beta。
            # 既然已经全量计算，就不再在全量结果上叠加局部更新判断。
            species_beta = _species_beta_array(beta_state, self.config.i1, self.config.i2) # 根据提供的state计算每个 species 的 beta
            return self._recompute_channel_beta_cache(lookup, species_beta, cache_key, beta_state=beta_state)
        # beta_compute_by_state_difference 模式：先看 rounded observed state 哪些 species 变了，
        # 再把 changed species 交给局部 beta 更新路径处理。
        if not self._channel_beta_cache_matches(lookup, cache_key):
            return self._recompute_channel_beta_cache_for_state(lookup, beta_state, cache_key)
        return self._channel_betas_by_state_difference(lookup, beta_state, cache_key)

    def _channel_betas_by_state_difference( # state cache 差异检测
        self,
        lookup: _ChannelBetaLookup,
        beta_state: np.ndarray,
        cache_key: tuple[int, str, float, float, str],
    ) -> np.ndarray:
        # state-difference 模式这里只负责比较当前 state 与上一次 state cache。
        cached_state = self._channel_beta_state_cache # 这里的beta_state也是缓存的rounded state
        cached_channel_beta = self._channel_beta_cache
        if cached_state is None or cached_channel_beta is None:
            # 防御性分支：理论上 _channel_beta_cache_matches 已经排除了 None。
            raise RuntimeError("channel beta state cache is not initialized")

        # 这里比较的是提前 round 后的 state，因此不会被 CLE 的微小浮点扰动放大。
        # species_changed_mask 只表达 state/species 是否改变，不再先算 beta_changed_mask。
        species_changed_mask = beta_state != cached_state # 这里的beta_state其实是rounded state
        changed_state_species = np.flatnonzero(species_changed_mask)
        if changed_state_species.size == 0:
            self._last_beta_full_recompute = False
            self._last_beta_affected_updates = 0
            self._clear_beta_reuse_hint()
            return cached_channel_beta

        return self._update_channel_beta_cache_for_changed_species(
            lookup,
            beta_state,
            cache_key,
            changed_state_species,
        )

    def _recompute_channel_beta_cache_for_state(
        self,
        lookup: _ChannelBetaLookup,
        beta_state: np.ndarray,
        cache_key: tuple[int, str, float, float, str],
    ) -> np.ndarray:
        species_beta = _species_beta_array(beta_state, self.config.i1, self.config.i2)
        return self._recompute_channel_beta_cache(lookup, species_beta, cache_key, beta_state=beta_state)

    def _update_channel_beta_cache_for_changed_species(
        self,
        lookup: _ChannelBetaLookup,
        beta_state: np.ndarray,
        cache_key: tuple[int, str, float, float, str],
        changed_state_species: np.ndarray,
    ) -> np.ndarray:
        cached_state = self._channel_beta_state_cache
        cached_species_beta = self._species_beta_cache # 只是引用
        cached_channel_beta = self._channel_beta_cache
        if cached_state is None or cached_species_beta is None or cached_channel_beta is None:
            # 防御性分支：理论上 _channel_beta_cache_matches 已经排除了 None。
            return self._recompute_channel_beta_cache_for_state(lookup, beta_state, cache_key)

        # 先从 state 变化出发，通过 reverse mapping 找出这些 species 会影响哪些 channel 的 beta。
        # 这里查到的是 changed_species_channels，不是 beta 本身。
        changed_species_channels = _lookup_beta_affected_channels(lookup, changed_state_species)

        if changed_species_channels.size == 0:
            # 这些 species 不参与任何 channel beta；species beta 缓存无需更新。
            cached_state[changed_state_species] = beta_state[changed_state_species]
            self._last_beta_full_recompute = False
            self._last_beta_affected_updates = 0
            self._set_beta_reuse_hint(
                cache_key,
                changed_state_species,
                changed_species_channels,
                np.empty(0, dtype=np.int64),
            )
            return cached_channel_beta
        if (
            changed_species_channels.size >= lookup.n_channels
            # or changed_species_channels.size > self._beta_local_update_limit(lookup.n_channels) # 暂时不开启限制，直接看局部更新会有多耗时
        ):
            # beta 全量计算很便宜；affected 过大或当前规模禁用局部更新时，full recompute 更稳。
            return self._recompute_channel_beta_cache_for_state(lookup, beta_state, cache_key)

        # 只有 very small affected set 才局部更新 channel beta。
        # 通过 changed_species_channels 的 relevant species 得到 affected_species，再只重算这些 species beta。
        affected_species = _lookup_beta_relevant_species_for_channels(lookup, changed_species_channels)
        if affected_species.size != 0:
            cached_species_beta[affected_species] = _species_beta_array(
                beta_state[affected_species],
                self.config.i1,
                self.config.i2,
            )
        cached_channel_beta[changed_species_channels] = self._compute_channel_betas_for_channels(
            lookup,
            cached_species_beta,
            changed_species_channels,
        )
        cached_state[changed_state_species] = beta_state[changed_state_species]
        self._last_beta_full_recompute = False
        self._last_beta_affected_updates = int(changed_species_channels.size)
        self._set_beta_reuse_hint(
            cache_key,
            changed_state_species,
            changed_species_channels,
            affected_species,
        )
        return cached_channel_beta

    def _channel_beta_lookup(self, network: ReactionNetworkData) -> _ChannelBetaLookup:
        mode = self.config.beta_species_mode # β计算引入哪些模式
        key = (id(network), mode)
        cached = self._beta_lookup_cache.get(key)
        if (
            cached is not None
            and cached.n_channels == int(network.n_channels)
            and cached.n_species == int(network.n_species)
            and cached.mode == mode
        ): # 每一步都要核验cached信息
            return cached

        lookup = _build_channel_beta_lookup(network, mode) # 这里是一个缓存表，如果上面cached不存在就会计算
        self._beta_lookup_cache[key] = lookup
        return lookup

    def _invalidate_beta_cache(self) -> None:
        # 清数值缓存；不清 _beta_lookup_cache，因为 lookup 是 network 拓扑缓存。
        self._channel_beta_cache = None
        self._species_beta_cache = None
        self._channel_beta_state_cache = None
        self._channel_beta_cache_key = None
        self._last_beta_full_recompute = False
        self._last_beta_affected_updates = 0
        self._clear_beta_reuse_hint()

    def _clear_beta_reuse_hint(self) -> None:
        self._last_beta_reuse_network_id = None
        self._last_beta_reuse_changed_species = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_affected_channels = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_affected_species = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_beta_channels = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_extra_channels = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_changed_catalyst_species = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_catalyst_channels = np.empty(0, dtype=np.int64)

    def _set_beta_reuse_hint(
        self,
        cache_key: tuple[int, str, float, float, str],
        changed_species: np.ndarray,
        beta_channels: np.ndarray,
        affected_species: np.ndarray,
    ) -> None:
        # 这个 hint 只给同一次 step 里随后的 observed propensity 局部更新复用。
        # beta 路径只记录它已经算出的 changed species、beta affected channels 和相关 species。
        # propensity-only extra/catalyst channels 延后到 observed propensity 局部更新真正命中时再查。
        self._last_beta_reuse_network_id = int(cache_key[0])
        self._last_beta_reuse_changed_species = np.array(changed_species, dtype=np.int64, copy=True)
        self._last_beta_reuse_affected_channels = np.array(beta_channels, dtype=np.int64, copy=True)
        self._last_beta_reuse_affected_species = np.array(affected_species, dtype=np.int64, copy=True)
        self._last_beta_reuse_beta_channels = np.array(beta_channels, dtype=np.int64, copy=True)
        self._last_beta_reuse_extra_channels = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_changed_catalyst_species = np.empty(0, dtype=np.int64)
        self._last_beta_reuse_catalyst_channels = np.empty(0, dtype=np.int64)

    def _beta_cache_key(
        self,
        network: ReactionNetworkData,
        lookup: _ChannelBetaLookup,
    ) -> tuple[int, str, float, float, str]:
        return (
            id(network),
            lookup.mode,
            float(self.config.i1),
            float(self.config.i2),
            str(self.config.beta_compute_mode),
        )

    def _channel_beta_cache_matches(
        self,
        lookup: _ChannelBetaLookup,
        cache_key: tuple[int, str, float, float, str],
    ) -> bool:
        # 数值缓存必须同时匹配 key 和数组形状；shape 检查防止同 id 对象被重建/替换后误用旧缓存。
        return (
            self._channel_beta_cache is not None
            and self._species_beta_cache is not None
            and self._channel_beta_state_cache is not None
            and self._channel_beta_cache_key == cache_key
            and self._channel_beta_cache.shape == (lookup.n_channels,)
            and self._species_beta_cache.shape == (lookup.n_species,)
            and self._channel_beta_state_cache.shape == (lookup.n_species,)
        )

    def _recompute_channel_beta_cache(
        self,
        lookup: _ChannelBetaLookup,
        species_beta: np.ndarray,
        cache_key: tuple[int, str, float, float, str],
        *,
        beta_state: np.ndarray | None = None,
    ) -> np.ndarray:
        # 全量路径：先由 species_beta 查表得到每个 channel 的 beta，再安装为当前数值缓存。
        beta = self._compute_channel_betas_from_species_beta(lookup, species_beta)
        self._channel_beta_cache = beta
        self._species_beta_cache = np.array(species_beta, dtype=float, copy=True)
        if beta_state is None:
            self._channel_beta_state_cache = None
        else:
            self._channel_beta_state_cache = np.array(beta_state[: lookup.n_species], dtype=float, copy=True)
        self._channel_beta_cache_key = cache_key
        self._last_beta_full_recompute = True
        self._last_beta_affected_updates = 0
        self._clear_beta_reuse_hint()
        return beta

    def _compute_channel_betas_from_species_beta( # 在propensity大头解决之前都不需要考虑这里的小数组问题
        self,
        lookup: _ChannelBetaLookup,
        species_beta: np.ndarray,
    ) -> np.ndarray:
        if lookup.relevant_species.shape[1] == 0:
            return np.zeros(lookup.n_channels, dtype=float)
        relevant_beta = species_beta[lookup.relevant_species] # 查表
        relevant_beta *= lookup.relevant_mask
        return np.max(relevant_beta, axis=1) # 只看最可能需要SSA的物种的beta

    def _compute_channel_betas_for_channels(
        self,
        lookup: _ChannelBetaLookup,
        species_beta: np.ndarray,
        channels: np.ndarray,
    ) -> np.ndarray:
        affected = np.asarray(channels, dtype=np.int64)
        if affected.size == 0:
            return np.empty(0, dtype=float)
        relevant_beta = species_beta[lookup.relevant_species[affected]]
        relevant_beta *= lookup.relevant_mask[affected]
        return np.max(relevant_beta, axis=1)

    def _beta_local_update_limit(self, n_channels: int) -> int:
        # beta 的全量计算通常只是小矩阵 gather/max，比局部维护还便宜。
        # 小网络默认禁用 beta 局部更新；大网络只允许极小 affected set 走局部路径。
        count = int(n_channels)
        if count < 1024:
            return 0
        return max(8, min(128, int(0.05 * count)))

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

    def _invalidate_observed_propensity_cache(self) -> None:
        self._observed_propensity_cache = None
        self._observed_propensity_state = None
        self._observed_propensity_network_id = None
        self._last_observed_propensity_full_recompute = False
        self._last_observed_propensity_affected_updates = 0
        self._last_observed_propensity_update_path = "not_used"
        self._last_observed_propensity_reused_beta_affected = False
        self._last_observed_propensity_beta_affected_species = 0
        self._last_observed_propensity_beta_affected_channels = 0
        self._last_observed_propensity_beta_extra_channels = 0
        self._last_observed_propensity_changed_catalyst_species = 0
        self._last_observed_propensity_catalyst_affected_channels = 0

    def _propensities_for_observed_cached(
        self,
        network: ReactionNetworkData,
        observed: np.ndarray,
        t: float,
        name: str,
    ) -> np.ndarray:
        values = np.asarray(observed, dtype=float)
        if values.ndim != 1:
            raise ValueError("observed state must be a 1D array")

        n_species = int(network.n_species)
        n_channels = int(network.n_channels)
        if values.shape[0] < n_species:
            raise ValueError("observed state has fewer species than the reaction network")

        self._last_observed_propensity_reused_beta_affected = False
        self._last_observed_propensity_beta_affected_species = 0
        self._last_observed_propensity_beta_affected_channels = 0
        self._last_observed_propensity_beta_extra_channels = 0
        self._last_observed_propensity_changed_catalyst_species = 0
        self._last_observed_propensity_catalyst_affected_channels = 0
        if not self._observed_propensity_cache_matches(network): # 第一次计算会走这个路初始化propensity
            return self._recompute_observed_propensity_cache(network, values, t, name)

        cached_state = self._observed_propensity_state
        cached_propensities = self._observed_propensity_cache
        if cached_state is None or cached_propensities is None:
            return self._recompute_observed_propensity_cache(network, values, t, name)

        changed_species = np.flatnonzero(values[:n_species] != cached_state) # 对比存下来的缓存和传入的state是否一致
        if changed_species.size == 0:
            self._last_observed_propensity_full_recompute = False
            self._last_observed_propensity_affected_updates = 0
            self._last_observed_propensity_update_path = "cache_hit"
            return cached_propensities

        affected = self._observed_propensity_affected_channels_from_beta_hint(network, changed_species)
        if affected is None:
            if not (
                hasattr(network, "affected_channels_for_species")
                and hasattr(network, "compute_propensities_for_channels")
            ):
                return self._recompute_observed_propensity_cache(network, values, t, name)

            try:
                affected = np.asarray(network.affected_channels_for_species(changed_species), dtype=np.int64)
            except Exception:
                return self._recompute_observed_propensity_cache(network, values, t, name)
        elif not hasattr(network, "compute_propensities_for_channels"):
            return self._recompute_observed_propensity_cache(network, values, t, name)

        if affected.size == 0:
            cached_state[:] = values[:n_species]
            self._last_observed_propensity_full_recompute = False
            self._last_observed_propensity_affected_updates = 0
            self._last_observed_propensity_update_path = "no_affected_channels"
            return cached_propensities
        if affected.size >= n_channels or affected.size > self._observed_propensity_local_update_limit(n_channels):
            return self._recompute_observed_propensity_cache(network, values, t, name)

        updated = network.compute_propensities_for_channels(affected, SystemState(t=float(t), x=values))
        cached_propensities[affected] = self._clean_propensities(updated, name) # 如果不同，更新缓存
        cached_state[:] = values[:n_species]
        self._last_observed_propensity_full_recompute = False
        self._last_observed_propensity_affected_updates = int(affected.size)
        self._last_observed_propensity_update_path = "local_update"
        return cached_propensities

    def _observed_propensity_local_update_limit(self, n_channels: int) -> int:
        return max(8, min(32, int(0.10 * int(n_channels))))

    def _observed_propensity_details(self) -> dict[str, Any]:
        return {
            "observed_propensity_update_path": str(self._last_observed_propensity_update_path),
            "observed_propensity_full_recompute": bool(self._last_observed_propensity_full_recompute),
            "observed_propensity_affected_updates": int(self._last_observed_propensity_affected_updates),
            "observed_propensity_reused_beta_affected": bool(
                self._last_observed_propensity_reused_beta_affected
            ),
            "observed_propensity_beta_affected_species": int(
                self._last_observed_propensity_beta_affected_species
            ),
            "observed_propensity_beta_affected_channels": int(
                self._last_observed_propensity_beta_affected_channels
            ),
            "observed_propensity_beta_extra_channels": int(
                self._last_observed_propensity_beta_extra_channels
            ),
            "observed_propensity_changed_catalyst_species": int(
                self._last_observed_propensity_changed_catalyst_species
            ),
            "observed_propensity_catalyst_affected_channels": int(
                self._last_observed_propensity_catalyst_affected_channels
            ),
        }

    def _observed_propensity_affected_channels_from_beta_hint(
        self,
        network: ReactionNetworkData,
        changed_species: np.ndarray,
    ) -> np.ndarray | None:
        if self.config.beta_compute_mode != "beta_compute_by_state_difference":
            return None
        if self._last_beta_reuse_network_id != id(network):
            return None
        species = np.asarray(changed_species, dtype=np.int64)
        if not np.array_equal(species, self._last_beta_reuse_changed_species):
            return None
        lookup = self._channel_beta_lookup(network)
        # 下面这些查找只服务于 propensity 局部更新，因此不能放在 beta 更新热路径里。
        beta_channels = np.asarray(self._last_beta_reuse_beta_channels, dtype=np.int64)
        dependency_extra_channels = _lookup_beta_propensity_extra_affected_channels(lookup, species)
        changed_catalyst_species = _lookup_beta_changed_catalyst_species(lookup, species)
        catalyst_channels = _lookup_beta_catalyst_affected_channels(lookup, changed_catalyst_species)
        extra_channels = _union_int_arrays(dependency_extra_channels, catalyst_channels)
        affected_channels = _union_int_arrays(beta_channels, extra_channels)

        self._last_beta_reuse_affected_channels = np.array(affected_channels, dtype=np.int64, copy=True)
        self._last_beta_reuse_extra_channels = np.array(extra_channels, dtype=np.int64, copy=True)
        self._last_beta_reuse_changed_catalyst_species = np.array(
            changed_catalyst_species,
            dtype=np.int64,
            copy=True,
        )
        self._last_beta_reuse_catalyst_channels = np.array(catalyst_channels, dtype=np.int64, copy=True)
        self._last_observed_propensity_reused_beta_affected = True
        self._last_observed_propensity_beta_affected_species = int(self._last_beta_reuse_affected_species.size)
        self._last_observed_propensity_beta_affected_channels = int(beta_channels.size)
        self._last_observed_propensity_beta_extra_channels = int(extra_channels.size)
        self._last_observed_propensity_changed_catalyst_species = int(changed_catalyst_species.size)
        self._last_observed_propensity_catalyst_affected_channels = int(catalyst_channels.size)
        return affected_channels

    def _observed_propensity_cache_matches(self, network: ReactionNetworkData) -> bool:
        return (
            self._observed_propensity_cache is not None
            and self._observed_propensity_state is not None
            and self._observed_propensity_network_id == id(network)
            and self._observed_propensity_cache.shape == (int(network.n_channels),)
            and self._observed_propensity_state.shape == (int(network.n_species),)
        )

    def _recompute_observed_propensity_cache(
        self,
        network: ReactionNetworkData,
        observed: np.ndarray,
        t: float,
        name: str,
    ) -> np.ndarray:
        propensities = self._propensities_for_x(network, observed, t)
        propensities = self._clean_propensities(propensities, name)
        self._observed_propensity_cache = propensities
        self._observed_propensity_state = np.array(observed[: int(network.n_species)], dtype=float, copy=True)
        self._observed_propensity_network_id = id(network)
        self._last_observed_propensity_full_recompute = True
        self._last_observed_propensity_affected_updates = 0
        self._last_observed_propensity_update_path = "full_recompute"
        self._last_observed_propensity_reused_beta_affected = False
        self._last_observed_propensity_beta_affected_species = 0
        self._last_observed_propensity_beta_affected_channels = 0
        self._last_observed_propensity_beta_extra_channels = 0
        self._last_observed_propensity_changed_catalyst_species = 0
        self._last_observed_propensity_catalyst_affected_channels = 0
        return propensities

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
            if self.config.adaptive_cle_dt:
                self._adaptive_dt_cle = self._clamp_cle_dt(self._reaction_interval_dt)
                self._adaptive_dt_macro = self._clamp_cle_dt(self._reaction_interval_dt)

    def _current_dt_cle(self) -> float:
        if self.config.adaptive_cle_dt:
            return self._get_adaptive_dt("cle")
        return self._base_dt_cle()

    def _current_dt_macro(self) -> float:
        if self.config.adaptive_cle_dt:
            return self._get_adaptive_dt("macro")
        return self._base_dt_macro()

    def _base_dt_cle(self) -> float:
        if self.config.use_reaction_interval_dt and self._reaction_interval_dt is not None:
            return self._reaction_interval_dt
        return self.config.dt_cle

    def _base_dt_macro(self) -> float:
        if self.config.use_reaction_interval_dt and self._reaction_interval_dt is not None:
            return self._reaction_interval_dt
        return self.config.effective_dt_macro

    def _get_adaptive_dt(self, kind: str) -> float:
        if kind == "macro":
            return self._clamp_cle_dt(self._adaptive_dt_macro)
        if kind == "cle":
            return self._clamp_cle_dt(self._adaptive_dt_cle)
        raise ValueError("kind must be 'cle' or 'macro'")

    def _set_adaptive_dt(self, kind: str, value: float) -> float:
        clamped = self._clamp_cle_dt(value)
        if kind == "macro":
            self._adaptive_dt_macro = clamped
            return clamped
        if kind == "cle":
            self._adaptive_dt_cle = clamped
            return clamped
        raise ValueError("kind must be 'cle' or 'macro'")

    def _shrink_adaptive_dt(self, kind: str, value: float) -> float:
        return self._set_adaptive_dt(kind, float(value) * self.config.cle_dt_shrink_factor)

    def _grow_adaptive_dt(self, kind: str, value: float) -> float:
        return self._set_adaptive_dt(kind, float(value) * self.config.cle_dt_growth_factor)

    def _clamp_cle_dt(self, value: float) -> float:
        dt = max(float(value), self.config.cle_dt_min)
        if self.config.cle_dt_max is not None:
            dt = min(dt, self.config.cle_dt_max)
        return dt
    
    def print_matrix_info(self):
        print(self._stoich_sparsity_profile)


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

    def invalidate_cache(self) -> None:
        super().invalidate_cache()
        if hasattr(self, "_nrm"):
            self._nrm.invalidate_cache()

    def step(self, state: SystemState, dt: float, context: StepperContext) -> StepResult:
        if dt <= 0.0:
            return StepResult(advanced_time=0.0, event_occurred=False, details={"mode": "nrm_blended_no_dt"})

        network = context.network
        self._maybe_update_reaction_interval_dt(network, state)
        x_float = self._float_nonnegative(state.x)
        observed = self._rounded_nonnegative(x_float)
        beta = self._channel_betas(network, observed)
        beta_min = float(np.min(beta)) if beta.size else 0.0
        beta_max = float(np.max(beta)) if beta.size else 0.0

        if beta_max <= self.config.beta_tol:
            return self._pure_cle_step(state, float(dt), context, beta, beta_min, beta_max)
        if beta_min >= 1.0 - self.config.beta_tol:
            return self._pure_nrm_step(state, float(dt), context, beta_min, beta_max, observed=observed)
        return self._mixed_step(state, float(dt), context, beta, beta_min, beta_max, observed=observed)

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
        *,
        observed: np.ndarray | None = None,
    ) -> StepResult:
        duration = min(self._current_dt_macro(), dt)
        state.x[:] = self._rounded_nonnegative(state.x) if observed is None else np.asarray(observed, dtype=float)
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
        *,
        observed: np.ndarray | None = None,
    ) -> StepResult:
        network = context.network
        rng = context.rng
        duration = min(self._current_dt_cle(), dt)
        start_time = float(state.t)
        end_time = start_time + duration
        x_work = self._float_nonnegative(state.x).copy()

        observed_initial = self._rounded_nonnegative(x_work) if observed is None else np.asarray(observed, dtype=float)
        if self.config.strict_int_for_CLE:
            base_initial = self._propensities_for_observed_cached(
                network,
                observed_initial,
                start_time,
                "initial mixed propensities",
            )
        else:
            base_initial = self._propensities_for_x(network, observed_initial, start_time)
            base_initial = self._clean_propensities(base_initial, "initial mixed propensities")
        initial_cle_base_propensities = base_initial if self.config.strict_int_for_CLE else None
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
            base_propensities=base_initial,
        )

        continuous_abs_total = np.zeros(network.n_channels, dtype=float)
        applied_event_ids: list[int] = []
        applied_event_times: list[float] = []
        invalid_jumps = 0
        n_clipped_total = 0
        n_low_count_rounded_total = 0
        cle_rejected_attempts_total = 0
        cle_min_dt_reached = False
        cle_stop_due_to_retry = False
        cle_dt_after = self._get_adaptive_dt("cle")
        current_time = start_time

        for event_time, channel_id in scheduled_events:
            event_t = float(event_time)
            segment_dt = max(event_t - current_time, 0.0)
            if segment_dt > 0.0:
                cle = self._adaptive_cle_increment(
                    network,
                    x_work,
                    beta,
                    segment_dt,
                    rng,
                    kind="cle",
                    grow_on_success=False,
                    base_propensities=initial_cle_base_propensities if current_time == start_time else None,
                )
                x_work = cle.x
                continuous_abs_total += cle.continuous_abs
                n_clipped_total += cle.n_clipped
                n_low_count_rounded_total += cle.n_low_count_rounded
                cle_rejected_attempts_total += cle.rejected_attempts
                cle_min_dt_reached = cle_min_dt_reached or cle.min_dt_reached
                cle_dt_after = cle.dt_after
                current_time += cle.dt
                if cle.dt < segment_dt - self.config.beta_tol or cle.rejected_attempts > 0:
                    cle_stop_due_to_retry = True
                    break
            applied = self._apply_jump_safely(network, x_work, int(channel_id))
            if applied:
                applied_event_ids.append(int(channel_id))
                applied_event_times.append(event_t)
            else:
                invalid_jumps += 1
            current_time = event_t

        if not cle_stop_due_to_retry:
            tail_dt = max(end_time - current_time, 0.0)
            if tail_dt > 0.0 or not scheduled_events:
                cle = self._adaptive_cle_increment(
                    network,
                    x_work,
                    beta,
                    tail_dt,
                    rng,
                    kind="cle",
                    grow_on_success=not scheduled_events,
                    base_propensities=initial_cle_base_propensities if not scheduled_events else None,
                )
                x_work = cle.x
                continuous_abs_total += cle.continuous_abs
                n_clipped_total += cle.n_clipped
                n_low_count_rounded_total += cle.n_low_count_rounded
                cle_rejected_attempts_total += cle.rejected_attempts
                cle_min_dt_reached = cle_min_dt_reached or cle.min_dt_reached
                cle_dt_after = cle.dt_after
                current_time += cle.dt
                if cle.dt < tail_dt - self.config.beta_tol or cle.rejected_attempts > 0:
                    cle_stop_due_to_retry = True

        state.x[:] = self._float_nonnegative(x_work)
        state.t = current_time
        state.step_count += 1
        state.event_count += len(applied_event_ids)
        self._nrm.invalidate_cache()

        self._last_n_clipped = int(n_clipped_total)
        self._last_n_low_count_rounded = int(n_low_count_rounded_total)
        self._last_total_cle_propensity = total_cle_initial
        self._last_continuous_channel_abs_increments = continuous_abs_total

        first_tau = None if not applied_event_times else float(applied_event_times[0] - start_time)
        first_channel = None if not applied_event_ids else int(applied_event_ids[0])
        advanced_time = max(float(current_time) - start_time, 0.0)
        return StepResult(
            advanced_time=advanced_time,
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
                "stepper_dt": advanced_time,
                "reaction_interval_dt": self._reaction_interval_dt,
                "n_scheduled_discrete_events": int(len(scheduled_events)),
                "n_applied_discrete_events": int(len(applied_event_ids)),
                "n_invalid_jump_skipped": int(invalid_jumps),
                "discrete_event_ids": list(applied_event_ids),
                "discrete_event_times": list(applied_event_times),
                "cle_requested_dt": float(duration),
                "cle_accepted_dt": float(advanced_time),
                "cle_rejected_attempts": int(cle_rejected_attempts_total),
                "cle_dt_after": float(cle_dt_after),
                "cle_dt_min_reached": bool(cle_min_dt_reached),
                "cle_adaptive_stop": bool(cle_stop_due_to_retry),
                "continuous_channel_abs_increments": continuous_abs_total.copy(),
                **self._observed_propensity_details(),
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
        *,
        base_propensities: np.ndarray | None = None,
    ) -> tuple[list[tuple[float, int]], dict[str, Any]]:
        if duration <= 0.0:
            return [], {"nrm_dependency_graph_used": False, "nrm_full_recompute": False, "nrm_affected_updates": 0}

        now = float(start_time)
        end_time = now + float(duration)
        x_jump = self._rounded_nonnegative(x).copy()
        jump_state = SystemState(t=now, x=x_jump)
        if base_propensities is None:
            base = network.compute_all_propensities(jump_state)
        else:
            base = np.asarray(base_propensities, dtype=float)
            if base.shape != (network.n_channels,):
                raise ValueError(f"base_propensities must have shape ({network.n_channels},)")
        propensities = self._clean_propensities(
            beta * base,
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


def _normalize_pdmp_discrete_event_method(value: str, *, legacy_use_heap: bool) -> str:
    method = str(value).strip().lower()
    if method == "auto":
        return "nrm_heap" if bool(legacy_use_heap) else "nrm_scan"
    aliases = {
        "heap": "nrm_heap",
        "nrm-heap": "nrm_heap",
        "nrm_heap": "nrm_heap",
        "nrm": "nrm_heap",
        "scan": "nrm_scan",
        "nrm-scan": "nrm_scan",
        "nrm_scan": "nrm_scan",
        "gillespie": "gillespie",
        "direct": "gillespie",
        "direct_ssa": "gillespie",
        "direct-ssa": "gillespie",
    }
    if method not in aliases:
        raise ValueError("discrete_event_method must be 'auto', 'nrm_heap', 'nrm_scan', or 'gillespie'")
    return aliases[method]


def _species_beta(x: float, i1: float, i2: float) -> float:
    value = float(x)
    if value <= float(i1):
        return 1.0
    if value >= float(i2):
        return 0.0
    return float((float(i2) - value) / (float(i2) - float(i1)))


def _species_beta_array(x: np.ndarray, i1: float, i2: float) -> np.ndarray: # 先算再clip，但向量化计算很快
    values = np.asarray(x, dtype=float)
    beta = (float(i2) - values) / (float(i2) - float(i1))
    return np.clip(beta, 0.0, 1.0)


def _build_species_to_channels_csr(n_species: int, species_by_channel: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    species_counts = np.zeros(int(n_species), dtype=np.int64)
    normalized: list[np.ndarray] = []
    for channel_species in species_by_channel:
        species = np.unique(np.asarray(channel_species, dtype=np.int64))
        if species.size and np.any((species < 0) | (species >= int(n_species))):
            raise ValueError("channel dependency references species outside network bounds")
        normalized.append(species)
        for sid in species:
            species_counts[int(sid)] += 1

    indptr = np.empty(int(n_species) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(species_counts, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    write_offsets = indptr[:-1].copy()
    for channel_id, species in enumerate(normalized):
        for sid in species:
            pos = int(write_offsets[int(sid)])
            indices[pos] = int(channel_id)
            write_offsets[int(sid)] += 1
    indptr.setflags(write=False)
    indices.setflags(write=False)
    return indptr, indices


def _build_channel_beta_lookup(network: ReactionNetworkData, mode: str) -> _ChannelBetaLookup: # 一个缓存表
    relevant_by_channel: list[list[int]] = []
    max_width = 0
    n_channels = int(network.n_channels)
    n_species = int(network.n_species)

    # 先构造正向关系：每个 channel 计算 beta 时需要看哪些 species。
    for channel_id in range(n_channels):
        if network.get_channel_block(channel_id) == ChannelBlock.INFLOW:
            relevant_species: list[int] = []
        else:
            relevant_species = _channel_relevant_species(network, channel_id, mode)
        for sid in relevant_species:
            if sid < 0 or sid >= n_species:
                raise ValueError(f"channel {channel_id} references species outside network bounds: {sid}")
        relevant_by_channel.append(relevant_species)
        max_width = max(max_width, len(relevant_species))

    # padded dense 表方便 _channel_betas 用一次 gather/max 得到全量 channel beta。
    # mask 区分真实 species id 和 padding 的 0。
    species = np.zeros((n_channels, max_width), dtype=np.int64)
    mask = np.zeros((n_channels, max_width), dtype=bool)
    for channel_id, relevant_species in enumerate(relevant_by_channel):
        if not relevant_species:
            continue
        width = len(relevant_species)
        species[channel_id, :width] = np.asarray(relevant_species, dtype=np.int64)
        mask[channel_id, :width] = True

    # beta reverse mapping：species -> beta 计算依赖该 species 的 channels。
    species_to_channels_indptr, species_to_channels_indices = _build_species_to_channels_csr(
        n_species,
        relevant_by_channel,
    )

    # propensity-extra reverse mapping：species -> beta relevant species 未覆盖、但 propensity 依赖该 species 的 channels。
    # 其中最重要的是 catalyst ids；另外也覆盖 finite-capacity inflow target、以及非默认 beta_species_mode 漏掉的 reactants。
    dependency_by_channel: list[np.ndarray] = []
    if getattr(network, "dependency_indices_dirty", False):
        rebuild = getattr(network, "rebuild_dependency_indices", None)
        if callable(rebuild):
            rebuild()
    channel_to_species = getattr(network, "channel_to_species", None)
    if isinstance(channel_to_species, list) and len(channel_to_species) == n_channels:
        for channel_id in range(n_channels):
            dependency_by_channel.append(np.asarray(channel_to_species[channel_id], dtype=np.int64))
    else:
        for relevant_species in relevant_by_channel:
            dependency_by_channel.append(np.asarray(relevant_species, dtype=np.int64))

    extra_by_channel: list[np.ndarray] = []
    for channel_id, dependency_species in enumerate(dependency_by_channel):
        relevant_set = set(int(sid) for sid in relevant_by_channel[channel_id])
        extra = [int(sid) for sid in np.asarray(dependency_species, dtype=np.int64) if int(sid) not in relevant_set]
        extra_by_channel.append(np.asarray(extra, dtype=np.int64))
    propensity_extra_species_to_channels_indptr, propensity_extra_species_to_channels_indices = _build_species_to_channels_csr(
        n_species,
        extra_by_channel,
    )

    catalyst_by_channel: list[np.ndarray] = []
    channel_to_catalysts = getattr(network, "channel_to_catalysts", None)
    if isinstance(channel_to_catalysts, list) and len(channel_to_catalysts) == n_channels:
        for channel_id in range(n_channels):
            catalyst_by_channel.append(np.asarray(channel_to_catalysts[channel_id], dtype=np.int64))
    else:
        for channel_id in range(n_channels):
            getter = getattr(network, "get_channel_catalysts", None)
            if callable(getter):
                catalyst_by_channel.append(np.asarray(getter(channel_id), dtype=np.int64))
            else:
                catalyst_by_channel.append(np.empty(0, dtype=np.int64))
    catalyst_species_to_channels_indptr, catalyst_species_to_channels_indices = _build_species_to_channels_csr(
        n_species,
        catalyst_by_channel,
    )
    catalyst_species_mask = np.zeros(n_species, dtype=bool)
    if catalyst_species_to_channels_indices.size:
        for catalyst_species in catalyst_by_channel:
            catalyst_species_mask[np.asarray(catalyst_species, dtype=np.int64)] = True
    catalyst_species_mask.setflags(write=False)

    # lookup 是只读拓扑缓存，后续 step 只读它，不应被运行时路径改写。
    species.setflags(write=False)
    mask.setflags(write=False)
    return _ChannelBetaLookup(
        relevant_species=species, # 每个 channel 在计算 beta 时需要参考的 species id 列表, 被用来找一个反应中最大的 beta
        relevant_mask=mask,
        species_to_channels_indptr=species_to_channels_indptr,
        species_to_channels_indices=species_to_channels_indices,
        propensity_extra_species_to_channels_indptr=propensity_extra_species_to_channels_indptr,
        propensity_extra_species_to_channels_indices=propensity_extra_species_to_channels_indices,
        catalyst_species_mask=catalyst_species_mask,
        catalyst_species_to_channels_indptr=catalyst_species_to_channels_indptr,
        catalyst_species_to_channels_indices=catalyst_species_to_channels_indices,
        n_channels=n_channels,
        n_species=n_species,
        mode=str(mode),
    )


def _lookup_beta_affected_channels(lookup: _ChannelBetaLookup, species_ids: np.ndarray) -> np.ndarray:
    # 根据 reverse mapping 把 changed species 合并成 affected beta channel 集合。
    species = np.asarray(species_ids, dtype=np.int64)
    if species.size == 0:
        return np.empty(0, dtype=np.int64)

    arrays: list[np.ndarray] = []
    for sid in np.unique(species):
        index = int(sid)
        if index < 0 or index >= lookup.n_species:
            raise IndexError("species_ids contain out-of-range values")
        start = int(lookup.species_to_channels_indptr[index])
        end = int(lookup.species_to_channels_indptr[index + 1])
        if end > start:
            arrays.append(lookup.species_to_channels_indices[start:end])

    if not arrays:
        return np.empty(0, dtype=np.int64)
    if len(arrays) == 1:
        return np.array(arrays[0], dtype=np.int64, copy=True)
    return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)


def _lookup_beta_propensity_extra_affected_channels(lookup: _ChannelBetaLookup, species_ids: np.ndarray) -> np.ndarray:
    # 根据 propensity-extra reverse mapping，把 beta 没覆盖的 propensity affected channels 补出来。
    species = np.asarray(species_ids, dtype=np.int64)
    if species.size == 0:
        return np.empty(0, dtype=np.int64)

    arrays: list[np.ndarray] = []
    for sid in np.unique(species):
        index = int(sid)
        if index < 0 or index >= lookup.n_species:
            raise IndexError("species_ids contain out-of-range values")
        start = int(lookup.propensity_extra_species_to_channels_indptr[index])
        end = int(lookup.propensity_extra_species_to_channels_indptr[index + 1])
        if end > start:
            arrays.append(lookup.propensity_extra_species_to_channels_indices[start:end])

    if not arrays:
        return np.empty(0, dtype=np.int64)
    if len(arrays) == 1:
        return np.array(arrays[0], dtype=np.int64, copy=True)
    return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)


def _lookup_beta_changed_catalyst_species(lookup: _ChannelBetaLookup, species_ids: np.ndarray) -> np.ndarray:
    species = np.unique(np.asarray(species_ids, dtype=np.int64))
    if species.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any((species < 0) | (species >= lookup.n_species)):
        raise IndexError("species_ids contain out-of-range values")
    return species[lookup.catalyst_species_mask[species]].astype(np.int64, copy=False)


def _lookup_beta_catalyst_affected_channels(lookup: _ChannelBetaLookup, species_ids: np.ndarray) -> np.ndarray:
    # 只用 changed catalyst ids 查 catalytic propensity affected channels，供 diagnostics 和复用验证。
    species = np.asarray(species_ids, dtype=np.int64)
    if species.size == 0:
        return np.empty(0, dtype=np.int64)

    arrays: list[np.ndarray] = []
    for sid in np.unique(species):
        index = int(sid)
        if index < 0 or index >= lookup.n_species:
            raise IndexError("species_ids contain out-of-range values")
        start = int(lookup.catalyst_species_to_channels_indptr[index])
        end = int(lookup.catalyst_species_to_channels_indptr[index + 1])
        if end > start:
            arrays.append(lookup.catalyst_species_to_channels_indices[start:end])

    if not arrays:
        return np.empty(0, dtype=np.int64)
    if len(arrays) == 1:
        return np.array(arrays[0], dtype=np.int64, copy=True)
    return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)


def _union_int_arrays(*arrays: np.ndarray) -> np.ndarray:
    nonempty = [np.asarray(arr, dtype=np.int64) for arr in arrays if np.asarray(arr, dtype=np.int64).size]
    if not nonempty:
        return np.empty(0, dtype=np.int64)
    if len(nonempty) == 1:
        return np.array(nonempty[0], dtype=np.int64, copy=True)
    return np.unique(np.concatenate(nonempty)).astype(np.int64, copy=False)


def _lookup_beta_relevant_species_for_channels(lookup: _ChannelBetaLookup, channel_ids: np.ndarray) -> np.ndarray:
    # 从 affected channels 反查这些 channel 计算 beta 时真正依赖的 species。
    channels = np.asarray(channel_ids, dtype=np.int64)
    if channels.size == 0 or lookup.relevant_species.shape[1] == 0:
        return np.empty(0, dtype=np.int64)
    if np.any((channels < 0) | (channels >= lookup.n_channels)):
        raise IndexError("channel_ids contain out-of-range values")

    relevant_species = lookup.relevant_species[channels]
    relevant_mask = lookup.relevant_mask[channels]
    species = relevant_species[relevant_mask]
    if species.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(species).astype(np.int64, copy=False)


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


def _sample_channel( # 对离散通道抽发生的反应
    channels: np.ndarray,
    propensities: np.ndarray,
    total: float,
    rng: np.random.Generator,
) -> int:
    threshold = float(rng.random() * total)
    # cumulative = 0.0
    # chosen = int(channels[-1])
    # for channel_id, propensity in zip(channels, propensities):
    #     cumulative += float(propensity)
    #     if cumulative >= threshold:
    #         chosen = int(channel_id)
    #         break
    selected_channels = np.asarray(channels, dtype=np.int64)
    selected_prop = propensities[selected_channels]
    cum = np.cumsum(selected_prop)
    chosen_idx = np.searchsorted(cum, threshold)
    chosen = selected_channels[chosen_idx]
    return chosen