from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from polymer_sim.core.enums import ChannelBlock
from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState


DEFAULT_STANDARD_ZERO_ORDER_INFLOW = True


@dataclass(slots=True)
class ElementaryExpansionConfig:
    """Configuration for expanding the structured polymer network to elementary mass-action channels.

    This view is intended for PDMP scaling/partition algorithms that assume
    elementary zero-, first-, and second-order mass-action reactions.  It does
    not replace the hot polymer-rule network used by SSA/CLE steppers.
    """

    standard_zero_order_inflow: bool = DEFAULT_STANDARD_ZERO_ORDER_INFLOW
    include_uncatalyzed_channels: bool = True
    expand_catalysis: bool = True
    catalyst_binding_rate: float = 1.0
    catalyst_binding_rate_per_strength: float = 0.0
    catalyst_unbinding_rate: float = 1.0
    catalytic_turnover_rate: float | None = None
    catalytic_turnover_scale: float = 1.0
    complex_initial_count: float = 0.0
    complex_name_prefix: str = "complex"

    def __post_init__(self) -> None:
        self.standard_zero_order_inflow = bool(self.standard_zero_order_inflow)
        self.include_uncatalyzed_channels = bool(self.include_uncatalyzed_channels)
        self.expand_catalysis = bool(self.expand_catalysis)
        self.catalyst_binding_rate = float(self.catalyst_binding_rate)
        self.catalyst_binding_rate_per_strength = float(self.catalyst_binding_rate_per_strength)
        self.catalyst_unbinding_rate = float(self.catalyst_unbinding_rate)
        self.catalytic_turnover_rate = (
            None if self.catalytic_turnover_rate is None else float(self.catalytic_turnover_rate)
        )
        self.catalytic_turnover_scale = float(self.catalytic_turnover_scale)
        self.complex_initial_count = float(self.complex_initial_count)
        self.complex_name_prefix = str(self.complex_name_prefix)
        if self.catalyst_binding_rate < 0.0:
            raise ValueError("catalyst_binding_rate must be >= 0")
        if self.catalyst_binding_rate_per_strength < 0.0:
            raise ValueError("catalyst_binding_rate_per_strength must be >= 0")
        if self.catalyst_unbinding_rate < 0.0:
            raise ValueError("catalyst_unbinding_rate must be >= 0")
        if self.catalytic_turnover_rate is not None and self.catalytic_turnover_rate < 0.0:
            raise ValueError("catalytic_turnover_rate must be >= 0 when provided")
        if self.catalytic_turnover_scale < 0.0:
            raise ValueError("catalytic_turnover_scale must be >= 0")
        if self.complex_initial_count < 0.0:
            raise ValueError("complex_initial_count must be >= 0")


@dataclass(slots=True)
class ElementaryMassActionNetwork:
    """Elementary mass-action network generated from a polymer-rule network.

    Reactions are stored as dense stoichiometric arrays for clarity:
    - ``nu_minus[r, i]``: reactant stoichiometry.
    - ``nu_plus[r, i]``: product stoichiometry.
    - ``nu[r, i] = nu_plus - nu_minus``.

    The supported propensity laws are exactly the elementary cases used by the
    PDMP scaling logic: zero-, first-, and second-order mass-action channels.
    """

    species_names: list[str]
    name_to_idx: dict[str, int]
    x0: np.ndarray
    nu_minus: np.ndarray
    nu_plus: np.ndarray
    rate_constants: np.ndarray
    reaction_labels: list[dict[str, object]] = field(default_factory=list)
    source_network: ReactionNetworkData | None = None
    source_to_elementary_channels: dict[int, list[int]] = field(default_factory=dict)
    polymer_species_count: int = 0
    dependency_indices_dirty: bool = False
    channel_to_species: list[np.ndarray] = field(default_factory=list)
    species_to_channels: list[np.ndarray] = field(default_factory=list)
    species_to_channels_indptr: np.ndarray = field(init=False, repr=False)
    species_to_channels_indices: np.ndarray = field(init=False, repr=False)
    reaction_order: np.ndarray = field(init=False, repr=False)
    reactant1: np.ndarray = field(init=False, repr=False)
    reactant2: np.ndarray = field(init=False, repr=False)
    homo_second_order: np.ndarray = field(init=False, repr=False)
    zero_order_channels: np.ndarray = field(init=False, repr=False)
    first_order_channels: np.ndarray = field(init=False, repr=False)
    second_order_channels: np.ndarray = field(init=False, repr=False)
    all_channels: np.ndarray = field(init=False, repr=False)
    _nu_cache: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        self.nu_minus = np.asarray(self.nu_minus, dtype=float)
        self.nu_plus = np.asarray(self.nu_plus, dtype=float)
        self.rate_constants = np.asarray(self.rate_constants, dtype=float)
        if self.x0.shape != (len(self.species_names),):
            raise ValueError("x0 must have shape (n_species,)")
        if self.nu_minus.shape != self.nu_plus.shape:
            raise ValueError("nu_minus and nu_plus must have the same shape")
        if self.nu_minus.ndim != 2:
            raise ValueError("nu_minus must have shape (n_channels, n_species)")
        if self.nu_minus.shape[1] != len(self.species_names):
            raise ValueError("stoichiometry species dimension does not match species_names")
        if self.rate_constants.shape != (self.nu_minus.shape[0],):
            raise ValueError("rate_constants must have shape (n_channels,)")
        if not self.reaction_labels:
            self.reaction_labels = [
                {"channel_id": int(channel_id), "block_type": "ELEMENTARY"}
                for channel_id in range(self.n_channels)
            ]
        if len(self.reaction_labels) != self.n_channels:
            raise ValueError("reaction_labels length must match n_channels")
        if self.polymer_species_count <= 0:
            self.polymer_species_count = len(self.species_names)
        self._nu_cache = self.nu_plus - self.nu_minus
        self._nu_cache.setflags(write=False)
        self._precompute_propensity_terms()
        self.rebuild_dependency_indices()

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    @property
    def n_channels(self) -> int:
        return int(self.rate_constants.shape[0])

    @property
    def nu(self) -> np.ndarray:
        return self._nu_cache

    def species_idx(self, name: str) -> int:
        return self.name_to_idx[name]

    def get_channel_block(self, channel_id: int):
        self._check_channel(channel_id)
        return self.reaction_labels[int(channel_id)].get("block_type", "ELEMENTARY")

    def get_channel_block_name(self, channel_id: int) -> str:
        return str(self.get_channel_block(channel_id))

    def get_channel_local_id(self, channel_id: int) -> int:
        self._check_channel(channel_id)
        return int(channel_id)

    def get_channel_reactants(self, channel_id: int) -> tuple[int, ...]:
        self._check_channel(channel_id)
        cid = int(channel_id)
        order = int(self.reaction_order[cid])
        if order == 0:
            return ()
        if order == 1:
            return (int(self.reactant1[cid]),)
        return (int(self.reactant1[cid]), int(self.reactant2[cid]))

    def get_channel_products(self, channel_id: int) -> tuple[int, ...]:
        self._check_channel(channel_id)
        row = self.nu_plus[int(channel_id)]
        values: list[int] = []
        for sid, count in enumerate(row):
            values.extend([int(sid)] * int(round(float(count))))
        return tuple(values)

    def describe_channel(self, channel_id: int) -> dict[str, object]:
        self._check_channel(channel_id)
        label = dict(self.reaction_labels[int(channel_id)])
        label.setdefault("channel_id", int(channel_id))
        label.setdefault("block_type", "ELEMENTARY")
        label["reactants"] = self.get_channel_reactants(channel_id)
        label["products"] = self.get_channel_products(channel_id)
        label["rate_constant"] = float(self.rate_constants[int(channel_id)])
        return label

    def compute_base_propensity(self, channel_id: int, state: SystemState) -> float:
        return self.compute_propensity(channel_id, state)

    def compute_propensity(self, channel_id: int, state: SystemState) -> float:
        self._check_channel(channel_id)
        cid = int(channel_id)
        rate = max(float(self.rate_constants[cid]), 0.0)
        if rate <= 0.0:
            return 0.0
        x = np.asarray(state.x, dtype=float)
        order = int(self.reaction_order[cid])
        if order == 0:
            return rate
        sid1 = int(self.reactant1[cid])
        count1 = max(float(x[sid1]), 0.0)
        if order == 1:
            return max(float(rate * count1), 0.0)
        sid2 = int(self.reactant2[cid])
        if bool(self.homo_second_order[cid]):
            value = rate * 0.5 * count1 * max(count1 - 1.0, 0.0)
        else:
            value = rate * count1 * max(float(x[sid2]), 0.0)
        return max(float(value), 0.0)

    def compute_all_propensities(self, state: SystemState, out: np.ndarray | None = None) -> np.ndarray:
        values = np.empty(self.n_channels, dtype=float) if out is None else out
        if values.shape != (self.n_channels,):
            raise ValueError(f"out must have shape ({self.n_channels},)")
        x = np.asarray(state.x, dtype=float)
        return self._compute_propensity_array(self.all_channels, x, values)

    def compute_propensities_for_channels(
        self,
        channel_ids: np.ndarray | Sequence[int],
        state: SystemState,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        channels = np.asarray(channel_ids, dtype=np.int64)
        if channels.ndim != 1:
            raise ValueError("channel_ids must be a 1D array")
        values = np.empty(channels.shape, dtype=float) if out is None else out
        if values.shape != channels.shape:
            raise ValueError(f"out must have shape {channels.shape}")
        if channels.size == 0:
            return values
        if np.any((channels < 0) | (channels >= self.n_channels)):
            raise IndexError("channel_ids contain out-of-range values")
        x = np.asarray(state.x, dtype=float)
        return self._compute_propensity_array(channels, x, values)

    def apply_channel_update(self, state: SystemState, channel_id: int) -> None:
        self.apply_channel_delta(state.x, channel_id, 1.0)

    def apply_channel_delta(self, x: np.ndarray, channel_id: int, amount: float) -> None:
        self._check_channel(channel_id)
        x[:] = np.asarray(x, dtype=float) + float(amount) * self.nu[int(channel_id)]

    def get_channel_changed_species(self, channel_id: int) -> np.ndarray:
        self._check_channel(channel_id)
        return np.flatnonzero(self.nu[int(channel_id)] != 0.0).astype(np.int64, copy=False)

    def affected_channels_for_species(self, species_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        species = np.asarray(species_ids, dtype=np.int64)
        if species.size == 0:
            return np.empty(0, dtype=np.int64)
        if self.dependency_indices_dirty or len(self.species_to_channels) != self.n_species:
            self.rebuild_dependency_indices()
        arrays = [self.species_to_channels[int(sid)] for sid in np.unique(species)]
        arrays = [arr for arr in arrays if arr.size]
        if not arrays:
            return np.empty(0, dtype=np.int64)
        if len(arrays) == 1:
            return np.array(arrays[0], dtype=np.int64, copy=True)
        return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)

    def rebuild_dependency_indices(self) -> None:
        channel_to_species: list[np.ndarray] = []
        reverse: list[set[int]] = [set() for _ in range(self.n_species)]
        for channel_id in range(self.n_channels):
            deps = np.flatnonzero(self.nu_minus[channel_id] > 0.0).astype(np.int64, copy=False)
            channel_to_species.append(deps)
            for sid in deps:
                reverse[int(sid)].add(int(channel_id))
        self.channel_to_species = channel_to_species
        self.species_to_channels = [np.asarray(sorted(values), dtype=np.int64) for values in reverse]
        self._rebuild_species_to_channels_csr()
        self.dependency_indices_dirty = False

    def _rebuild_species_to_channels_csr(self) -> None:
        """Build a CSR view of ``species_to_channels`` for compiled kernels."""

        indptr = np.zeros(self.n_species + 1, dtype=np.int64)
        total = 0
        for sid, channels in enumerate(self.species_to_channels):
            total += int(np.asarray(channels, dtype=np.int64).size)
            indptr[sid + 1] = total
        indices = np.empty(total, dtype=np.int64)
        offset = 0
        for channels in self.species_to_channels:
            arr = np.asarray(channels, dtype=np.int64)
            end = offset + int(arr.size)
            if arr.size:
                indices[offset:end] = arr
            offset = end
        indptr.setflags(write=False)
        indices.setflags(write=False)
        self.species_to_channels_indptr = indptr
        self.species_to_channels_indices = indices

    def _check_channel(self, channel_id: int) -> None:
        cid = int(channel_id)
        if cid < 0 or cid >= self.n_channels:
            raise IndexError(f"channel_id out of range: {cid}")

    def _precompute_propensity_terms(self) -> None:
        """Cache elementary reactant structure for the propensity hot path.

        ``nu_minus`` is kept as a dense float stoichiometric matrix because the
        PDMP partition code needs matrix operations on it.  Propensity
        evaluation, however, should not repeatedly parse each dense row.  This
        method converts every elementary channel once into:
        - total reactant order: 0, 1, or 2;
        - first and optional second reactant species id;
        - whether the second-order reaction is X + X.

        After this precomputation no hot propensity path needs ``round(...)`` or
        an O(n_species) scan per channel.
        """

        rounded = np.rint(self.nu_minus)
        if not np.allclose(self.nu_minus, rounded, rtol=0.0, atol=1e-12):
            raise ValueError("elementary reactant stoichiometry must be integer-valued")
        if np.any(rounded < 0.0):
            raise ValueError("elementary reactant stoichiometry must be nonnegative")

        n_channels = self.n_channels
        order = np.zeros(n_channels, dtype=np.int8)
        reactant1 = np.full(n_channels, -1, dtype=np.int64)
        reactant2 = np.full(n_channels, -1, dtype=np.int64)
        homo = np.zeros(n_channels, dtype=bool)

        for channel_id in range(n_channels):
            row = rounded[channel_id]
            ids = np.flatnonzero(row > 0.0)
            counts = row[ids].astype(np.int64, copy=False)
            total = int(np.sum(counts))
            if total > 2:
                raise ValueError("elementary network only supports total reactant order <= 2")
            order[channel_id] = total
            if total == 0:
                continue
            if total == 1:
                if ids.size != 1 or int(counts[0]) != 1:
                    raise ValueError("invalid first-order elementary reaction")
                reactant1[channel_id] = int(ids[0])
                continue
            if ids.size == 1:
                if int(counts[0]) != 2:
                    raise ValueError("invalid same-species second-order reaction")
                reactant1[channel_id] = int(ids[0])
                reactant2[channel_id] = int(ids[0])
                homo[channel_id] = True
            elif ids.size == 2:
                if not np.all(counts == 1):
                    raise ValueError("invalid two-species second-order reaction")
                reactant1[channel_id] = int(ids[0])
                reactant2[channel_id] = int(ids[1])
            else:
                raise ValueError("invalid second-order elementary reaction")

        self.reaction_order = order
        self.reactant1 = reactant1
        self.reactant2 = reactant2
        self.homo_second_order = homo
        self.all_channels = np.arange(n_channels, dtype=np.int64)
        self.zero_order_channels = np.flatnonzero(order == 0).astype(np.int64, copy=False)
        self.first_order_channels = np.flatnonzero(order == 1).astype(np.int64, copy=False)
        self.second_order_channels = np.flatnonzero(order == 2).astype(np.int64, copy=False)

    def _compute_propensity_array(self, channels: np.ndarray, x: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Vectorized elementary mass-action propensity evaluation."""

        values = out
        values[...] = 0.0
        if channels.size == 0:
            return values

        # 局部 propensity 更新通常只涉及少量受影响通道。此时布尔 mask、
        # 临时数组和多次 np.any 的分配成本会超过公式本身，因此用预计算好的
        # order/reactant 数组走一条无额外数组分配的小循环。全量计算仍保留
        # NumPy 向量化路径。
        if channels.size <= 128:
            return self._compute_propensity_array_loop(channels, x, values)

        rates = np.maximum(self.rate_constants[channels], 0.0)
        orders = self.reaction_order[channels]

        zero = orders == 0
        if np.any(zero):
            values[zero] = rates[zero]

        first = orders == 1
        if np.any(first):
            first_channels = channels[first]
            values[first] = rates[first] * np.maximum(x[self.reactant1[first_channels]], 0.0)

        second = orders == 2
        if np.any(second):
            second_channels = channels[second]
            second_rates = rates[second]
            sid1 = self.reactant1[second_channels]
            sid2 = self.reactant2[second_channels]
            x1 = np.maximum(x[sid1], 0.0)
            same = self.homo_second_order[second_channels]
            second_values = np.empty(second_channels.shape, dtype=float)
            if np.any(same):
                second_values[same] = second_rates[same] * 0.5 * x1[same] * np.maximum(x1[same] - 1.0, 0.0)
            if np.any(~same):
                second_values[~same] = second_rates[~same] * x1[~same] * np.maximum(x[sid2[~same]], 0.0)
            values[second] = second_values

        np.maximum(values, 0.0, out=values)
        return values

    def _compute_propensity_array_loop(self, channels: np.ndarray, x: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Small-subset elementary propensity kernel.

        该函数依赖 ``_precompute_propensity_terms`` 在建网阶段得到的
        ``reaction_order/reactant1/reactant2/homo_second_order``，避免在 PDMP
        局部刷新热路径里反复扫描 ``nu_minus`` 或创建多个布尔 mask。
        """

        rates = self.rate_constants
        order = self.reaction_order
        reactant1 = self.reactant1
        reactant2 = self.reactant2
        homo = self.homo_second_order
        values = out
        for pos, channel_id in enumerate(channels):
            cid = int(channel_id)
            rate = float(rates[cid])
            if rate <= 0.0:
                values[pos] = 0.0
                continue
            reaction_order = int(order[cid])
            if reaction_order == 0:
                values[pos] = rate
                continue
            count1 = max(float(x[int(reactant1[cid])]), 0.0)
            if reaction_order == 1:
                values[pos] = rate * count1
                continue
            if bool(homo[cid]):
                values[pos] = rate * 0.5 * count1 * max(count1 - 1.0, 0.0)
            else:
                values[pos] = rate * count1 * max(float(x[int(reactant2[cid])]), 0.0)
        return values


class _ElementaryBuilder:
    def __init__(self, network: ReactionNetworkData, config: ElementaryExpansionConfig):
        self.network = network
        self.config = config
        self.species_names = list(network.species_names)
        self.name_to_idx = dict(network.name_to_idx)
        self.x0 = [float(value) for value in np.asarray(network.x0, dtype=float)]
        self.nu_minus_rows: list[np.ndarray] = []
        self.nu_plus_rows: list[np.ndarray] = []
        self.rates: list[float] = []
        self.labels: list[dict[str, object]] = []
        self.source_to_elementary: dict[int, list[int]] = {}
        self._complex_by_pair: dict[tuple[int, int], int] = {}
        self._bound_complex_pairs: set[tuple[int, int]] = set()
        self._binding_rate_by_pair: dict[tuple[int, int], float] = {}

    def build(self) -> ElementaryMassActionNetwork:
        for channel_id in range(self.network.n_channels):
            if self.config.include_uncatalyzed_channels:
                self._add_uncatalyzed_channel(channel_id)
            if self.config.expand_catalysis:
                self._add_catalyzed_channels(channel_id)
        return ElementaryMassActionNetwork(
            species_names=list(self.species_names),
            name_to_idx=dict(self.name_to_idx),
            x0=np.asarray(self.x0, dtype=float),
            nu_minus=np.vstack(self.nu_minus_rows) if self.nu_minus_rows else np.empty((0, len(self.species_names))),
            nu_plus=np.vstack(self.nu_plus_rows) if self.nu_plus_rows else np.empty((0, len(self.species_names))),
            rate_constants=np.asarray(self.rates, dtype=float),
            reaction_labels=list(self.labels),
            source_network=self.network,
            source_to_elementary_channels={int(k): list(v) for k, v in self.source_to_elementary.items()},
            polymer_species_count=self.network.n_species,
        )

    def _add_uncatalyzed_channel(self, source_channel: int) -> None:
        block = self.network.get_channel_block(source_channel)
        if block == ChannelBlock.INFLOW and not self.config.standard_zero_order_inflow:
            raise ValueError(
                "ElementaryMassActionNetwork requires standard_zero_order_inflow=True for inflow channels. "
                "Hill-capped inflow is not an elementary mass-action reaction."
            )
        reactants = self.network.get_channel_reactants(source_channel)
        products = self.network.get_channel_products(source_channel)
        rate = self._base_rate(source_channel)
        self._add_reaction(
            reactants,
            products,
            rate,
            source_channel=source_channel,
            role="uncatalyzed",
            block_type=self.network.get_channel_block_name(source_channel),
        )

    def _add_catalyzed_channels(self, source_channel: int) -> None:
        if self.network.get_channel_block(source_channel) == ChannelBlock.INFLOW:
            return
        catalysts = self.network.get_channel_catalysts(source_channel)
        if catalysts.size == 0:
            return
        row = self.network._cat_row(source_channel)
        for catalyst_sid in catalysts:
            strength = float(row[int(catalyst_sid)])
            if strength <= 0.0:
                continue
            self._add_complex_catalytic_mechanism(source_channel, int(catalyst_sid), strength)

    def _add_complex_catalytic_mechanism(self, source_channel: int, catalyst_sid: int, strength: float) -> None:
        reactants = self.network.get_channel_reactants(source_channel)
        products = self.network.get_channel_products(source_channel)
        if not reactants:
            return
        substrate_sid = int(self.network.get_channel_main_species(source_channel))
        complex_sid = self._complex_species(catalyst_sid, substrate_sid)
        binding_rate = self._binding_rate(strength)
        self._add_binding_pair(catalyst_sid, substrate_sid, complex_sid, binding_rate)

        residual_reactants = [int(sid) for sid in reactants]
        try:
            residual_reactants.remove(substrate_sid)
        except ValueError:
            residual_reactants = []
        turnover_reactants = tuple([complex_sid, *residual_reactants])
        if len(turnover_reactants) > 2:
            # The expansion is only elementary for zero-, one-, and two-reactant
            # channels.  This protects Algorithm 3 from non-elementary rows.
            return
        turnover_products = tuple([catalyst_sid, *products])
        rate = self._catalytic_turnover_rate(source_channel, strength)
        self._add_reaction(
            turnover_reactants,
            turnover_products,
            rate,
            source_channel=source_channel,
            role="catalytic_turnover",
            catalyst_sid=catalyst_sid,
            complex_sid=complex_sid,
            block_type=f"{self.network.get_channel_block_name(source_channel)}_CAT",
        )

    def _add_binding_pair(
        self,
        catalyst_sid: int,
        substrate_sid: int,
        complex_sid: int,
        binding_rate: float,
    ) -> None:
        key = (int(catalyst_sid), int(substrate_sid))
        if key in self._bound_complex_pairs:
            previous = float(self._binding_rate_by_pair[key])
            if not np.isclose(previous, float(binding_rate), rtol=1e-12, atol=1e-12):
                raise ValueError(
                    "conflicting catalyst-substrate binding rates for one complex; "
                    "assign one gamma per catalyst-substrate pair"
                )
            return
        self._bound_complex_pairs.add(key)
        self._binding_rate_by_pair[key] = float(binding_rate)
        self._add_reaction(
            (catalyst_sid, substrate_sid),
            (complex_sid,),
            float(binding_rate),
            role="complex_binding",
            catalyst_sid=catalyst_sid,
            substrate_sid=substrate_sid,
            complex_sid=complex_sid,
            block_type="COMPLEX_BIND",
        )
        self._add_reaction(
            (complex_sid,),
            (catalyst_sid, substrate_sid),
            self.config.catalyst_unbinding_rate,
            role="complex_unbinding",
            catalyst_sid=catalyst_sid,
            substrate_sid=substrate_sid,
            complex_sid=complex_sid,
            block_type="COMPLEX_UNBIND",
        )

    def _complex_species(self, catalyst_sid: int, substrate_sid: int) -> int:
        key = (int(catalyst_sid), int(substrate_sid))
        existing = self._complex_by_pair.get(key)
        if existing is not None:
            return existing
        catalyst = self.network.species_names[int(catalyst_sid)]
        substrate = self.network.species_names[int(substrate_sid)]
        name = f"{self.config.complex_name_prefix}:{catalyst}|{substrate}"
        sid = len(self.species_names)
        self.species_names.append(name)
        self.name_to_idx[name] = sid
        self.x0.append(float(self.config.complex_initial_count))
        self._complex_by_pair[key] = sid
        return sid

    def _add_reaction(
        self,
        reactants: Sequence[int],
        products: Sequence[int],
        rate: float,
        *,
        source_channel: int | None = None,
        role: str,
        block_type: str,
        **metadata: object,
    ) -> int:
        if rate < 0.0:
            raise ValueError("elementary reaction rate must be >= 0")
        n_species = len(self.species_names)
        minus = np.zeros(n_species, dtype=float)
        plus = np.zeros(n_species, dtype=float)
        for sid in reactants:
            minus[int(sid)] += 1.0
        for sid in products:
            plus[int(sid)] += 1.0
        if np.any(minus > 2.0) or float(np.sum(minus)) > 2.0:
            raise ValueError("elementary reaction must have total reactant order <= 2")
        self._resize_existing_rows(n_species)
        channel_id = len(self.rates)
        self.nu_minus_rows.append(minus)
        self.nu_plus_rows.append(plus)
        self.rates.append(float(rate))
        label = {
            "channel_id": int(channel_id),
            "block_type": str(block_type),
            "role": str(role),
            "source_channel": None if source_channel is None else int(source_channel),
            **metadata,
        }
        self.labels.append(label)
        if source_channel is not None:
            self.source_to_elementary.setdefault(int(source_channel), []).append(int(channel_id))
        return channel_id

    def _resize_existing_rows(self, n_species: int) -> None:
        for rows in (self.nu_minus_rows, self.nu_plus_rows):
            for index, row in enumerate(rows):
                if row.shape[0] == n_species:
                    continue
                grown = np.zeros(n_species, dtype=float)
                grown[: row.shape[0]] = row
                rows[index] = grown

    def _base_rate(self, channel_id: int) -> float:
        block, local = self.network._block_and_local(channel_id)
        if block == ChannelBlock.LEFT_ADD:
            return float(self.network.left_add_rates[local])
        if block == ChannelBlock.RIGHT_ADD:
            return float(self.network.right_add_rates[local])
        if block == ChannelBlock.LEFT_SPLIT:
            return float(self.network.left_split_rates[local] * self.network.left_split_multiplicity[local])
        if block == ChannelBlock.RIGHT_SPLIT:
            return float(self.network.right_split_rates[local] * self.network.right_split_multiplicity[local])
        if block == ChannelBlock.OUTFLOW:
            return float(self.network.outflow_rates[local])
        if block == ChannelBlock.INFLOW:
            return float(self.network.inflow_rates[local])
        raise ValueError(f"unsupported channel block: {block}")

    def _binding_rate(self, strength: float) -> float:
        if self.config.catalyst_binding_rate_per_strength > 0.0:
            return float(self.config.catalyst_binding_rate_per_strength * float(strength))
        return float(self.config.catalyst_binding_rate)

    def _catalytic_turnover_rate(self, source_channel: int, strength: float) -> float:
        if self.config.catalytic_turnover_rate is not None:
            return float(self.config.catalytic_turnover_rate)
        return float(self._base_rate(source_channel) * float(strength) * self.config.catalytic_turnover_scale)


def build_elementary_mass_action_network(
    network: ReactionNetworkData,
    config: ElementaryExpansionConfig | None = None,
) -> ElementaryMassActionNetwork:
    """Precompute an elementary mass-action view of a polymer-rule network."""

    return _ElementaryBuilder(network, config or ElementaryExpansionConfig()).build()
