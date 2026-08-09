from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from polymer_sim.core.enums import BLOCK_NAMES, BLOCK_ORDER, ChannelBlock
from polymer_sim.core.state import SystemState
from polymer_sim.model.catalysis import dense_catalysis_block
from polymer_sim.model.rules import ReactionRuleTables
from polymer_sim.model.species import SpeciesSpace


class NumericalGuardStop(RuntimeError):
    """Raised when a numerical guard stops simulation before floating overflow."""

    def __init__(self, reason: str, metadata: dict[str, object]):
        super().__init__(reason)
        self.reason = str(reason)
        self.metadata = dict(metadata)


@dataclass(slots=True)
class ReactionNetworkData:
    species_names: list[str]
    name_to_idx: dict[str, int]
    x0: np.ndarray
    lengths: np.ndarray
    n_monomers: int
    max_len: int

    left_join: np.ndarray
    right_join: np.ndarray
    split_left_monomer: np.ndarray
    split_left_rest: np.ndarray
    split_right_rest: np.ndarray
    split_right_monomer: np.ndarray
    can_split: np.ndarray

    left_add_target: np.ndarray
    left_add_monomer: np.ndarray
    left_add_species: np.ndarray
    right_add_target: np.ndarray
    right_add_species: np.ndarray
    right_add_monomer: np.ndarray
    left_split_source: np.ndarray
    left_split_monomer: np.ndarray
    left_split_rest: np.ndarray
    left_split_multiplicity: np.ndarray
    right_split_source: np.ndarray
    right_split_rest: np.ndarray
    right_split_monomer: np.ndarray
    right_split_multiplicity: np.ndarray
    outflow_source: np.ndarray
    inflow_target: np.ndarray

    left_add_local_id: np.ndarray
    right_add_local_id: np.ndarray
    left_split_local_id_by_source: np.ndarray
    right_split_local_id_by_source: np.ndarray
    outflow_local_id_by_source: np.ndarray
    inflow_local_id_by_target: np.ndarray

    left_add_rates: np.ndarray
    right_add_rates: np.ndarray
    left_split_rates: np.ndarray
    right_split_rates: np.ndarray
    outflow_rates: np.ndarray
    inflow_rates: np.ndarray
    inflow_capacity: np.ndarray
    inflow_hill_coefficient: np.ndarray

    cat_left_add: np.ndarray
    cat_right_add: np.ndarray
    cat_left_split: np.ndarray
    cat_right_split: np.ndarray
    cat_outflow: np.ndarray
    cat_inflow: np.ndarray

    catalysis_mode: str
    saturation_alpha: float

    channel_block_type: np.ndarray
    channel_local_id: np.ndarray
    channel_offsets: dict[ChannelBlock, int]
    channel_sizes: dict[ChannelBlock, int]

    channel_to_species: list[np.ndarray]
    species_to_channels: list[np.ndarray]
    channel_to_catalysts: list[np.ndarray]
    channel_has_catalysts: np.ndarray
    dependency_indices_dirty: bool
    species_to_channels_indptr: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), init=False, repr=False)
    species_to_channels_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), init=False, repr=False)
    channel_to_catalyst_strengths: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    chemostat_species_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), init=False, repr=False)
    chemostat_species_values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float), init=False, repr=False)
    chemostat_species_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool), init=False, repr=False)
    _nu_minus_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _nu_plus_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _nu_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _nu_csr_cache: csr_matrix | None = field(default=None, init=False, repr=False)
    _block_local_ids_cache: dict[ChannelBlock, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _block_has_catalysts_cache: dict[ChannelBlock, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _block_any_catalysts_cache: dict[ChannelBlock, bool] = field(default_factory=dict, init=False, repr=False)
    _block_catalyst_local_ids: dict[ChannelBlock, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _block_catalyst_species_ids: dict[ChannelBlock, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _block_catalyst_strengths: dict[ChannelBlock, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _block_catalyst_row_ptr: dict[ChannelBlock, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _sparse_catalysis_ready: bool = field(default=False, init=False, repr=False)
    _all_inflow_capacity_infinite: bool = field(default=True, init=False, repr=False)
    channel_reverse_ids: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    reaction_order: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8), init=False, repr=False)
    reactant1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), init=False, repr=False)
    reactant2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), init=False, repr=False)
    homo_second_order: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool), init=False, repr=False)
    zero_order_channels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), init=False, repr=False)
    first_order_channels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), init=False, repr=False)
    second_order_channels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), init=False, repr=False)
    all_channels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), init=False, repr=False)

    @classmethod
    def from_species_space(
        cls,
        space: SpeciesSpace,
        tables: ReactionRuleTables,
        *,
        k_poly_left: float | Sequence[float] = 1.0,
        k_poly_right: float | Sequence[float] | None = None,
        k_frag_left: float | Sequence[float] = 1.0,
        k_frag_right: float | Sequence[float] | None = None,
        k_outflow: float | Sequence[float] = 0.0,
        outflow_species_ids: Sequence[int] | np.ndarray | None = None,
        k_inflow: float | Sequence[float] = 0.0,
        inflow_species_ids: Sequence[int] | np.ndarray | None = None,
        inflow_capacity: float | Sequence[float] | None = None,
        inflow_hill_coefficient: float | Sequence[float] = 1.0,
        chemostat_species_counts: dict[int | str, float] | None = None,
        catalysis_mode: str = "linear",
        saturation_alpha: float = 0.25,
    ) -> "ReactionNetworkData":
        k_poly_right = k_poly_left if k_poly_right is None else k_poly_right
        k_frag_right = k_frag_left if k_frag_right is None else k_frag_right
        catalysis_mode = _validate_catalysis_mode(catalysis_mode)
        saturation_alpha = _validate_saturation_alpha(saturation_alpha)

        n_species = space.n_species
        n_monomers = space.n_monomers

        left_add_local_id = np.full((n_monomers, n_species), -1, dtype=np.int64)
        left_add_monomer: list[int] = []
        left_add_species: list[int] = []
        left_add_target: list[int] = []
        for m in range(n_monomers):
            for sid in range(n_species):
                target = int(tables.left_join[m, sid])
                if target < 0:
                    continue
                local_id = len(left_add_target)
                left_add_local_id[m, sid] = local_id
                left_add_monomer.append(m)
                left_add_species.append(sid)
                left_add_target.append(target)

        right_add_local_id = np.full((n_species, n_monomers), -1, dtype=np.int64)
        right_add_species: list[int] = []
        right_add_monomer: list[int] = []
        right_add_target: list[int] = []
        for sid in range(n_species):
            for m in range(n_monomers):
                target = int(tables.right_join[sid, m])
                if target < 0:
                    continue
                local_id = len(right_add_target)
                right_add_local_id[sid, m] = local_id
                right_add_species.append(sid)
                right_add_monomer.append(m)
                right_add_target.append(target)

        left_split_local_id_by_source = np.full(n_species, -1, dtype=np.int64)
        left_split_source: list[int] = []
        left_split_monomer: list[int] = []
        left_split_rest: list[int] = []
        left_split_multiplicity: list[float] = []
        for sid in range(n_species):
            if not bool(tables.can_split[sid]):
                continue
            local_id = len(left_split_source)
            left_split_local_id_by_source[sid] = local_id
            left_split_source.append(sid)
            left_split_monomer.append(int(tables.split_left_monomer[sid]))
            left_split_rest.append(int(tables.split_left_rest[sid]))
            left_split_multiplicity.append(2.0 if int(space.lengths[sid]) == 2 else 1.0)

        right_split_local_id_by_source = np.full(n_species, -1, dtype=np.int64)
        right_split_source: list[int] = []
        right_split_rest: list[int] = []
        right_split_monomer: list[int] = []
        right_split_multiplicity: list[float] = []
        for sid in range(n_species):
            if int(space.lengths[sid]) <= 2:
                continue
            local_id = len(right_split_source)
            right_split_local_id_by_source[sid] = local_id
            right_split_source.append(sid)
            right_split_rest.append(int(tables.split_right_rest[sid]))
            right_split_monomer.append(int(tables.split_right_monomer[sid]))
            right_split_multiplicity.append(1.0)

        outflow_local_id_by_source = np.full(n_species, -1, dtype=np.int64)
        outflow_source: list[int] = []
        if outflow_species_ids is not None:
            for sid in np.asarray(outflow_species_ids, dtype=np.int64):
                local_id = len(outflow_source)
                outflow_local_id_by_source[int(sid)] = local_id
                outflow_source.append(int(sid))

        inflow_local_id_by_target = np.full(n_species, -1, dtype=np.int64)
        inflow_target: list[int] = []
        if inflow_species_ids is not None:
            for sid in np.asarray(inflow_species_ids, dtype=np.int64):
                local_id = len(inflow_target)
                inflow_local_id_by_target[int(sid)] = local_id
                inflow_target.append(int(sid))

        left_add_target_a = np.asarray(left_add_target, dtype=np.int64)
        right_add_target_a = np.asarray(right_add_target, dtype=np.int64)
        left_split_source_a = np.asarray(left_split_source, dtype=np.int64)
        right_split_source_a = np.asarray(right_split_source, dtype=np.int64)
        outflow_source_a = np.asarray(outflow_source, dtype=np.int64)
        inflow_target_a = np.asarray(inflow_target, dtype=np.int64)

        channel_sizes = {
            ChannelBlock.LEFT_ADD: len(left_add_target_a),
            ChannelBlock.RIGHT_ADD: len(right_add_target_a),
            ChannelBlock.LEFT_SPLIT: len(left_split_source_a),
            ChannelBlock.RIGHT_SPLIT: len(right_split_source_a),
            ChannelBlock.OUTFLOW: len(outflow_source_a),
            ChannelBlock.INFLOW: len(inflow_target_a),
        }
        channel_offsets: dict[ChannelBlock, int] = {}
        cursor = 0
        for block in BLOCK_ORDER:
            channel_offsets[block] = cursor
            cursor += channel_sizes[block]

        channel_block_type = np.empty(cursor, dtype=np.int8)
        channel_local_id = np.empty(cursor, dtype=np.int64)
        for block in BLOCK_ORDER:
            start = channel_offsets[block]
            size = channel_sizes[block]
            channel_block_type[start : start + size] = int(block)
            channel_local_id[start : start + size] = np.arange(size, dtype=np.int64)

        network = cls(
            species_names=list(space.species_names),
            name_to_idx=dict(space.name_to_idx),
            x0=np.array(space.x0, dtype=float, copy=True),
            lengths=np.array(space.lengths, copy=True),
            n_monomers=space.n_monomers,
            max_len=space.max_len,
            left_join=np.array(tables.left_join, copy=True),
            right_join=np.array(tables.right_join, copy=True),
            split_left_monomer=np.array(tables.split_left_monomer, copy=True),
            split_left_rest=np.array(tables.split_left_rest, copy=True),
            split_right_rest=np.array(tables.split_right_rest, copy=True),
            split_right_monomer=np.array(tables.split_right_monomer, copy=True),
            can_split=np.array(tables.can_split, copy=True),
            left_add_target=left_add_target_a,
            left_add_monomer=np.asarray(left_add_monomer, dtype=np.int64),
            left_add_species=np.asarray(left_add_species, dtype=np.int64),
            right_add_target=right_add_target_a,
            right_add_species=np.asarray(right_add_species, dtype=np.int64),
            right_add_monomer=np.asarray(right_add_monomer, dtype=np.int64),
            left_split_source=left_split_source_a,
            left_split_monomer=np.asarray(left_split_monomer, dtype=np.int64),
            left_split_rest=np.asarray(left_split_rest, dtype=np.int64),
            left_split_multiplicity=np.asarray(left_split_multiplicity, dtype=float),
            right_split_source=right_split_source_a,
            right_split_rest=np.asarray(right_split_rest, dtype=np.int64),
            right_split_monomer=np.asarray(right_split_monomer, dtype=np.int64),
            right_split_multiplicity=np.asarray(right_split_multiplicity, dtype=float),
            outflow_source=outflow_source_a,
            inflow_target=inflow_target_a,
            left_add_local_id=left_add_local_id,
            right_add_local_id=right_add_local_id,
            left_split_local_id_by_source=left_split_local_id_by_source,
            right_split_local_id_by_source=right_split_local_id_by_source,
            outflow_local_id_by_source=outflow_local_id_by_source,
            inflow_local_id_by_target=inflow_local_id_by_target,
            left_add_rates=_rates(k_poly_left, len(left_add_target_a), "k_poly_left"),
            right_add_rates=_rates(k_poly_right, len(right_add_target_a), "k_poly_right"),
            left_split_rates=_rates(k_frag_left, len(left_split_source_a), "k_frag_left"),
            right_split_rates=_rates(k_frag_right, len(right_split_source_a), "k_frag_right"),
            outflow_rates=_rates(k_outflow, len(outflow_source_a), "k_outflow"),
            inflow_rates=_rates(k_inflow, len(inflow_target_a), "k_inflow"),
            inflow_capacity=_capacities(inflow_capacity, len(inflow_target_a), "inflow_capacity"),
            inflow_hill_coefficient=_positive_values(
                inflow_hill_coefficient,
                len(inflow_target_a),
                "inflow_hill_coefficient",
            ),
            cat_left_add=dense_catalysis_block(len(left_add_target_a), n_species),
            cat_right_add=dense_catalysis_block(len(right_add_target_a), n_species),
            cat_left_split=dense_catalysis_block(len(left_split_source_a), n_species),
            cat_right_split=dense_catalysis_block(len(right_split_source_a), n_species),
            cat_outflow=dense_catalysis_block(len(outflow_source_a), n_species),
            cat_inflow=dense_catalysis_block(len(inflow_target_a), n_species),
            catalysis_mode=catalysis_mode,
            saturation_alpha=saturation_alpha,
            channel_block_type=channel_block_type,
            channel_local_id=channel_local_id,
            channel_offsets=channel_offsets,
            channel_sizes=channel_sizes,
            channel_to_species=[],
            species_to_channels=[],
            channel_to_catalysts=[],
            channel_has_catalysts=np.zeros(cursor, dtype=bool),
            dependency_indices_dirty=False,
        )
        network._initialize_runtime_caches()
        if chemostat_species_counts:
            network.set_chemostat_species(chemostat_species_counts, rebuild=False)
        network.rebuild_dependency_indices()
        network.precompute_stoichiometry_matrices()
        return network

    def _initialize_runtime_caches(self) -> None:
        """Initialize structural caches used by vectorized hot paths.

        These arrays are derived from channel sizes and inflow configuration,
        not from the current state.  Catalytic sparse caches are rebuilt later
        by ``rebuild_dependency_indices`` because they depend on the current
        catalytic assignments.
        """

        self._block_local_ids_cache = {}
        for block in BLOCK_ORDER:
            ids = np.arange(int(self.channel_sizes[block]), dtype=np.int64)
            ids.setflags(write=False)
            self._block_local_ids_cache[block] = ids
        if self.chemostat_species_mask.shape != (self.n_species,):
            self.chemostat_species_mask = np.zeros(self.n_species, dtype=bool)
        if self.chemostat_species_values.shape != (self.n_species,):
            values = np.asarray(self.x0, dtype=float).copy()
            self.chemostat_species_values = values
        self._all_inflow_capacity_infinite = bool(np.all(~np.isfinite(self.inflow_capacity))) if self.inflow_capacity.size else True
        self._rebuild_reverse_channel_cache()
        self._precompute_channel_reactant_terms()

    def set_chemostat_species(
        self,
        target_counts: dict[int | str, float],
        *,
        rebuild: bool = True,
    ) -> None:
        """Treat selected species as fixed external concentrations.

        Chemostatted species still appear in reaction reactant/product labels,
        but simulation kernels read their fixed counts from this network object
        and exclude them from state deltas, stoichiometry matrices, and
        dependency propagation.  This replaces runner-level food projection for
        ``food_supply_mode='constant'``.
        """

        mask = np.zeros(self.n_species, dtype=bool)
        values = np.asarray(self.x0, dtype=float).copy()
        ids: list[int] = []
        for species, value in target_counts.items():
            sid = self.species_idx(str(species)) if isinstance(species, str) else int(species)
            if sid < 0 or sid >= self.n_species:
                raise IndexError(f"chemostat species id out of range: {sid}")
            count = float(value)
            if not np.isfinite(count) or count < 0.0:
                raise ValueError("chemostat target counts must be finite values >= 0")
            mask[sid] = True
            values[sid] = count
            self.x0[sid] = count
            ids.append(sid)

        self.chemostat_species_mask = mask
        self.chemostat_species_values = values
        self.chemostat_species_ids = np.asarray(sorted(set(ids)), dtype=np.int64)
        for arr in (self.chemostat_species_ids, self.chemostat_species_values, self.chemostat_species_mask):
            arr.setflags(write=False)
        self._nu_minus_cache = None
        self._nu_plus_cache = None
        self._nu_cache = None
        self._nu_csr_cache = None
        self.dependency_indices_dirty = True
        if rebuild:
            self.rebuild_dependency_indices()
            self.precompute_stoichiometry_matrices()

    @property
    def has_chemostat_species(self) -> bool:
        return bool(self.chemostat_species_ids.size)

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    @property
    def n_channels(self) -> int:
        return int(self.channel_block_type.shape[0])

    def species_idx(self, name: str) -> int:
        return self.name_to_idx[name]

    def channel_id(self, block: ChannelBlock | int, local_id: int) -> int:
        block_e = ChannelBlock(int(block))
        local = int(local_id)
        if local < 0 or local >= self.channel_sizes[block_e]:
            raise IndexError(f"local_id out of range for {BLOCK_NAMES[block_e]}: {local}")
        return self.channel_offsets[block_e] + local

    def get_channel_block(self, channel_id: int) -> ChannelBlock:
        self._check_channel(channel_id)
        return ChannelBlock(int(self.channel_block_type[int(channel_id)]))

    def get_channel_local_id(self, channel_id: int) -> int:
        self._check_channel(channel_id)
        return int(self.channel_local_id[int(channel_id)])

    def get_channel_block_name(self, channel_id: int) -> str:
        return BLOCK_NAMES[self.get_channel_block(channel_id)]

    @property
    def rate_constants(self) -> np.ndarray:
        """Return the uncatalyzed base rate for each global channel."""

        values = np.empty(self.n_channels, dtype=float)
        for channel_id in range(self.n_channels):
            block, local = self._block_and_local(channel_id)
            if block == ChannelBlock.LEFT_ADD:
                values[channel_id] = float(self.left_add_rates[local])
            elif block == ChannelBlock.RIGHT_ADD:
                values[channel_id] = float(self.right_add_rates[local])
            elif block == ChannelBlock.LEFT_SPLIT:
                values[channel_id] = float(self.left_split_rates[local] * self.left_split_multiplicity[local])
            elif block == ChannelBlock.RIGHT_SPLIT:
                values[channel_id] = float(self.right_split_rates[local] * self.right_split_multiplicity[local])
            elif block == ChannelBlock.OUTFLOW:
                values[channel_id] = float(self.outflow_rates[local])
            elif block == ChannelBlock.INFLOW:
                values[channel_id] = float(self.inflow_rates[local])
            else:
                raise ValueError(f"unsupported channel block: {block}")
        return values

    @property
    def nu_minus(self) -> np.ndarray:
        if self._nu_minus_cache is None:
            self.precompute_stoichiometry_matrices()
        return self._nu_minus_cache

    @property
    def nu_plus(self) -> np.ndarray:
        if self._nu_plus_cache is None:
            self.precompute_stoichiometry_matrices()
        return self._nu_plus_cache

    @property
    def nu(self) -> np.ndarray:
        if self._nu_cache is None:
            self.precompute_stoichiometry_matrices()
        return self._nu_cache

    @property
    def nu_csr(self) -> csr_matrix:
        """Return cached sparse effective stoichiometry, shape ``(n_channels, n_species)``.

        This is the sparse companion to ``nu`` used by CLE-style matrix
        multiplies.  It is built once with the dense stoichiometry cache instead
        of being reconstructed inside stepper hot paths.
        """

        if self._nu_csr_cache is None:
            self.precompute_stoichiometry_matrices()
        return self._nu_csr_cache

    def get_channel_reactants(self, channel_id: int) -> tuple[int, ...]:
        block, local = self._block_and_local(channel_id)
        if block == ChannelBlock.LEFT_ADD:
            return (int(self.left_add_monomer[local]), int(self.left_add_species[local]))
        if block == ChannelBlock.RIGHT_ADD:
            return (int(self.right_add_species[local]), int(self.right_add_monomer[local]))
        if block == ChannelBlock.LEFT_SPLIT:
            return (int(self.left_split_source[local]),)
        if block == ChannelBlock.OUTFLOW:
            return (int(self.outflow_source[local]),)
        if block == ChannelBlock.INFLOW:
            return ()
        return (int(self.right_split_source[local]),)

    def get_channel_products(self, channel_id: int) -> tuple[int, ...]:
        block, local = self._block_and_local(channel_id)
        if block == ChannelBlock.LEFT_ADD:
            return (int(self.left_add_target[local]),)
        if block == ChannelBlock.RIGHT_ADD:
            return (int(self.right_add_target[local]),)
        if block == ChannelBlock.LEFT_SPLIT:
            return (int(self.left_split_monomer[local]), int(self.left_split_rest[local]))
        if block == ChannelBlock.OUTFLOW:
            return ()
        if block == ChannelBlock.INFLOW:
            return (int(self.inflow_target[local]),)
        return (int(self.right_split_rest[local]), int(self.right_split_monomer[local]))

    def get_channel_main_species(self, channel_id: int) -> int:
        block, local = self._block_and_local(channel_id)
        if block == ChannelBlock.LEFT_ADD:
            return int(self.left_add_species[local])
        if block == ChannelBlock.RIGHT_ADD:
            return int(self.right_add_species[local])
        if block == ChannelBlock.LEFT_SPLIT:
            return int(self.left_split_source[local])
        if block == ChannelBlock.OUTFLOW:
            return int(self.outflow_source[local])
        if block == ChannelBlock.INFLOW:
            return int(self.inflow_target[local])
        return int(self.right_split_source[local])

    def describe_channel(self, channel_id: int) -> dict[str, object]:
        return {
            "channel_id": int(channel_id),
            "block_type": self.get_channel_block_name(channel_id),
            "local_id": self.get_channel_local_id(channel_id),
            "reactants": self.get_channel_reactants(channel_id),
            "products": self.get_channel_products(channel_id),
            "catalysts": tuple(int(sid) for sid in self.get_channel_catalysts(channel_id)),
        }

    def apply_channel_update(self, state: SystemState, channel_id: int) -> None:
        self.apply_channel_delta(state.x, channel_id, 1.0)

    def get_channel_changed_species(self, channel_id: int) -> np.ndarray:
        """Return species ids whose counts can change when this channel fires."""

        block, local = self._block_and_local(channel_id)
        if block == ChannelBlock.LEFT_ADD:
            return self._dynamic_species_ids(_unique_ints(
                int(self.left_add_monomer[local]),
                int(self.left_add_species[local]),
                int(self.left_add_target[local]),
            ))
        if block == ChannelBlock.RIGHT_ADD:
            return self._dynamic_species_ids(_unique_ints(
                int(self.right_add_species[local]),
                int(self.right_add_monomer[local]),
                int(self.right_add_target[local]),
            ))
        if block == ChannelBlock.LEFT_SPLIT:
            return self._dynamic_species_ids(_unique_ints(
                int(self.left_split_source[local]),
                int(self.left_split_monomer[local]),
                int(self.left_split_rest[local]),
            ))
        if block == ChannelBlock.OUTFLOW:
            return self._dynamic_species_ids(np.asarray([int(self.outflow_source[local])], dtype=np.int64))
        if block == ChannelBlock.INFLOW:
            return self._dynamic_species_ids(np.asarray([int(self.inflow_target[local])], dtype=np.int64))
        return self._dynamic_species_ids(_unique_ints(
            int(self.right_split_source[local]),
            int(self.right_split_rest[local]),
            int(self.right_split_monomer[local]),
        ))

    def apply_channel_delta(self, x: np.ndarray, channel_id: int, amount: float) -> None:
        block, local = self._block_and_local(channel_id)
        a = float(amount)
        if not self.has_chemostat_species:
            self._apply_channel_delta_unchecked(x, block, local, a)
            return
        if block == ChannelBlock.LEFT_ADD:
            m = int(self.left_add_monomer[local])
            sid = int(self.left_add_species[local])
            target = int(self.left_add_target[local])
            self._add_dynamic_delta(x, m, -a)
            self._add_dynamic_delta(x, sid, -a)
            self._add_dynamic_delta(x, target, a)
            return
        if block == ChannelBlock.RIGHT_ADD:
            sid = int(self.right_add_species[local])
            m = int(self.right_add_monomer[local])
            target = int(self.right_add_target[local])
            self._add_dynamic_delta(x, sid, -a)
            self._add_dynamic_delta(x, m, -a)
            self._add_dynamic_delta(x, target, a)
            return
        if block == ChannelBlock.LEFT_SPLIT:
            source = int(self.left_split_source[local])
            monomer = int(self.left_split_monomer[local])
            rest = int(self.left_split_rest[local])
            self._add_dynamic_delta(x, source, -a)
            self._add_dynamic_delta(x, monomer, a)
            self._add_dynamic_delta(x, rest, a)
            return
        if block == ChannelBlock.OUTFLOW:
            source = int(self.outflow_source[local])
            self._add_dynamic_delta(x, source, -a)
            return
        if block == ChannelBlock.INFLOW:
            target = int(self.inflow_target[local])
            self._add_dynamic_delta(x, target, a)
            return
        source = int(self.right_split_source[local])
        rest = int(self.right_split_rest[local])
        monomer = int(self.right_split_monomer[local])
        self._add_dynamic_delta(x, source, -a)
        self._add_dynamic_delta(x, rest, a)
        self._add_dynamic_delta(x, monomer, a)

    def _apply_channel_delta_unchecked(self, x: np.ndarray, block: ChannelBlock, local: int, amount: float) -> None:
        a = float(amount)
        if block == ChannelBlock.LEFT_ADD:
            m = int(self.left_add_monomer[local])
            sid = int(self.left_add_species[local])
            target = int(self.left_add_target[local])
            x[m] -= a
            x[sid] -= a
            x[target] += a
            return
        if block == ChannelBlock.RIGHT_ADD:
            sid = int(self.right_add_species[local])
            m = int(self.right_add_monomer[local])
            target = int(self.right_add_target[local])
            x[sid] -= a
            x[m] -= a
            x[target] += a
            return
        if block == ChannelBlock.LEFT_SPLIT:
            source = int(self.left_split_source[local])
            monomer = int(self.left_split_monomer[local])
            rest = int(self.left_split_rest[local])
            x[source] -= a
            x[monomer] += a
            x[rest] += a
            return
        if block == ChannelBlock.OUTFLOW:
            source = int(self.outflow_source[local])
            x[source] -= a
            return
        if block == ChannelBlock.INFLOW:
            target = int(self.inflow_target[local])
            x[target] += a
            return
        source = int(self.right_split_source[local])
        rest = int(self.right_split_rest[local])
        monomer = int(self.right_split_monomer[local])
        x[source] -= a
        x[rest] += a
        x[monomer] += a

    def _add_dynamic_delta(self, x: np.ndarray, sid: int, delta: float) -> None:
        if not bool(self.chemostat_species_mask[int(sid)]):
            x[int(sid)] += float(delta)

    def _dynamic_species_ids(self, species_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(species_ids, dtype=np.int64)
        if ids.size == 0 or not self.has_chemostat_species:
            return ids
        return ids[~self.chemostat_species_mask[ids]].astype(np.int64, copy=False)

    def _count_value(self, x: np.ndarray, sid: int) -> float:
        species_id = int(sid)
        if self.has_chemostat_species and bool(self.chemostat_species_mask[species_id]):
            return float(self.chemostat_species_values[species_id])
        return float(np.asarray(x, dtype=float)[species_id])

    def _count_values(self, x: np.ndarray, species_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        ids = np.asarray(species_ids, dtype=np.int64)
        values = np.asarray(x, dtype=float)[ids]
        if not self.has_chemostat_species or ids.size == 0:
            return values
        mask = self.chemostat_species_mask[ids]
        if not np.any(mask):
            return values
        values = np.array(values, dtype=float, copy=True)
        values[mask] = self.chemostat_species_values[ids[mask]]
        return values

    def _effective_state_values(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if not self.has_chemostat_species:
            return values
        effective = np.array(values, dtype=float, copy=True)
        effective[self.chemostat_species_mask] = self.chemostat_species_values[self.chemostat_species_mask]
        return effective

    def precompute_stoichiometry_matrices(self) -> None:
        """Build and cache dense stoichiometry matrices for structural analyses.

        The polymer-rule hot path uses O(1) local updates and block-vectorized
        propensities, so it does not need these matrices.  PDMP partitioning,
        Algorithm-4-style fast-subnetwork scans, and debug/export code do need
        ``nu_minus``, ``nu_plus``, and ``nu`` repeatedly.  They are structural:
        catalytic strength changes do not affect them, so caching avoids
        rebuilding the full dense matrix on every property access.
        """

        nu_minus = self._stoichiometry_matrix(products=False)
        nu_plus = self._stoichiometry_matrix(products=True)
        nu = nu_plus - nu_minus
        nu_csr = csr_matrix(nu)
        for matrix in (nu_minus, nu_plus, nu):
            matrix.setflags(write=False)
        self._nu_minus_cache = nu_minus
        self._nu_plus_cache = nu_plus
        self._nu_cache = nu
        self._nu_csr_cache = nu_csr

    def _stoichiometry_matrix(self, *, products: bool) -> np.ndarray:
        matrix = np.zeros((self.n_channels, self.n_species), dtype=float)
        for channel_id in range(self.n_channels):
            species = self.get_channel_products(channel_id) if products else self.get_channel_reactants(channel_id)
            for sid in species:
                if self.has_chemostat_species and bool(self.chemostat_species_mask[int(sid)]):
                    continue
                matrix[int(channel_id), int(sid)] += 1.0
        return matrix

    def set_catalytic_strength(
        self,
        channel_id: int,
        catalyst_sid: int,
        strength: float,
        *,
        rebuild: bool = True,
        mirror_reverse: bool = True,
    ) -> None:
        sid = int(catalyst_sid)
        if sid < 0 or sid >= self.n_species:
            raise IndexError(f"catalyst_sid out of range: {sid}")
        self._set_catalytic_strength_no_rebuild(channel_id, sid, float(strength))
        self._refresh_channel_has_catalysts(channel_id)
        if mirror_reverse:
            for reverse_channel_id in self.get_reverse_channel_ids(channel_id):
                self._set_catalytic_strength_no_rebuild(int(reverse_channel_id), sid, float(strength))
                self._refresh_channel_has_catalysts(int(reverse_channel_id))
        self._invalidate_catalysis_runtime_caches()
        if rebuild:
            self.rebuild_dependency_indices()
        else:
            self.dependency_indices_dirty = True

    def set_catalytic_strengths(
        self,
        channel_ids: np.ndarray | Sequence[int],
        catalyst_sids: int | np.integer | np.ndarray | Sequence[int],
        strengths: float | np.floating | np.ndarray | Sequence[float],
        *,
        rebuild: bool = True,
        mirror_reverse: bool = True,
    ) -> None:
        """Assign catalytic strengths for many channels in one structural edit.

        This is the preferred construction-time path for deterministic or
        large catalytic networks.  It keeps the dense per-block matrices as the
        authoritative storage, mirrors reverse reactions through the cached
        structural reverse lookup, and rebuilds sparse dependency/catalysis
        indices once at the end instead of once per channel.
        """

        channels = np.asarray(channel_ids, dtype=np.int64)
        if channels.ndim != 1:
            raise ValueError("channel_ids must be a 1D array")
        if channels.size == 0:
            if rebuild:
                self.rebuild_dependency_indices()
            return
        if np.any((channels < 0) | (channels >= self.n_channels)):
            raise IndexError("channel_ids contain out-of-range values")

        catalysts = _broadcast_1d(catalyst_sids, channels.size, "catalyst_sids", np.int64)
        if np.any((catalysts < 0) | (catalysts >= self.n_species)):
            raise IndexError("catalyst_sids contain out-of-range values")
        values = _broadcast_1d(strengths, channels.size, "strengths", float)

        expanded_channels = channels
        expanded_catalysts = catalysts
        expanded_values = values
        if mirror_reverse:
            reverse_ids_by_channel = [self.get_reverse_channel_ids(int(channel_id)) for channel_id in channels]
            reverse_count = int(sum(reverse_ids.size for reverse_ids in reverse_ids_by_channel))
            if reverse_count:
                total = int(channels.size + reverse_count)
                expanded_channels = np.empty(total, dtype=np.int64)
                expanded_catalysts = np.empty(total, dtype=np.int64)
                expanded_values = np.empty(total, dtype=float)
                expanded_channels[: channels.size] = channels
                expanded_catalysts[: channels.size] = catalysts
                expanded_values[: channels.size] = values
                cursor = int(channels.size)
                for index, reverse_ids in enumerate(reverse_ids_by_channel):
                    n_reverse = int(reverse_ids.size)
                    if n_reverse == 0:
                        continue
                    end = cursor + n_reverse
                    expanded_channels[cursor:end] = reverse_ids
                    expanded_catalysts[cursor:end] = catalysts[index]
                    expanded_values[cursor:end] = values[index]
                    cursor = end

        self._set_catalytic_strengths_no_rebuild(expanded_channels, expanded_catalysts, expanded_values)
        self._refresh_channel_has_catalysts_many(expanded_channels)
        self._invalidate_catalysis_runtime_caches()
        if rebuild:
            self.rebuild_dependency_indices()
        else:
            self.dependency_indices_dirty = True

    def get_reverse_channel_ids(self, channel_id: int) -> np.ndarray:
        self._check_channel(channel_id)
        if len(self.channel_reverse_ids) != self.n_channels:
            self._rebuild_reverse_channel_cache()
        return self.channel_reverse_ids[int(channel_id)]

    def get_channel_catalysts(self, channel_id: int) -> np.ndarray:
        self._check_channel(channel_id)
        if not self.dependency_indices_dirty and len(self.channel_to_catalysts) == self.n_channels:
            return self.channel_to_catalysts[int(channel_id)]
        return self._scan_channel_catalysts(channel_id)

    def get_catalytic_strength(self, channel_id: int, catalyst_sid: int) -> float:
        row = self._cat_row(channel_id)
        return float(row[int(catalyst_sid)])

    def _channel_catalyst_strengths(self, channel_id: int, catalysts: np.ndarray) -> np.ndarray:
        if (
            not self.dependency_indices_dirty
            and len(self.channel_to_catalyst_strengths) == self.n_channels
        ):
            return self.channel_to_catalyst_strengths[int(channel_id)]
        row = self._cat_row(channel_id)
        return row[np.asarray(catalysts, dtype=np.int64)]

    def get_catalytic_factor(self, channel_id: int, state: SystemState) -> float:
        cid = int(channel_id)
        cats = self.get_channel_catalysts(channel_id)
        if cats.size == 0:
            return 1.0
        strengths = self._channel_catalyst_strengths(cid, cats)
        if self.catalysis_mode == "linear" or not self._uses_substrate_saturating_catalysis(channel_id):
            return float(1.0 + np.dot(strengths, self._count_values(state.x, cats)))

        substrate_capacity = self._substrate_capacity(channel_id, state)
        if substrate_capacity <= 0.0:
            return 1.0

        contribution = 0.0
        denominator_base = self.saturation_alpha * substrate_capacity
        for position, catalyst_sid in enumerate(cats):
            x_c = max(self._count_value(state.x, int(catalyst_sid)), 0.0)
            if x_c <= 0.0:
                continue
            strength = float(strengths[position])
            contribution += strength * substrate_capacity * x_c / (denominator_base + x_c)
        return float(1.0 + contribution)

    def compute_base_propensity(self, channel_id: int, state: SystemState) -> float:
        x = state.x
        block, local = self._block_and_local(channel_id)
        if block == ChannelBlock.LEFT_ADD:
            m = int(self.left_add_monomer[local])
            sid = int(self.left_add_species[local])
            return float(self.left_add_rates[local] * _pair_count(self._count_value(x, m), self._count_value(x, sid), m == sid))
        if block == ChannelBlock.RIGHT_ADD:
            sid = int(self.right_add_species[local])
            m = int(self.right_add_monomer[local])
            return float(self.right_add_rates[local] * _pair_count(self._count_value(x, sid), self._count_value(x, m), sid == m))
        if block == ChannelBlock.LEFT_SPLIT:
            source = int(self.left_split_source[local])
            return float(self.left_split_rates[local] * self.left_split_multiplicity[local] * max(self._count_value(x, source), 0.0))
        if block == ChannelBlock.OUTFLOW:
            source = int(self.outflow_source[local])
            return float(self.outflow_rates[local] * max(self._count_value(x, source), 0.0))
        if block == ChannelBlock.INFLOW:
            return float(self.inflow_rates[local] * self._inflow_capacity_factor(local, x))
        source = int(self.right_split_source[local])
        return float(self.right_split_rates[local] * self.right_split_multiplicity[local] * max(self._count_value(x, source), 0.0))

    def compute_propensity(self, channel_id: int, state: SystemState) -> float:
        base = self.compute_base_propensity(channel_id, state)
        if self.get_channel_block(channel_id) == ChannelBlock.INFLOW:
            return max(float(base), 0.0)
        if base <= 0.0:
            return 0.0
        if self._uses_substrate_saturating_catalysis(channel_id) and self._substrate_capacity(channel_id, state) <= 0.0:
            return 0.0
        value = base * self.get_catalytic_factor(channel_id, state)
        return max(float(value), 0.0)

    def compute_all_propensities(self, state: SystemState, out: np.ndarray | None = None) -> np.ndarray:
        propensities = np.empty(self.n_channels, dtype=float) if out is None else out
        if propensities.shape != (self.n_channels,):
            raise ValueError(f"out must have shape ({self.n_channels},)")
        x = np.asarray(state.x, dtype=float)
        for block in BLOCK_ORDER:
            start, end = self._block_bounds(block)
            if start == end:
                continue
            values = self._compute_block_base_propensities(block, None, x)
            propensities[start:end] = self._apply_block_catalysis(block, None, values, x)
        return propensities

    def compute_propensities_for_channels(
        self,
        channel_ids: np.ndarray | Sequence[int],
        state: SystemState,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute propensities for an arbitrary ordered set of global channels.

        This is the local-update companion to ``compute_all_propensities``.  It
        groups the requested channels by block so the inner propensity formulas
        still run on NumPy arrays rather than one Python call per channel.
        """

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
        block_types = self.channel_block_type[channels]
        for block in BLOCK_ORDER:
            mask = block_types == int(block)
            if not np.any(mask):
                continue
            local_ids = self.channel_local_id[channels[mask]]
            base = self._compute_block_base_propensities(block, local_ids, x)
            values[mask] = self._apply_block_catalysis(block, local_ids, base, x)
        return values

    def affected_channels_for_species(self, species_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        """Return channels whose propensity depends on any of ``species_ids``."""

        species = np.asarray(species_ids, dtype=np.int64)
        if species.ndim != 1:
            raise ValueError("species_ids must be a 1D array")
        if species.size == 0:
            return np.empty(0, dtype=np.int64)
        if np.any((species < 0) | (species >= self.n_species)):
            raise IndexError("species_ids contain out-of-range values")
        if self.dependency_indices_dirty or len(self.species_to_channels) != self.n_species:
            self.rebuild_dependency_indices()

        arrays = [self.species_to_channels[int(sid)] for sid in np.unique(species)]
        arrays = [arr for arr in arrays if arr.size]
        if not arrays:
            return np.empty(0, dtype=np.int64)
        if len(arrays) == 1:
            return np.array(arrays[0], dtype=np.int64, copy=True)
        return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)

    def update_propensities_for_species(
        self,
        propensities: np.ndarray,
        state: SystemState,
        species_ids: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        """Recompute cached propensities affected by changed species counts."""

        if propensities.shape != (self.n_channels,):
            raise ValueError(f"propensities must have shape ({self.n_channels},)")
        affected = self.affected_channels_for_species(species_ids)
        if affected.size:
            propensities[affected] = self.compute_propensities_for_channels(affected, state)
        return affected

    def rebuild_dependency_indices(self) -> None:
        channel_to_species: list[np.ndarray] = []
        channel_to_catalysts: list[np.ndarray] = []
        channel_to_catalyst_strengths: list[np.ndarray] = []
        reverse: list[set[int]] = [set() for _ in range(self.n_species)]
        channel_has_catalysts = np.zeros(self.n_channels, dtype=bool)

        for channel_id in range(self.n_channels):
            base_deps = self._base_dependency_species(channel_id)
            cats = self._scan_channel_catalysts(channel_id)
            strengths = self._cat_row(channel_id)[cats].astype(float, copy=True)
            cats.setflags(write=False)
            strengths.setflags(write=False)
            deps = _unique_concat(base_deps, self._dynamic_species_ids(cats))
            deps = self._dynamic_species_ids(deps)
            deps.setflags(write=False)
            channel_to_species.append(deps)
            channel_to_catalysts.append(cats)
            channel_to_catalyst_strengths.append(strengths)
            channel_has_catalysts[channel_id] = bool(cats.size)
            for sid in deps:
                reverse[int(sid)].add(channel_id)

        self.channel_to_species = channel_to_species
        self.channel_to_catalysts = channel_to_catalysts
        self.channel_to_catalyst_strengths = channel_to_catalyst_strengths
        self.channel_has_catalysts = channel_has_catalysts
        self.species_to_channels = [np.asarray(sorted(channels), dtype=np.int64) for channels in reverse]
        for channels in self.species_to_channels:
            channels.setflags(write=False)
        self._rebuild_species_to_channels_csr()
        self._rebuild_catalysis_runtime_caches(channel_has_catalysts)
        self.dependency_indices_dirty = False

    def _rebuild_species_to_channels_csr(self) -> None:
        """Build a CSR dependency view for compiled local-update kernels."""

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

    def _rebuild_catalysis_runtime_caches(self, channel_has_catalysts: np.ndarray) -> None:
        """Build per-block sparse catalyst indices from the dense assignment blocks.

        Dense ``cat_*`` matrices remain the authoritative assignment store
        because they are convenient for deterministic construction.  The
        propensity hot path uses the sparse CSR-like arrays below after a clean
        rebuild:

        - ``_block_catalyst_row_ptr[block][local_id:local_id+2]`` gives the
          slice into the flattened catalyst arrays for one local channel;
        - ``_block_catalyst_species_ids`` and ``_block_catalyst_strengths`` hold
          only nonzero catalytic entries;
        - ``_block_has_catalysts_cache`` and ``_block_any_catalysts_cache`` avoid
          repeated ``np.any`` calls for blocks without catalysts.
        """

        self._block_has_catalysts_cache = {}
        self._block_any_catalysts_cache = {}
        self._block_catalyst_local_ids = {}
        self._block_catalyst_species_ids = {}
        self._block_catalyst_strengths = {}
        self._block_catalyst_row_ptr = {}

        for block in BLOCK_ORDER:
            start, end = self._block_bounds(block)
            mask = np.array(channel_has_catalysts[start:end], dtype=bool, copy=True)
            mask.setflags(write=False)
            self._block_has_catalysts_cache[block] = mask
            self._block_any_catalysts_cache[block] = bool(mask.any())

            cat_block = self._cat_block(block)
            local_ids, catalyst_ids = np.nonzero(cat_block)
            local_ids = local_ids.astype(np.int64, copy=False)
            catalyst_ids = catalyst_ids.astype(np.int64, copy=False)
            strengths = cat_block[local_ids, catalyst_ids].astype(float, copy=True)

            row_counts = np.bincount(local_ids, minlength=int(self.channel_sizes[block]))
            row_ptr = np.empty(int(self.channel_sizes[block]) + 1, dtype=np.int64)
            row_ptr[0] = 0
            np.cumsum(row_counts, out=row_ptr[1:])

            for arr in (local_ids, catalyst_ids, strengths, row_ptr):
                arr.setflags(write=False)
            self._block_catalyst_local_ids[block] = local_ids
            self._block_catalyst_species_ids[block] = catalyst_ids
            self._block_catalyst_strengths[block] = strengths
            self._block_catalyst_row_ptr[block] = row_ptr

        self._sparse_catalysis_ready = True

    def _invalidate_catalysis_runtime_caches(self) -> None:
        """Mark sparse catalyst caches stale after direct catalytic assignment edits."""

        self._sparse_catalysis_ready = False
        self._block_has_catalysts_cache.clear()
        self._block_any_catalysts_cache.clear()
        self._block_catalyst_local_ids.clear()
        self._block_catalyst_species_ids.clear()
        self._block_catalyst_strengths.clear()
        self._block_catalyst_row_ptr.clear()

    def _base_dependency_species(self, channel_id: int) -> np.ndarray:
        block, local = self._block_and_local(channel_id)
        if block == ChannelBlock.INFLOW:
            return self._dynamic_species_ids(np.asarray([int(self.inflow_target[local])], dtype=np.int64))
        reactants = self.get_channel_reactants(channel_id)
        return self._dynamic_species_ids(np.asarray(sorted(set(int(sid) for sid in reactants)), dtype=np.int64))

    def _scan_channel_catalysts(self, channel_id: int) -> np.ndarray:
        row = self._cat_row(channel_id)
        return np.flatnonzero(row != 0.0).astype(np.int64, copy=False)

    def _cat_row(self, channel_id: int) -> np.ndarray:
        block, local = self._block_and_local(channel_id)
        if block == ChannelBlock.LEFT_ADD:
            return self.cat_left_add[local]
        if block == ChannelBlock.RIGHT_ADD:
            return self.cat_right_add[local]
        if block == ChannelBlock.LEFT_SPLIT:
            return self.cat_left_split[local]
        if block == ChannelBlock.OUTFLOW:
            return self.cat_outflow[local]
        if block == ChannelBlock.INFLOW:
            return self.cat_inflow[local]
        return self.cat_right_split[local]

    def _set_catalytic_strength_no_rebuild(self, channel_id: int, catalyst_sid: int, strength: float) -> None:
        row = self._cat_row(channel_id)
        row[int(catalyst_sid)] = float(strength)

    def _set_catalytic_strengths_no_rebuild(
        self,
        channel_ids: np.ndarray,
        catalyst_sids: np.ndarray,
        strengths: np.ndarray,
    ) -> None:
        channels = np.asarray(channel_ids, dtype=np.int64)
        catalysts = np.asarray(catalyst_sids, dtype=np.int64)
        values = np.asarray(strengths, dtype=float)
        block_types = self.channel_block_type[channels]
        for block in BLOCK_ORDER:
            mask = block_types == int(block)
            if not np.any(mask):
                continue
            local_ids = self.channel_local_id[channels[mask]]
            self._cat_block(block)[local_ids, catalysts[mask]] = values[mask]

    def _refresh_channel_has_catalysts(self, channel_id: int) -> None:
        cid = int(channel_id)
        if self.channel_has_catalysts.shape != (self.n_channels,):
            self.channel_has_catalysts = np.zeros(self.n_channels, dtype=bool)
        self.channel_has_catalysts[cid] = bool(np.any(self._cat_row(cid) != 0.0))

    def _refresh_channel_has_catalysts_many(self, channel_ids: np.ndarray) -> None:
        channels = np.unique(np.asarray(channel_ids, dtype=np.int64))
        if channels.size == 0:
            return
        if self.channel_has_catalysts.shape != (self.n_channels,):
            self.channel_has_catalysts = np.zeros(self.n_channels, dtype=bool)
        block_types = self.channel_block_type[channels]
        for block in BLOCK_ORDER:
            mask = block_types == int(block)
            if not np.any(mask):
                continue
            block_channels = channels[mask]
            local_ids = self.channel_local_id[block_channels]
            self.channel_has_catalysts[block_channels] = np.any(self._cat_block(block)[local_ids] != 0.0, axis=1)

    def _rebuild_reverse_channel_cache(self) -> None:
        """Build channel -> reverse-channel ids once from structural stoichiometry.

        Reverse lookup is used heavily when catalytic assignments mirror forward
        reactions to their fragmentation partners.  The mapping depends only on
        reactant/product species ids, so it should not be rebuilt during normal
        simulation or catalytic-strength sweeps.
        """

        keys: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        lookup: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
        for channel_id in range(self.n_channels):
            reactants_key = _species_tuple_key(self.get_channel_reactants(channel_id))
            products_key = _species_tuple_key(self.get_channel_products(channel_id))
            key = (reactants_key, products_key)
            keys.append(key)
            lookup.setdefault(key, []).append(int(channel_id))

        reverse_ids: list[np.ndarray] = []
        for channel_id, (reactants_key, products_key) in enumerate(keys):
            candidates = [
                int(other_channel_id)
                for other_channel_id in lookup.get((products_key, reactants_key), [])
                if int(other_channel_id) != int(channel_id)
            ]
            arr = np.asarray(candidates, dtype=np.int64)
            arr.setflags(write=False)
            reverse_ids.append(arr)
        self.channel_reverse_ids = reverse_ids

    def _precompute_channel_reactant_terms(self) -> None:
        """Precompute elementary reactant terms for vectorized availability checks.

        PDMP-Gillespie needs to reject discrete channels whose mass-action
        propensity is positive on a fractional continuous state but whose
        reactants cannot be consumed by one discrete firing.  Calling
        ``get_channel_reactants`` for every candidate channel is too expensive,
        so the hot path reads these arrays directly.
        """

        n_channels = self.n_channels
        order = np.zeros(n_channels, dtype=np.int8)
        reactant1 = np.full(n_channels, -1, dtype=np.int64)
        reactant2 = np.full(n_channels, -1, dtype=np.int64)
        homo_second_order = np.zeros(n_channels, dtype=bool)

        def assign(block: ChannelBlock, local_ids: np.ndarray, values_order: int, r1: np.ndarray | None, r2: np.ndarray | None = None) -> None:
            if local_ids.size == 0:
                return
            start = int(self.channel_offsets[block])
            channels = start + np.asarray(local_ids, dtype=np.int64)
            order[channels] = int(values_order)
            if r1 is not None:
                reactant1[channels] = np.asarray(r1, dtype=np.int64)
            if r2 is not None:
                r2_values = np.asarray(r2, dtype=np.int64)
                reactant2[channels] = r2_values
                homo_second_order[channels] = reactant1[channels] == r2_values

        assign(
            ChannelBlock.LEFT_ADD,
            self._block_local_ids_cache[ChannelBlock.LEFT_ADD],
            2,
            self.left_add_monomer,
            self.left_add_species,
        )
        assign(
            ChannelBlock.RIGHT_ADD,
            self._block_local_ids_cache[ChannelBlock.RIGHT_ADD],
            2,
            self.right_add_species,
            self.right_add_monomer,
        )
        assign(
            ChannelBlock.LEFT_SPLIT,
            self._block_local_ids_cache[ChannelBlock.LEFT_SPLIT],
            1,
            self.left_split_source,
        )
        assign(
            ChannelBlock.RIGHT_SPLIT,
            self._block_local_ids_cache[ChannelBlock.RIGHT_SPLIT],
            1,
            self.right_split_source,
        )
        assign(
            ChannelBlock.OUTFLOW,
            self._block_local_ids_cache[ChannelBlock.OUTFLOW],
            1,
            self.outflow_source,
        )

        all_channels = np.arange(n_channels, dtype=np.int64)
        zero_order_channels = np.flatnonzero(order == 0).astype(np.int64, copy=False)
        first_order_channels = np.flatnonzero(order == 1).astype(np.int64, copy=False)
        second_order_channels = np.flatnonzero(order == 2).astype(np.int64, copy=False)
        for arr in (
            order,
            reactant1,
            reactant2,
            homo_second_order,
            all_channels,
            zero_order_channels,
            first_order_channels,
            second_order_channels,
        ):
            arr.setflags(write=False)
        self.reaction_order = order
        self.reactant1 = reactant1
        self.reactant2 = reactant2
        self.homo_second_order = homo_second_order
        self.all_channels = all_channels
        self.zero_order_channels = zero_order_channels
        self.first_order_channels = first_order_channels
        self.second_order_channels = second_order_channels

    def _uses_substrate_saturating_catalysis(self, channel_id: int) -> bool:
        if self.catalysis_mode != "substrate_saturating":
            return False
        block = self.get_channel_block(channel_id)
        return block in (ChannelBlock.LEFT_ADD, ChannelBlock.RIGHT_ADD)

    def _substrate_capacity(self, channel_id: int, state: SystemState) -> float:
        block, local = self._block_and_local(channel_id)
        if block == ChannelBlock.LEFT_ADD:
            a = int(self.left_add_monomer[local])
            b = int(self.left_add_species[local])
        elif block == ChannelBlock.RIGHT_ADD:
            a = int(self.right_add_species[local])
            b = int(self.right_add_monomer[local])
        else:
            return 0.0

        x = state.x
        x_a = max(self._count_value(x, a), 0.0)
        x_b = max(self._count_value(x, b), 0.0)
        if a == b:
            return float(np.floor(x_a / 2.0))
        return float(min(x_a, x_b))

    def _block_bounds(self, block: ChannelBlock | int) -> tuple[int, int]:
        block_e = ChannelBlock(int(block))
        start = int(self.channel_offsets[block_e])
        return start, start + int(self.channel_sizes[block_e])

    def _cat_block(self, block: ChannelBlock | int) -> np.ndarray:
        block_e = ChannelBlock(int(block))
        if block_e == ChannelBlock.LEFT_ADD:
            return self.cat_left_add
        if block_e == ChannelBlock.RIGHT_ADD:
            return self.cat_right_add
        if block_e == ChannelBlock.LEFT_SPLIT:
            return self.cat_left_split
        if block_e == ChannelBlock.OUTFLOW:
            return self.cat_outflow
        if block_e == ChannelBlock.INFLOW:
            return self.cat_inflow
        return self.cat_right_split

    def _compute_block_base_propensities(
        self,
        block: ChannelBlock | int,
        local_ids: np.ndarray | None,
        x: np.ndarray,
    ) -> np.ndarray:
        block_e = ChannelBlock(int(block))
        ids = self._local_ids_for_block(block_e, local_ids)
        if ids.size == 0:
            return np.empty(0, dtype=float)

        if block_e == ChannelBlock.LEFT_ADD:
            m = self.left_add_monomer[ids]
            sid = self.left_add_species[ids]
            return self.left_add_rates[ids] * _pair_count_array(
                self._count_values(x, m),
                self._count_values(x, sid),
                m == sid,
            )
        if block_e == ChannelBlock.RIGHT_ADD:
            sid = self.right_add_species[ids]
            m = self.right_add_monomer[ids]
            return self.right_add_rates[ids] * _pair_count_array(
                self._count_values(x, sid),
                self._count_values(x, m),
                sid == m,
            )
        if block_e == ChannelBlock.LEFT_SPLIT:
            source = self.left_split_source[ids]
            return self.left_split_rates[ids] * self.left_split_multiplicity[ids] * np.maximum(self._count_values(x, source), 0.0)
        if block_e == ChannelBlock.OUTFLOW:
            source = self.outflow_source[ids]
            return self.outflow_rates[ids] * np.maximum(self._count_values(x, source), 0.0)
        if block_e == ChannelBlock.INFLOW:
            return self.inflow_rates[ids] * self._inflow_capacity_factor_values(ids, x)
        source = self.right_split_source[ids]
        return self.right_split_rates[ids] * self.right_split_multiplicity[ids] * np.maximum(self._count_values(x, source), 0.0)

    def _apply_block_catalysis(
        self,
        block: ChannelBlock | int,
        local_ids: np.ndarray | None,
        base: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        block_e = ChannelBlock(int(block))
        values = np.maximum(np.asarray(base, dtype=float), 0.0)
        if values.size == 0:
            return values
        if block_e == ChannelBlock.INFLOW:
            return values

        ids = self._local_ids_for_block(block_e, local_ids)
        use_sparse = self._can_use_sparse_catalysis(block_e)
        block_has_catalysts = (
            bool(self._block_any_catalysts_cache.get(block_e, False)) if use_sparse else True
        )

        if self.catalysis_mode == "substrate_saturating" and block_e in (
            ChannelBlock.LEFT_ADD,
            ChannelBlock.RIGHT_ADD,
        ):
            capacity = self._substrate_capacity_values(block_e, ids, x)
            values[capacity <= 0.0] = 0.0
            if not block_has_catalysts:
                np.maximum(values, 0.0, out=values)
                return values
            active = self._has_catalysts_for_local_ids(block_e, ids)
            active = active & (values > 0.0)
            if np.any(active):
                if use_sparse:
                    factors = self._sparse_substrate_saturating_factors(block_e, ids, capacity, x)
                    self._check_catalytic_multiply_guard(block_e, values[active], factors[active], x)
                    values[active] *= factors[active]
                else:
                    cat_block = self._cat_block(block_e)
                    factors = self._substrate_saturating_factors(cat_block[ids[active]], capacity[active], x)
                    self._check_catalytic_multiply_guard(block_e, values[active], factors, x)
                    values[active] *= factors
            np.maximum(values, 0.0, out=values)
            return values

        if not block_has_catalysts:
            np.maximum(values, 0.0, out=values)
            return values
        active = self._has_catalysts_for_local_ids(block_e, ids)
        active = active & (values > 0.0)
        if np.any(active):
            if use_sparse:
                factors = self._sparse_linear_catalytic_factors(block_e, ids, x)
                self._check_catalytic_multiply_guard(block_e, values[active], factors[active], x)
                values[active] *= factors[active]
            else:
                cat_block = self._cat_block(block_e)
                factors = 1.0 + cat_block[ids[active]] @ self._effective_state_values(x)
                self._check_catalytic_multiply_guard(block_e, values[active], factors, x)
                values[active] *= factors
        np.maximum(values, 0.0, out=values)
        return values

    def _check_catalytic_multiply_guard(
        self,
        block: ChannelBlock,
        values: np.ndarray,
        factors: np.ndarray,
        x: np.ndarray,
    ) -> None:
        threshold = 0.5 * float(np.finfo(float).max)
        base_values = np.asarray(values, dtype=float)
        factor_values = np.asarray(factors, dtype=float)
        if base_values.size == 0:
            return

        abs_base = np.abs(base_values)
        abs_factor = np.abs(factor_values)
        finite = np.isfinite(abs_base) & np.isfinite(abs_factor)
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_limit = threshold / abs_factor
        risky = (~finite) | ((abs_factor > 0.0) & (abs_base > safe_limit))
        if not np.any(risky):
            return

        projected = np.full(abs_base.shape, np.inf, dtype=float)
        safe = finite & (abs_factor > 0.0) & (abs_base <= safe_limit)
        np.multiply(abs_base, abs_factor, out=projected, where=safe)
        projected[finite & (abs_factor == 0.0)] = 0.0
        x_values = np.asarray(x, dtype=float)
        raise NumericalGuardStop(
            "numerical_guard_catalysis_overflow_risk",
            {
                "stop_reason": "numerical_guard_catalysis_overflow_risk",
                "guard": "catalytic_propensity_multiply",
                "threshold_fraction_of_float_max": 0.5,
                "threshold": float(threshold),
                "block_type": block.name,
                "n_risky_entries": int(np.count_nonzero(risky)),
                "max_state": float(np.nanmax(x_values)) if x_values.size else 0.0,
                "max_base_propensity": float(np.nanmax(abs_base)) if abs_base.size else 0.0,
                "max_catalytic_factor": float(np.nanmax(abs_factor)) if abs_factor.size else 0.0,
                "max_projected_propensity": float(np.nanmax(projected)) if projected.size else 0.0,
            },
        )

    def _local_ids_for_block(self, block: ChannelBlock, local_ids: np.ndarray | None) -> np.ndarray:
        if local_ids is None:
            ids = self._block_local_ids_cache.get(block)
            if ids is None or ids.shape != (int(self.channel_sizes[block]),):
                ids = np.arange(int(self.channel_sizes[block]), dtype=np.int64)
                ids.setflags(write=False)
                self._block_local_ids_cache[block] = ids
            return ids
        ids = np.asarray(local_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("local_ids must be a 1D array")
        return ids

    def _has_catalysts_for_local_ids(self, block: ChannelBlock, local_ids: np.ndarray) -> np.ndarray:
        if local_ids.size == 0:
            return np.zeros(0, dtype=bool)
        if not self.dependency_indices_dirty:
            mask = self._block_has_catalysts_cache.get(block)
            if mask is not None:
                all_ids = self._block_local_ids_cache.get(block)
                if local_ids is all_ids:
                    return mask
                return mask[local_ids]
        if self.channel_has_catalysts.shape != (self.n_channels,):
            self.rebuild_dependency_indices()
        offset = int(self.channel_offsets[block])
        return self.channel_has_catalysts[offset + local_ids]

    def _can_use_sparse_catalysis(self, block: ChannelBlock) -> bool:
        return (
            self._sparse_catalysis_ready
            and not self.dependency_indices_dirty
            and block in self._block_catalyst_row_ptr
        )

    def _sparse_linear_catalytic_factors(
        self,
        block: ChannelBlock,
        local_ids: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        row_ptr = self._block_catalyst_row_ptr[block]
        cat_local_ids = self._block_catalyst_local_ids[block]
        cat_species_ids = self._block_catalyst_species_ids[block]
        strengths = self._block_catalyst_strengths[block]
        factors = np.ones(local_ids.shape, dtype=float)
        if cat_species_ids.size == 0:
            return factors

        all_ids = self._block_local_ids_cache.get(block)
        if local_ids is all_ids:
            contribution = np.bincount(
                cat_local_ids,
                weights=strengths * self._count_values(x, cat_species_ids),
                minlength=int(self.channel_sizes[block]),
            )
            factors += contribution
            return factors

        x_values = self._effective_state_values(x)
        positions, entries = self._csr_entries_for_local_ids(row_ptr, local_ids)
        if entries.size:
            contribution = np.bincount(
                positions,
                weights=strengths[entries] * x_values[cat_species_ids[entries]],
                minlength=int(local_ids.size),
            )
            factors += contribution
        return factors

    def _sparse_substrate_saturating_factors(
        self,
        block: ChannelBlock,
        local_ids: np.ndarray,
        capacity: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        row_ptr = self._block_catalyst_row_ptr[block]
        cat_local_ids = self._block_catalyst_local_ids[block]
        cat_species_ids = self._block_catalyst_species_ids[block]
        strengths = self._block_catalyst_strengths[block]
        factors = np.ones(local_ids.shape, dtype=float)
        if cat_species_ids.size == 0:
            return factors

        x_values = np.maximum(self._effective_state_values(x), 0.0)
        all_ids = self._block_local_ids_cache.get(block)
        if local_ids is all_ids:
            cap_by_entry = np.asarray(capacity, dtype=float)[cat_local_ids]
            x_c = x_values[cat_species_ids]
            denom = self.saturation_alpha * cap_by_entry + x_c
            weights = np.zeros(cat_species_ids.shape, dtype=float)
            np.divide(
                strengths * cap_by_entry * x_c,
                denom,
                out=weights,
                where=denom > 0.0,
            )
            contribution = np.bincount(
                cat_local_ids,
                weights=weights,
                minlength=int(self.channel_sizes[block]),
            )
            factors += contribution
            return factors

        active_positions = np.flatnonzero(np.asarray(capacity, dtype=float) > 0.0)
        if active_positions.size == 0:
            return factors
        active_local_ids = local_ids[active_positions]
        positions, entries = self._csr_entries_for_local_ids(row_ptr, active_local_ids, active_positions)
        if entries.size:
            cap_by_entry = np.asarray(capacity, dtype=float)[positions]
            x_c = x_values[cat_species_ids[entries]]
            denom = self.saturation_alpha * cap_by_entry + x_c
            weights = np.zeros(entries.shape, dtype=float)
            np.divide(
                strengths[entries] * cap_by_entry * x_c,
                denom,
                out=weights,
                where=denom > 0.0,
            )
            contribution = np.bincount(
                positions,
                weights=weights,
                minlength=int(local_ids.size),
            )
            factors += contribution
        return factors

    def _csr_entries_for_local_ids(
        self,
        row_ptr: np.ndarray,
        local_ids: np.ndarray,
        positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(local_ids, dtype=np.int64)
        if ids.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        starts = row_ptr[ids]
        counts = row_ptr[ids + 1] - starts
        total = int(np.sum(counts))
        if total <= 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        if positions is None:
            row_positions = np.arange(ids.size, dtype=np.int64)
        else:
            row_positions = np.asarray(positions, dtype=np.int64)
            if row_positions.shape != ids.shape:
                raise ValueError("positions must match local_ids shape")

        repeated_positions = np.repeat(row_positions, counts)
        segment_starts = np.repeat(starts, counts)
        segment_offsets = np.repeat(np.cumsum(counts) - counts, counts)
        entries = segment_starts + (np.arange(total, dtype=np.int64) - segment_offsets)
        return repeated_positions, entries

    def _substrate_capacity_values(self, block: ChannelBlock, local_ids: np.ndarray, x: np.ndarray) -> np.ndarray:
        if block == ChannelBlock.LEFT_ADD:
            a = self.left_add_monomer[local_ids]
            b = self.left_add_species[local_ids]
        elif block == ChannelBlock.RIGHT_ADD:
            a = self.right_add_species[local_ids]
            b = self.right_add_monomer[local_ids]
        else:
            return np.zeros(local_ids.shape, dtype=float)

        x_a = np.maximum(self._count_values(x, a), 0.0)
        x_b = np.maximum(self._count_values(x, b), 0.0)
        same = a == b
        capacity = np.minimum(x_a, x_b)
        if np.any(same):
            capacity = np.array(capacity, dtype=float, copy=True)
            capacity[same] = np.floor(x_a[same] / 2.0)
        return capacity

    def _substrate_saturating_factors(
        self,
        cat_rows: np.ndarray,
        capacity: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        if cat_rows.size == 0:
            return np.ones(capacity.shape, dtype=float)
        x_c = np.maximum(self._effective_state_values(x), 0.0)
        cap = np.asarray(capacity, dtype=float)
        denom = self.saturation_alpha * cap[:, None] + x_c[None, :]
        scaled = np.zeros_like(denom, dtype=float)
        np.divide(cap[:, None] * x_c[None, :], denom, out=scaled, where=denom > 0.0)
        return 1.0 + np.sum(cat_rows * scaled, axis=1)

    def _inflow_capacity_factor(self, local_id: int, x: np.ndarray) -> float:
        capacity = float(self.inflow_capacity[int(local_id)])
        if not np.isfinite(capacity):
            return 1.0
        if capacity <= 0.0:
            return 0.0
        target = int(self.inflow_target[int(local_id)])
        count = max(self._count_value(x, target), 0.0)
        if count >= capacity:
            return 0.0
        hill = float(self.inflow_hill_coefficient[int(local_id)])
        return float(max(1.0 - (count / capacity) ** hill, 0.0))

    def _inflow_capacity_factor_values(self, local_ids: np.ndarray, x: np.ndarray) -> np.ndarray:
        ids = np.asarray(local_ids, dtype=np.int64)
        factors = np.ones(ids.shape, dtype=float)
        if ids.size == 0:
            return factors
        if self._all_inflow_capacity_infinite:
            return factors
        capacities = self.inflow_capacity[ids]
        finite = np.isfinite(capacities)
        if not np.any(finite):
            return factors

        factors[finite] = 0.0
        positive = finite & (capacities > 0.0)
        if not np.any(positive):
            return factors

        targets = self.inflow_target[ids[positive]]
        counts = np.maximum(self._count_values(x, targets), 0.0)
        capacity = capacities[positive]
        hill = self.inflow_hill_coefficient[ids[positive]]
        ratios = counts / capacity
        positive_factors = np.maximum(1.0 - ratios ** hill, 0.0)
        positive_factors[counts >= capacity] = 0.0
        factors[positive] = positive_factors
        return factors

    def _block_and_local(self, channel_id: int) -> tuple[ChannelBlock, int]:
        self._check_channel(channel_id)
        cid = int(channel_id)
        return ChannelBlock(int(self.channel_block_type[cid])), int(self.channel_local_id[cid])

    def _check_channel(self, channel_id: int) -> None:
        cid = int(channel_id)
        if cid < 0 or cid >= self.n_channels:
            raise IndexError(f"channel_id out of range: {cid}")


def _pair_count(a: float, b: float, same_species: bool) -> float:
    aa = max(float(a), 0.0)
    bb = max(float(b), 0.0)
    if same_species:
        return 0.5 * aa * max(aa - 1.0, 0.0)
    return aa * bb


def _pair_count_array(a: np.ndarray, b: np.ndarray, same_species: np.ndarray) -> np.ndarray:
    aa = np.maximum(np.asarray(a, dtype=float), 0.0)
    bb = np.maximum(np.asarray(b, dtype=float), 0.0)
    same = np.asarray(same_species, dtype=bool)
    counts = aa * bb
    if np.any(same):
        counts = np.array(counts, dtype=float, copy=True)
        counts[same] = 0.5 * aa[same] * np.maximum(aa[same] - 1.0, 0.0)
    return counts


def _unique_ints(*values: int) -> np.ndarray:
    return np.unique(np.asarray(values, dtype=np.int64)).astype(np.int64, copy=False)


def _species_tuple_key(species_ids: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(sid) for sid in species_ids))


def _broadcast_1d(value, n: int, name: str, dtype) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full(int(n), arr.item(), dtype=dtype)
    if arr.shape != (int(n),):
        raise ValueError(f"{name} must be scalar or shape ({int(n)},)")
    return np.asarray(arr, dtype=dtype)


def _validate_catalysis_mode(value: str) -> str:
    mode = str(value)
    if mode not in {"linear", "substrate_saturating"}:
        raise ValueError("catalysis_mode must be 'linear' or 'substrate_saturating'")
    return mode


def _validate_saturation_alpha(value: float) -> float:
    alpha = float(value)
    if alpha <= 0.0:
        raise ValueError("saturation_alpha must be > 0")
    return alpha


def _rates(value: float | Sequence[float], n: int, name: str) -> np.ndarray:
    if np.isscalar(value):
        return np.full(n, float(value), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},)")
    return np.array(arr, dtype=float, copy=True)


def _capacities(value: float | Sequence[float] | None, n: int, name: str) -> np.ndarray:
    if value is None:
        return np.full(n, np.inf, dtype=float)
    arr = _rates(value, n, name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} values must be >= 0")
    return arr


def _positive_values(value: float | Sequence[float], n: int, name: str) -> np.ndarray:
    arr = _rates(value, n, name)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} values must be > 0")
    return arr


def _unique_concat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.unique(b).astype(np.int64, copy=False)
    if b.size == 0:
        return np.unique(a).astype(np.int64, copy=False)
    return np.unique(np.concatenate((a, b))).astype(np.int64, copy=False)
