from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from polymer_sim.core.enums import BLOCK_NAMES, BLOCK_ORDER, ChannelBlock
from polymer_sim.core.state import SystemState
from polymer_sim.model.catalysis import dense_catalysis_block
from polymer_sim.model.rules import ReactionRuleTables
from polymer_sim.model.species import SpeciesSpace


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

    left_add_local_id: np.ndarray
    right_add_local_id: np.ndarray
    left_split_local_id_by_source: np.ndarray
    right_split_local_id_by_source: np.ndarray
    outflow_local_id_by_source: np.ndarray

    left_add_rates: np.ndarray
    right_add_rates: np.ndarray
    left_split_rates: np.ndarray
    right_split_rates: np.ndarray
    outflow_rates: np.ndarray

    cat_left_add: np.ndarray
    cat_right_add: np.ndarray
    cat_left_split: np.ndarray
    cat_right_split: np.ndarray
    cat_outflow: np.ndarray

    catalysis_mode: str
    saturation_alpha: float

    channel_block_type: np.ndarray
    channel_local_id: np.ndarray
    channel_offsets: dict[ChannelBlock, int]
    channel_sizes: dict[ChannelBlock, int]

    channel_to_species: list[np.ndarray]
    species_to_channels: list[np.ndarray]
    channel_to_catalysts: list[np.ndarray]

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

        left_add_target_a = np.asarray(left_add_target, dtype=np.int64)
        right_add_target_a = np.asarray(right_add_target, dtype=np.int64)
        left_split_source_a = np.asarray(left_split_source, dtype=np.int64)
        right_split_source_a = np.asarray(right_split_source, dtype=np.int64)
        outflow_source_a = np.asarray(outflow_source, dtype=np.int64)

        channel_sizes = {
            ChannelBlock.LEFT_ADD: len(left_add_target_a),
            ChannelBlock.RIGHT_ADD: len(right_add_target_a),
            ChannelBlock.LEFT_SPLIT: len(left_split_source_a),
            ChannelBlock.RIGHT_SPLIT: len(right_split_source_a),
            ChannelBlock.OUTFLOW: len(outflow_source_a),
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
            left_add_local_id=left_add_local_id,
            right_add_local_id=right_add_local_id,
            left_split_local_id_by_source=left_split_local_id_by_source,
            right_split_local_id_by_source=right_split_local_id_by_source,
            outflow_local_id_by_source=outflow_local_id_by_source,
            left_add_rates=_rates(k_poly_left, len(left_add_target_a), "k_poly_left"),
            right_add_rates=_rates(k_poly_right, len(right_add_target_a), "k_poly_right"),
            left_split_rates=_rates(k_frag_left, len(left_split_source_a), "k_frag_left"),
            right_split_rates=_rates(k_frag_right, len(right_split_source_a), "k_frag_right"),
            outflow_rates=_rates(k_outflow, len(outflow_source_a), "k_outflow"),
            cat_left_add=dense_catalysis_block(len(left_add_target_a), n_species),
            cat_right_add=dense_catalysis_block(len(right_add_target_a), n_species),
            cat_left_split=dense_catalysis_block(len(left_split_source_a), n_species),
            cat_right_split=dense_catalysis_block(len(right_split_source_a), n_species),
            cat_outflow=dense_catalysis_block(len(outflow_source_a), n_species),
            catalysis_mode=catalysis_mode,
            saturation_alpha=saturation_alpha,
            channel_block_type=channel_block_type,
            channel_local_id=channel_local_id,
            channel_offsets=channel_offsets,
            channel_sizes=channel_sizes,
            channel_to_species=[],
            species_to_channels=[],
            channel_to_catalysts=[],
        )
        network.rebuild_dependency_indices()
        return network

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

    def apply_channel_delta(self, x: np.ndarray, channel_id: int, amount: float) -> None:
        block, local = self._block_and_local(channel_id)
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
        source = int(self.right_split_source[local])
        rest = int(self.right_split_rest[local])
        monomer = int(self.right_split_monomer[local])
        x[source] -= a
        x[rest] += a
        x[monomer] += a

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
        if mirror_reverse:
            for reverse_channel_id in self.get_reverse_channel_ids(channel_id):
                self._set_catalytic_strength_no_rebuild(int(reverse_channel_id), sid, float(strength))
        if rebuild:
            self.rebuild_dependency_indices()

    def get_reverse_channel_ids(self, channel_id: int) -> np.ndarray:
        self._check_channel(channel_id)
        reactants_key = _species_tuple_key(self.get_channel_reactants(channel_id))
        products_key = _species_tuple_key(self.get_channel_products(channel_id))
        reverse_channel_ids: list[int] = []
        for other_channel_id in range(self.n_channels):
            if other_channel_id == int(channel_id):
                continue
            if _species_tuple_key(self.get_channel_reactants(other_channel_id)) != products_key:
                continue
            if _species_tuple_key(self.get_channel_products(other_channel_id)) != reactants_key:
                continue
            reverse_channel_ids.append(other_channel_id)
        return np.asarray(reverse_channel_ids, dtype=np.int64)

    def get_channel_catalysts(self, channel_id: int) -> np.ndarray:
        self._check_channel(channel_id)
        if len(self.channel_to_catalysts) == self.n_channels:
            return self.channel_to_catalysts[int(channel_id)]
        return self._scan_channel_catalysts(channel_id)

    def get_catalytic_strength(self, channel_id: int, catalyst_sid: int) -> float:
        row = self._cat_row(channel_id)
        return float(row[int(catalyst_sid)])

    def get_catalytic_factor(self, channel_id: int, state: SystemState) -> float:
        cats = self.get_channel_catalysts(channel_id)
        if cats.size == 0:
            return 1.0
        row = self._cat_row(channel_id)
        if self.catalysis_mode == "linear" or not self._uses_substrate_saturating_catalysis(channel_id):
            return float(1.0 + np.dot(row[cats], state.x[cats]))

        substrate_capacity = self._substrate_capacity(channel_id, state)
        if substrate_capacity <= 0.0:
            return 1.0

        contribution = 0.0
        denominator_base = self.saturation_alpha * substrate_capacity
        for catalyst_sid in cats:
            x_c = max(float(state.x[int(catalyst_sid)]), 0.0)
            if x_c <= 0.0:
                continue
            strength = float(row[int(catalyst_sid)])
            contribution += strength * substrate_capacity * x_c / (denominator_base + x_c)
        return float(1.0 + contribution)

    def compute_base_propensity(self, channel_id: int, state: SystemState) -> float:
        x = state.x
        block, local = self._block_and_local(channel_id)
        if block == ChannelBlock.LEFT_ADD:
            m = int(self.left_add_monomer[local])
            sid = int(self.left_add_species[local])
            return float(self.left_add_rates[local] * _pair_count(x[m], x[sid], m == sid))
        if block == ChannelBlock.RIGHT_ADD:
            sid = int(self.right_add_species[local])
            m = int(self.right_add_monomer[local])
            return float(self.right_add_rates[local] * _pair_count(x[sid], x[m], sid == m))
        if block == ChannelBlock.LEFT_SPLIT:
            source = int(self.left_split_source[local])
            return float(self.left_split_rates[local] * self.left_split_multiplicity[local] * max(float(x[source]), 0.0))
        if block == ChannelBlock.OUTFLOW:
            source = int(self.outflow_source[local])
            return float(self.outflow_rates[local] * max(float(x[source]), 0.0))
        source = int(self.right_split_source[local])
        return float(self.right_split_rates[local] * self.right_split_multiplicity[local] * max(float(x[source]), 0.0))

    def compute_propensity(self, channel_id: int, state: SystemState) -> float:
        base = self.compute_base_propensity(channel_id, state)
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
        for channel_id in range(self.n_channels):
            propensities[channel_id] = self.compute_propensity(channel_id, state)
        return propensities

    def rebuild_dependency_indices(self) -> None:
        channel_to_species: list[np.ndarray] = []
        channel_to_catalysts: list[np.ndarray] = []
        reverse: list[set[int]] = [set() for _ in range(self.n_species)]

        for channel_id in range(self.n_channels):
            base_deps = self._base_dependency_species(channel_id)
            cats = self._scan_channel_catalysts(channel_id)
            deps = _unique_concat(base_deps, cats)
            channel_to_species.append(deps)
            channel_to_catalysts.append(cats)
            for sid in deps:
                reverse[int(sid)].add(channel_id)

        self.channel_to_species = channel_to_species
        self.channel_to_catalysts = channel_to_catalysts
        self.species_to_channels = [np.asarray(sorted(channels), dtype=np.int64) for channels in reverse]

    def _base_dependency_species(self, channel_id: int) -> np.ndarray:
        reactants = self.get_channel_reactants(channel_id)
        return np.asarray(sorted(set(int(sid) for sid in reactants)), dtype=np.int64)

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
        return self.cat_right_split[local]

    def _set_catalytic_strength_no_rebuild(self, channel_id: int, catalyst_sid: int, strength: float) -> None:
        row = self._cat_row(channel_id)
        row[int(catalyst_sid)] = float(strength)

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
        x_a = max(float(x[a]), 0.0)
        x_b = max(float(x[b]), 0.0)
        if a == b:
            return float(np.floor(x_a / 2.0))
        return float(min(x_a, x_b))

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


def _species_tuple_key(species_ids: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(sid) for sid in species_ids))


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


def _unique_concat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.unique(b).astype(np.int64, copy=False)
    if b.size == 0:
        return np.unique(a).astype(np.int64, copy=False)
    return np.unique(np.concatenate((a, b))).astype(np.int64, copy=False)
