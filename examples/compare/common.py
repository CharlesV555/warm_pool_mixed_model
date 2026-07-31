from __future__ import annotations

"""Shared helpers for lightweight method comparisons.

All scripts in this folder use the same network registry and run settings.
The comparison target is: under a fixed wall-clock budget, how far each method
advances the simulation clock on the same reaction network input.
"""

import argparse
import cProfile
import csv
import io
import json
import pstats
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (  # noqa: E402
    BlendedHybridConfig,
    BlendedHybridStepper,
    ChannelBlock,
    ElementaryExpansionConfig,
    ElementaryMassActionNetwork,
    ExperimentRunner,
    FiniteMarkovConfig,
    FiniteMarkovScalingPDMPPartitionStrategy,
    LinearCatalysisScalingPDMPPartitionStrategy,
    NRMBlendedHybridStepper,
    OptimizedNRMStepper,
    PDMPConfig,
    PDMPStepper,
    ReactionNetworkData,
    ScalingPDMPConfig,
    SSAStepper,
    build_elementary_mass_action_network,
    build_reaction_rule_tables,
    clear_all_catalysis,
    generate_fixed_species_space,
    set_catalytic_strengths_for_channels,
)


STRICT_2018_PDMP_METHOD = "strict_2018_pdmp"

METHOD_ORDER = (
    "gillespie_ssa",
    "optimized_nrm",
    "gillespie_cle_hybrid",
    "nrm_cle_hybrid",
    "gillespie_pdmp_lp",
    "nrm_pdmp_lp",
    STRICT_2018_PDMP_METHOD,
)


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    name: str
    kind: str = "polymer_cross"
    max_len: int = 0
    alphabet: tuple[str, ...] = ("0", "1")
    initial_food_count: float = 1000.0
    food_max_count: float = 1000.0
    k_left_add: float = 0.01
    k_right_add: float = 0.01
    k_left_split: float = 0.011
    k_right_split: float = 0.011
    k_nonfood_outflow: float = 1.5
    food_inflow_rate: float = 10000.0
    food_inflow_hill_coefficient: float = 2.0
    use_hill_capped_food_inflow: bool = True
    catalysis_mode: str = "substrate_saturating"
    saturation_alpha: float = 0.01
    catalytic_gamma: float = 1000.0
    catalysis_seed: int = 123
    cross_catalysis_rules: tuple[tuple[str, str], ...] = (("0000", "0"),)


@dataclass(frozen=True, slots=True)
class RunSettings:
    seed: int = 123
    t_end: float | None = None
    max_steps: int = 100_000_000
    max_runtime_seconds: float = 10.0
    blended_i1: float = 100.0
    blended_i2: float = 150.0
    blended_dt_cle: float = 0.00033981
    # Keep the pure-CLE macro step no larger than the CLE micro step by default.
    # The fast dimerization benchmark is stiff enough that dt_macro=0.01 can
    # drive explicit CLE/Euler updates to NaN/inf before the wall-clock stop.
    blended_dt_macro: float = 0.00033981
    blended_beta_species_mode: str = "reactants"
    pdmp_ode_step: float = 0.001
    pdmp_n0: float = 500.0
    pdmp_mu: float = 1.0
    pdmp_eta: float = 0.9
    pdmp_delta: float = 0.9
    pdmp_repartition_on_event: bool = False
    pdmp_repartition_on_bounds: bool = True
    pdmp_enable_fast_subnetworks: bool = True
    pdmp_fast_subnetwork_threshold: float = 1.0
    pdmp_fast_subnetwork_max_size: int = 3
    pdmp_use_local_propensity_updates: bool = True
    pdmp_local_propensity_full_recompute_fraction: float = 0.5
    pdmp_heap_rebuild_factor: float = 4.0
    pdmp_finite_markov_max_states: int = 4096
    pdmp_finite_markov_max_total_internal_count: int = 10_000


NETWORK_SPECS: dict[str, NetworkSpec] = {
    "fast_dimerization": NetworkSpec(
        name="fast_dimerization",
        kind="fast_dimerization",
    ),
    "toggle_switch": NetworkSpec(
        name="toggle_switch",
        kind="toggle_switch",
    ),
    "repressilator": NetworkSpec(
        name="repressilator",
        kind="repressilator",
    ),
    "polymer_food_dimer_inhibition_len3": NetworkSpec(
        name="polymer_food_dimer_inhibition_len3",
        kind="polymer_food_dimer_inhibition_len3",
    ),
    "quasi_disjoint_slow_fast": NetworkSpec(
        name="quasi_disjoint_slow_fast",
        kind="quasi_disjoint_slow_fast",
    ),
    "shared_interface_species": NetworkSpec(
        name="shared_interface_species",
        kind="shared_interface_species",
    ),
    "loose_biological_modules": NetworkSpec(
        name="loose_biological_modules",
        kind="loose_biological_modules",
    ),
    "polymer_len5_00000_catalyzes_0": NetworkSpec(
        name="polymer_len5_00000_catalyzes_0",
        max_len=5,
        cross_catalysis_rules=(("00000", "0"),),
    ),
    "polymer_len10_two_stage_1_catalysis": NetworkSpec(
        name="polymer_len10_two_stage_1_catalysis",
        kind="polymer_len10_two_stage_1_catalysis",
        max_len=10,
    ),
    "polymer_len10_0000000000_catalyzes_0": NetworkSpec(
        name="polymer_len10_0000000000_catalyzes_0",
        max_len=10,
        cross_catalysis_rules=(("0000000000", "0"),),
    ),
    "linear_cross_len3": NetworkSpec(
        name="linear_cross_len3",
        max_len=3,
        cross_catalysis_rules=(("000", "0"),),
    ),
    "linear_cross_len4": NetworkSpec(
        name="linear_cross_len4",
        max_len=4,
        cross_catalysis_rules=(("0000", "0"),),
    ),
    "linear_cross_len5": NetworkSpec(
        name="linear_cross_len5",
        max_len=5,
        cross_catalysis_rules=(("00000", "0"),),
    ),
}
DEFAULT_NETWORKS = (
    "fast_dimerization",
    "toggle_switch",
    "repressilator",
    "polymer_len5_00000_catalyzes_0",
    "polymer_food_dimer_inhibition_len3",
    "polymer_len10_two_stage_1_catalysis",
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_SETTINGS = RunSettings()
STRICT_2018_PDMP_SKIP_NETWORKS = (
    # In DEFAULT_NETWORKS these are the 4th and 5th entries.  The strict 2018
    # path expands polymer catalysis to an elementary SRN and then performs
    # finite-CTMC fast-subnetwork checks, so these benchmarks are intentionally
    # skipped unless this constant is edited.
    DEFAULT_NETWORKS[3],
    DEFAULT_NETWORKS[4],
)


def build_network(network_name: str) -> tuple[ReactionNetworkData | ElementaryMassActionNetwork, dict[str, Any], NetworkSpec]:
    spec = network_spec(network_name)
    if spec.kind == "fast_dimerization":
        return (*build_fast_dimerization_benchmark_network(spec), spec)
    if spec.kind == "toggle_switch":
        return (*build_toggle_switch_network(spec), spec)
    if spec.kind == "repressilator":
        return (*build_repressilator_network(spec), spec)
    if spec.kind == "polymer_food_dimer_inhibition_len3":
        return (*build_polymer_food_dimer_inhibition_len3_network(spec), spec)
    if spec.kind == "polymer_len10_two_stage_1_catalysis":
        return (*build_polymer_len10_two_stage_one_catalysis_network(spec), spec)
    if spec.kind == "quasi_disjoint_slow_fast":
        return (*build_quasi_disjoint_slow_fast_network(spec), spec)
    if spec.kind == "shared_interface_species":
        return (*build_shared_interface_species_network(spec), spec)
    if spec.kind == "loose_biological_modules":
        return (*build_loose_biological_modules_network(spec), spec)
    if spec.kind != "polymer_cross":
        raise ValueError(f"unknown network kind {spec.kind!r}")
    return (*build_polymer_cross_catalysis_network(spec), spec)


def prepare_network_for_method(
    method: str,
    network: ReactionNetworkData | ElementaryMassActionNetwork,
    catalysis_result: dict[str, Any],
) -> tuple[ReactionNetworkData | ElementaryMassActionNetwork, dict[str, Any]]:
    method_key = normalize_method(method)
    if method_key != STRICT_2018_PDMP_METHOD:
        return network, catalysis_result
    if isinstance(network, ElementaryMassActionNetwork):
        metadata = dict(catalysis_result)
        metadata["strict_2018_srn"] = {
            "source": "already_elementary_mass_action",
            "standard_zero_order_inflow": True,
            "expanded_catalysis": False,
        }
        return network, metadata

    elementary_config = ElementaryExpansionConfig(
        standard_zero_order_inflow=True,
        include_uncatalyzed_channels=True,
        expand_catalysis=True,
        # Keep the strict SRN elementary: catalysis is represented by
        # catalyst+substrate <-> complex and complex turnover channels.
        # Existing catalytic strength gamma enters the turnover rate as
        # base_rate * gamma unless the config is changed here.
        catalyst_binding_rate=1.0,
        catalyst_unbinding_rate=1.0,
        catalytic_turnover_rate=None,
        catalytic_turnover_scale=1.0,
    )
    elementary = build_elementary_mass_action_network(network, elementary_config)
    metadata = {
        "source_catalysis_assignment": json_ready(catalysis_result),
        "strict_2018_srn": {
            "source": "ReactionNetworkData",
            "target": "ElementaryMassActionNetwork",
            "standard_zero_order_inflow": True,
            "expanded_catalysis": True,
            "source_n_species": int(network.n_species),
            "source_n_channels": int(network.n_channels),
            "elementary_n_species": int(elementary.n_species),
            "elementary_n_channels": int(elementary.n_channels),
            "polymer_species_count": int(elementary.polymer_species_count),
            "complex_species_count": int(max(elementary.n_species - elementary.polymer_species_count, 0)),
            "catalyst_binding_rate": float(elementary_config.catalyst_binding_rate),
            "catalyst_unbinding_rate": float(elementary_config.catalyst_unbinding_rate),
            "catalytic_turnover_scale": float(elementary_config.catalytic_turnover_scale),
        },
    }
    return elementary, metadata


def build_polymer_cross_catalysis_network(spec: NetworkSpec) -> tuple[ReactionNetworkData, dict[str, Any]]:
    initial_counts = {
        name: min(float(spec.initial_food_count), float(spec.food_max_count))
        for name in spec.alphabet
    }
    space = generate_fixed_species_space(spec.alphabet, max_len=int(spec.max_len), initial_counts=initial_counts)
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=float(spec.k_left_add),
        k_poly_right=float(spec.k_right_add),
        k_frag_left=float(spec.k_left_split),
        k_frag_right=float(spec.k_right_split),
        k_outflow=float(spec.k_nonfood_outflow),
        outflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name not in spec.alphabet
        ],
        k_inflow=float(spec.food_inflow_rate),
        inflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name in spec.alphabet
        ],
        inflow_capacity=float(spec.food_max_count) if spec.use_hill_capped_food_inflow else None,
        inflow_hill_coefficient=float(spec.food_inflow_hill_coefficient),
        catalysis_mode=str(spec.catalysis_mode),
        saturation_alpha=float(spec.saturation_alpha),
    )
    catalysis_result = assign_cross_terminal_catalysis(network, spec)
    return network, catalysis_result


def build_polymer_len10_two_stage_one_catalysis_network(spec: NetworkSpec) -> tuple[ReactionNetworkData, dict[str, Any]]:
    initial_counts = {
        name: min(float(spec.initial_food_count), float(spec.food_max_count))
        for name in spec.alphabet
    }
    space = generate_fixed_species_space(spec.alphabet, max_len=10, initial_counts=initial_counts)
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=float(spec.k_left_add),
        k_poly_right=float(spec.k_right_add),
        k_frag_left=float(spec.k_left_split),
        k_frag_right=float(spec.k_right_split),
        k_outflow=float(spec.k_nonfood_outflow),
        outflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name not in spec.alphabet
        ],
        k_inflow=float(spec.food_inflow_rate),
        inflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name in spec.alphabet
        ],
        inflow_capacity=float(spec.food_max_count) if spec.use_hill_capped_food_inflow else None,
        inflow_hill_coefficient=float(spec.food_inflow_hill_coefficient),
        catalysis_mode=str(spec.catalysis_mode),
        saturation_alpha=float(spec.saturation_alpha),
    )
    catalysis_result = assign_two_stage_one_polymer_catalysis(network, spec)
    return network, catalysis_result


def build_fast_dimerization_benchmark_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("s0", "s1", "s2")
    initial = {"s0": 540.0, "s1": 730.0, "s2": 0.0}
    reactions = [
        _reaction("R1", ("s0", "s0"), ("s1",), 1.0, partition="fast", note="s0 + s0 -> s1"),
        _reaction("R2", ("s1",), ("s0", "s0"), 200.0, partition="fast", note="s1 -> s0 + s0"),
        _reaction("R3", ("s0",), (), 0.02, partition="slow", note="s0 -> empty"),
        _reaction("R4", ("s1",), ("s2",), 0.004, partition="slow", note="s1 -> s2"),
    ]
    expected = {
        "slow_reactions": ["R3", "R4"],
        "fast_reactions": ["R1", "R2"],
        "R_star": [],
        "S_star": ["s0", "s1"],
        "source": "fast dimerization network table",
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


def build_toggle_switch_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("m_A", "m_B", "s_A", "s_B", "p_A", "p_B")
    initial = {name: 0.0 for name in species}
    reactions = [
        _reaction("R1", (), ("m_A",), 1.0, partition="slow"),
        _reaction("R2", (), ("m_B",), 1.0, partition="slow"),
        _reaction("R3", ("m_A",), (), 0.1, partition="slow"),
        _reaction("R4", ("m_B",), (), 0.1, partition="slow"),
        _reaction("R5", ("p_A",), (), 0.1, partition="slow"),
        _reaction("R6", ("p_B",), (), 0.1, partition="slow"),
        _reaction("R7", ("m_A",), ("s_A",), 5.0, partition="fast"),
        _reaction("R8", ("m_B",), ("s_B",), 5.0, partition="fast"),
        _reaction("R9", ("s_A", "m_B"), ("s_A",), 20.0, partition="slow", interface="S*"),
        _reaction("R10", ("s_B", "m_A"), ("s_B",), 20.0, partition="slow", interface="S*"),
        _reaction("R11", ("s_A",), ("s_A", "p_A"), 10.0, partition="fast"),
        _reaction("R12", ("s_B",), ("s_B", "p_B"), 10.0, partition="fast"),
        _reaction("R13", ("s_A",), (), 0.01, partition="fast"),
        _reaction("R14", ("s_B",), (), 0.01, partition="fast"),
    ]
    expected = {
        "slow_reactions": ["R1", "R2", "R3", "R4", "R5", "R6", "R9", "R10"],
        "fast_reactions": ["R7", "R8", "R11", "R12", "R13", "R14"],
        "R_star": [],
        "S_star": ["s_A", "s_B"],
        "source": "toggle switch reaction table",
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


def build_repressilator_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("m_A", "m_B", "m_C", "p_A", "p_B", "p_C")
    initial = {"m_A":10.0, "p_A":500,"m_B":0.0, "p_B":0.0, "m_C":0.0, "p_C":0.0}
    reactions = [
        _reaction("R1", (), ("m_A",), 0.1, partition="slow"),
        _reaction("R2", (), ("m_B",), 0.1, partition="slow"),
        _reaction("R3", (), ("m_C",), 0.1, partition="slow"),
        _reaction("R4", ("m_A",), ("m_A", "p_A"), 50.0, partition="fast"),
        _reaction("R5", ("m_B",), ("m_B", "p_B"), 50.0, partition="fast"),
        _reaction("R6", ("m_C",), ("m_C", "p_C"), 50.0, partition="fast"),
        _reaction("R7", ("m_A",), (), 0.01, partition="slow"),
        _reaction("R8", ("m_B",), (), 0.01, partition="slow"),
        _reaction("R9", ("m_C",), (), 0.01, partition="slow"),
        _reaction("R10", ("m_A", "p_B"), ("p_B",), 50.0, partition="slow", interface="S*"),
        _reaction("R11", ("m_B", "p_C"), ("p_C",), 50.0, partition="slow", interface="S*"),
        _reaction("R12", ("m_C", "p_A"), ("p_A",), 50.0, partition="slow", interface="S*"),
        _reaction("R13", ("p_A",), (), 0.01, partition="fast"),
        _reaction("R14", ("p_B",), (), 0.01, partition="fast"),
        _reaction("R15", ("p_C",), (), 0.01, partition="fast"),
    ]
    expected = {
        "slow_reactions": ["R1", "R2", "R3", "R7", "R8", "R9", "R10", "R11", "R12"],
        "fast_reactions": ["R4", "R5", "R6", "R13", "R14", "R15"],
        "R_star": [],
        "S_star": ["p_A", "p_B", "p_C"],
        "source": "repressilator reaction table",
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


def build_polymer_food_dimer_inhibition_len3_network(
    spec: NetworkSpec,
) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("0", "1", "00", "11", "000", "111")
    initial = {"0": 100.0, "1": 100.0, "00": 0.0, "11": 0.0, "000": 0.0, "111": 0.0}
    reactions = [
        _reaction("R1", (), ("0",), 1.0, partition="slow", note="food inflow 0"),
        _reaction("R2", (), ("1",), 1.0, partition="slow", note="food inflow 1"),
        _reaction("R3", ("0",), (), 0.1, partition="slow", note="food outflow 0"),
        _reaction("R4", ("1",), (), 0.1, partition="slow", note="food outflow 1"),
        _reaction("R5", ("0", "0"), ("00",), 5.0, partition="fast", note="0 dimerization using food monomers"),
        _reaction("R6", ("1", "1"), ("11",), 5.0, partition="fast", note="1 dimerization using food monomers"),
        _reaction("R7", ("00", "0"), ("000",), 10.0, partition="fast", note="0 elongation needs food monomer"),
        _reaction("R8", ("11", "1"), ("111",), 10.0, partition="fast", note="1 elongation needs food monomer"),
        _reaction("R9", ("00", "1"), ("00",), 20.0, partition="slow", interface="S*", note="00 suppresses 1 dimerization by depleting 1"),
        _reaction("R10", ("11", "0"), ("11",), 20.0, partition="slow", interface="S*", note="11 suppresses 0 dimerization by depleting 0"),
        _reaction("R11", ("00",), (), 0.01, partition="fast", note="00 outflow"),
        _reaction("R12", ("11",), (), 0.01, partition="fast", note="11 outflow"),
        _reaction("R13", ("000",), (), 0.1, partition="fast", note="000 outflow"),
        _reaction("R14", ("111",), (), 0.1, partition="fast", note="111 outflow"),
    ]
    expected = {
        "slow_reactions": ["R1", "R2", "R3", "R4", "R9", "R10"],
        "fast_reactions": ["R5", "R6", "R7", "R8", "R11", "R12", "R13", "R14"],
        "R_star": [],
        "S_star": ["00", "11"],
        "source": "custom food-dependent polymer toggle analogue",
        "inhibition_implementation": "cross-monomer depletion: 00 + 1 -> 00 and 11 + 0 -> 11",
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


def build_quasi_disjoint_slow_fast_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("S1", "S2", "S3", "S4", "S5")
    initial = {"S1": 20.0, "S2": 20.0, "S3": 200.0, "S4": 100.0, "S5": 100.0}
    reactions = [
        _reaction("R1", (), ("S1",), 1.0, partition="slow", note="birth S1"),
        _reaction("R2", ("S1",), (), 0.05, partition="slow", note="death S1"),
        _reaction("R3", (), ("S2",), 1.0, partition="slow", note="birth S2"),
        _reaction("R4", ("S2",), (), 0.05, partition="slow", note="death S2"),
        _reaction("R5", ("S1", "S2"), ("S2", "S3", "S3"), 0.001, partition="slow", interface="R*"),
        _reaction("R6", ("S3", "S4"), ("S5",), 0.001, partition="fast"),
        _reaction("R7", ("S4",), (), 1.0, partition="fast"),
        _reaction("R8", (), ("S4",), 100.0, partition="fast"),
        _reaction("R9", ("S5",), (), 1.0, partition="fast"),
        _reaction("R10", (), ("S5",), 100.0, partition="fast"),
    ]
    expected = {
        "slow_reactions": ["R1", "R2", "R3", "R4", "R5"],
        "fast_reactions": ["R6", "R7", "R8", "R9", "R10"],
        "R_star": ["R5"],
        "S_star": [],
        "slow_species": ["S1", "S2"],
        "fast_species": ["S3", "S4", "S5"],
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


def build_shared_interface_species_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("A", "B", "C", "D")
    initial = {"A": 20.0, "B": 0.0, "C": 500.0, "D": 500.0}
    reactions = [
        _reaction("R1", (), ("A",), 1.0, partition="slow", note="birth A"),
        _reaction("R2", ("A",), (), 0.05, partition="slow", note="death A"),
        _reaction("R3", ("A", "C"), ("C",), 0.001, partition="slow", interface="S*", note="C modulates slow propensity"),
        _reaction("R4", (), ("C",), 100.0, partition="fast"),
        _reaction("R5", ("C",), ("D",), 1.0, partition="fast"),
        _reaction("R6", ("D",), ("C",), 1.0, partition="fast"),
    ]
    expected = {
        "slow_reactions": ["R1", "R2", "R3"],
        "fast_reactions": ["R4", "R5", "R6"],
        "R_star": [],
        "S_star": ["C"],
        "slow_species": ["A"],
        "fast_species": ["C", "D"],
        "unused_species": ["B"],
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


def build_loose_biological_modules_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("DNA", "mRNA", "Protein", "M1", "M2", "M3", "M4")
    initial = {
        "DNA": 1.0,
        "mRNA": 0.0,
        "Protein": 20.0,
        "M1": 250.0,
        "M2": 250.0,
        "M3": 250.0,
        "M4": 250.0,
    }
    reactions = [
        _reaction("R1", ("DNA",), ("DNA", "mRNA"), 0.5, partition="slow"),
        _reaction("R2", ("mRNA",), (), 0.2, partition="slow"),
        _reaction("R3", ("mRNA",), ("mRNA", "Protein"), 2.0, partition="slow"),
        _reaction("R4", ("Protein",), (), 0.05, partition="slow"),
        _reaction("R5", ("M1",), ("M2",), 1.0, partition="fast"),
        _reaction("R6", ("M2",), ("M3",), 1.0, partition="fast"),
        _reaction("R7", ("M3",), ("M4",), 1.0, partition="fast"),
        _reaction("R8", ("M4",), ("M1",), 1.0, partition="fast"),
        _reaction("R9", ("Protein", "M1"), ("Protein", "M2"), 0.001, partition="slow", interface="R*"),
    ]
    expected = {
        "slow_reactions": ["R1", "R2", "R3", "R4", "R9"],
        "fast_reactions": ["R5", "R6", "R7", "R8"],
        "R_star": ["R9"],
        "S_star": [],
        "slow_species": ["DNA", "mRNA", "Protein"],
        "fast_species": ["M1", "M2", "M3", "M4"],
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


def _reaction(
    reaction_id: str,
    reactants: Sequence[str],
    products: Sequence[str],
    rate: float,
    *,
    partition: str,
    interface: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "reaction_id": str(reaction_id),
        "reactants": tuple(str(name) for name in reactants),
        "products": tuple(str(name) for name in products),
        "rate": float(rate),
        "expected_partition": str(partition),
        "interface": interface,
        "note": note,
    }


def _build_elementary_network(
    name: str,
    species: Sequence[str],
    initial_counts: dict[str, float],
    reactions: Sequence[dict[str, Any]],
) -> ElementaryMassActionNetwork:
    species_names = [str(item) for item in species]
    name_to_idx = {species_name: index for index, species_name in enumerate(species_names)}
    x0 = np.asarray([float(initial_counts.get(species_name, 0.0)) for species_name in species_names], dtype=float)
    nu_minus = []
    nu_plus = []
    rates = []
    labels = []
    for channel_id, reaction in enumerate(reactions):
        minus = np.zeros(len(species_names), dtype=float)
        plus = np.zeros(len(species_names), dtype=float)
        for species_name in reaction["reactants"]:
            minus[name_to_idx[str(species_name)]] += 1.0
        for species_name in reaction["products"]:
            plus[name_to_idx[str(species_name)]] += 1.0
        nu_minus.append(minus)
        nu_plus.append(plus)
        rates.append(float(reaction["rate"]))
        labels.append(
            {
                "channel_id": int(channel_id),
                "block_type": "ELEMENTARY",
                "reaction_id": reaction["reaction_id"],
                "expected_partition": reaction["expected_partition"],
                "interface": reaction["interface"],
                "note": reaction["note"],
            }
        )
    network = ElementaryMassActionNetwork(
        species_names=species_names,
        name_to_idx=name_to_idx,
        x0=x0,
        nu_minus=np.vstack(nu_minus),
        nu_plus=np.vstack(nu_plus),
        rate_constants=np.asarray(rates, dtype=float),
        reaction_labels=labels,
    )
    return network


def _benchmark_metadata(spec: NetworkSpec, expected_partition: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "manual_elementary_benchmark",
        "network": spec.name,
        "network_kind": spec.kind,
        "expected_partition": expected_partition,
    }


def network_spec(network_name: str) -> NetworkSpec:
    key = str(network_name).lower()
    if key not in NETWORK_SPECS:
        raise ValueError(f"unknown network {network_name!r}; available networks: {sorted(NETWORK_SPECS)}")
    return NETWORK_SPECS[key]


def assign_cross_terminal_catalysis(network: ReactionNetworkData, spec: NetworkSpec) -> dict[str, Any]:
    clear_all_catalysis(network, rebuild=False)
    channels_by_catalyst: dict[str, list[int]] = {}
    primary_channels_by_catalyst: dict[str, list[int]] = {}
    gamma_by_primary_channel: dict[int, float] = {}

    for catalyst_name, added_monomer_name in spec.cross_catalysis_rules:
        catalyst_sid = network.species_idx(catalyst_name)
        added_monomer_sid = network.species_idx(added_monomer_name)
        primary_channels = [
            int(channel_id)
            for channel_id in terminal_matched_addition_channels(network, added_monomer_sid, added_monomer_name)
        ]
        strengths = np.full(len(primary_channels), float(spec.catalytic_gamma), dtype=float)
        for channel_id in primary_channels:
            gamma_by_primary_channel[int(channel_id)] = float(spec.catalytic_gamma)
        set_catalytic_strengths_for_channels(
            network,
            np.asarray(primary_channels, dtype=np.int64),
            int(catalyst_sid),
            strengths,
            mirror_reverse=True,
            rebuild=False,
        )
        mirrored = [
            int(reverse_id)
            for channel_id in primary_channels
            for reverse_id in network.get_reverse_channel_ids(channel_id)
        ]
        primary_channels_by_catalyst[catalyst_name] = primary_channels
        channels_by_catalyst[catalyst_name] = sorted(set(primary_channels + mirrored))

    network.rebuild_dependency_indices()
    return {
        "method": "cross_terminal_matched_addition",
        "rules": dict(spec.cross_catalysis_rules),
        "gamma": float(spec.catalytic_gamma),
        "mirror_reverse": True,
        "primary_channels_by_catalyst": primary_channels_by_catalyst,
        "channels_by_catalyst": channels_by_catalyst,
        "gamma_by_primary_channel": gamma_by_primary_channel,
    }


def assign_two_stage_one_polymer_catalysis(network: ReactionNetworkData, spec: NetworkSpec) -> dict[str, Any]:
    clear_all_catalysis(network, rebuild=False)
    monomer_sid = network.species_idx("1")
    rules = (
        ("11111", tuple("1" * length for length in range(1, 5))),
        ("1111111111", tuple("1" * length for length in range(5, 10))),
    )
    channels_by_catalyst: dict[str, list[int]] = {}
    primary_channels_by_catalyst: dict[str, list[int]] = {}
    gamma_by_primary_channel: dict[int, float] = {}

    for catalyst_name, source_names in rules:
        catalyst_sid = network.species_idx(catalyst_name)
        primary_channels = [
            int(channel_id)
            for channel_id in exact_source_addition_channels(network, monomer_sid, source_names)
        ]
        strengths = np.full(len(primary_channels), float(spec.catalytic_gamma), dtype=float)
        for channel_id in primary_channels:
            gamma_by_primary_channel[int(channel_id)] = float(spec.catalytic_gamma)
        set_catalytic_strengths_for_channels(
            network,
            np.asarray(primary_channels, dtype=np.int64),
            int(catalyst_sid),
            strengths,
            mirror_reverse=True,
            rebuild=False,
        )
        mirrored = [
            int(reverse_id)
            for channel_id in primary_channels
            for reverse_id in network.get_reverse_channel_ids(channel_id)
        ]
        primary_channels_by_catalyst[catalyst_name] = primary_channels
        channels_by_catalyst[catalyst_name] = sorted(set(primary_channels + mirrored))

    network.rebuild_dependency_indices()
    return {
        "method": "two_stage_exact_one_polymer_addition",
        "rules": {
            catalyst_name: list(source_names)
            for catalyst_name, source_names in rules
        },
        "added_monomer": "1",
        "gamma": float(spec.catalytic_gamma),
        "mirror_reverse": True,
        "primary_channels_by_catalyst": primary_channels_by_catalyst,
        "channels_by_catalyst": channels_by_catalyst,
        "gamma_by_primary_channel": gamma_by_primary_channel,
    }


def exact_source_addition_channels(
    network: ReactionNetworkData,
    added_monomer_sid: int,
    source_names: Sequence[str],
) -> np.ndarray:
    source_set = {str(name) for name in source_names}
    channels: list[int] = []
    for local_id, monomer_sid in enumerate(network.left_add_monomer):
        if int(monomer_sid) != int(added_monomer_sid):
            continue
        polymer_sid = int(network.left_add_species[int(local_id)])
        if network.species_names[polymer_sid] in source_set:
            channels.append(network.channel_id(ChannelBlock.LEFT_ADD, int(local_id)))

    for local_id, monomer_sid in enumerate(network.right_add_monomer):
        if int(monomer_sid) != int(added_monomer_sid):
            continue
        polymer_sid = int(network.right_add_species[int(local_id)])
        if network.species_names[polymer_sid] in source_set:
            channels.append(network.channel_id(ChannelBlock.RIGHT_ADD, int(local_id)))

    return np.asarray(channels, dtype=np.int64)


def terminal_matched_addition_channels(
    network: ReactionNetworkData,
    added_monomer_sid: int,
    added_monomer_name: str,
) -> np.ndarray:
    channels: list[int] = []
    for local_id, monomer_sid in enumerate(network.left_add_monomer):
        if int(monomer_sid) != int(added_monomer_sid):
            continue
        polymer_sid = int(network.left_add_species[int(local_id)])
        if network.species_names[polymer_sid].startswith(added_monomer_name):
            channels.append(network.channel_id(ChannelBlock.LEFT_ADD, int(local_id)))

    for local_id, monomer_sid in enumerate(network.right_add_monomer):
        if int(monomer_sid) != int(added_monomer_sid):
            continue
        polymer_sid = int(network.right_add_species[int(local_id)])
        if network.species_names[polymer_sid].endswith(added_monomer_name):
            channels.append(network.channel_id(ChannelBlock.RIGHT_ADD, int(local_id)))

    return np.asarray(channels, dtype=np.int64)


def make_stepper(
    method: str,
    settings: RunSettings,
    network: ReactionNetworkData | ElementaryMassActionNetwork,
):
    name = normalize_method(method)
    if name == "gillespie_ssa":
        return SSAStepper(use_local_propensity_updates=isinstance(network, ReactionNetworkData)), None
    if name == "optimized_nrm":
        return OptimizedNRMStepper(), None
    if name == "gillespie_cle_hybrid":
        return BlendedHybridStepper(make_blended_config(settings)), None
    if name == "nrm_cle_hybrid":
        return NRMBlendedHybridStepper(make_blended_config(settings)), None
    if name == "gillespie_pdmp_lp":
        return make_pdmp_stepper(settings, discrete_event_method="gillespie", network=network), None
    if name == "nrm_pdmp_lp":
        return make_pdmp_stepper(settings, discrete_event_method="nrm_heap", network=network), None
    if name == STRICT_2018_PDMP_METHOD:
        return make_strict_2018_pdmp_stepper(settings, network), None
    raise ValueError(f"unknown method {method!r}")


def make_blended_config(settings: RunSettings) -> BlendedHybridConfig:
    return BlendedHybridConfig(
        i1=float(settings.blended_i1),
        i2=float(settings.blended_i2),
        dt_cle=float(settings.blended_dt_cle),
        dt_macro=float(settings.blended_dt_macro),
        use_reaction_interval_dt=False,
        reaction_interval_update_steps=1,
        beta_species_mode=str(settings.blended_beta_species_mode),
    )


def make_pdmp_stepper(
    settings: RunSettings,
    *,
    discrete_event_method: str,
    network: ReactionNetworkData | ElementaryMassActionNetwork,
) -> PDMPStepper:
    partition_config = ScalingPDMPConfig(
        N0=float(settings.pdmp_n0),
        continuous_copy_number_scale_threshold_mu=float(settings.pdmp_mu),
        adaptation_scale_threshold_eta=float(settings.pdmp_eta),
        reaction_relaxation_delta=float(settings.pdmp_delta),
        use_lp=True,
        enable_fast_subnetworks=bool(settings.pdmp_enable_fast_subnetworks),
        fast_subnetwork_threshold=float(settings.pdmp_fast_subnetwork_threshold),
        fast_subnetwork_max_size=int(settings.pdmp_fast_subnetwork_max_size),
    )
    partition_method = "linear_catalysis_scaling" if isinstance(network, ReactionNetworkData) else "scaling"
    stepper = PDMPStepper(
        partition_method=partition_method,
        partition_config=partition_config,
        config=PDMPConfig(
            ode_step=float(settings.pdmp_ode_step),
            adaptive=True,
            repartition_on_event=bool(settings.pdmp_repartition_on_event),
            repartition_on_bounds=bool(settings.pdmp_repartition_on_bounds),
            discrete_event_method=str(discrete_event_method),
            use_discrete_event_heap=str(discrete_event_method) == "nrm_heap",
            use_local_propensity_updates=bool(settings.pdmp_use_local_propensity_updates),
            local_propensity_full_recompute_fraction=float(settings.pdmp_local_propensity_full_recompute_fraction),
            heap_rebuild_factor=float(settings.pdmp_heap_rebuild_factor),
        ),
    )
    if isinstance(network, ReactionNetworkData) and not isinstance(
        stepper.partition_strategy,
        LinearCatalysisScalingPDMPPartitionStrategy,
    ):
        raise RuntimeError("PDMPStepper did not select linear_catalysis_scaling partition")
    return stepper


def make_strict_2018_pdmp_stepper(
    settings: RunSettings,
    network: ReactionNetworkData | ElementaryMassActionNetwork,
) -> PDMPStepper:
    if not isinstance(network, ElementaryMassActionNetwork):
        raise TypeError("strict_2018_pdmp requires an ElementaryMassActionNetwork standard SRN view")
    partition_config = ScalingPDMPConfig(
        N0=float(settings.pdmp_n0),
        continuous_copy_number_scale_threshold_mu=float(settings.pdmp_mu),
        adaptation_scale_threshold_eta=float(settings.pdmp_eta),
        reaction_relaxation_delta=float(settings.pdmp_delta),
        use_lp=True,
        enable_fast_subnetworks=True,
        fast_subnetwork_threshold=float(settings.pdmp_fast_subnetwork_threshold),
        fast_subnetwork_max_size=int(settings.pdmp_fast_subnetwork_max_size),
    )
    finite_config = FiniteMarkovConfig(
        max_states=int(settings.pdmp_finite_markov_max_states),
        max_total_internal_count=int(settings.pdmp_finite_markov_max_total_internal_count),
    )
    return PDMPStepper(
        partition_strategy=FiniteMarkovScalingPDMPPartitionStrategy(
            partition_config,
            finite_config,
        ),
        partition_method="scaling",
        partition_config=partition_config,
        config=PDMPConfig(
            ode_step=float(settings.pdmp_ode_step),
            adaptive=True,
            # Algorithm 2 reruns adaptation on RQ events or scale-bound
            # violations; it does not force repartition after every discrete
            # event.
            repartition_on_event=False,
            repartition_on_bounds=True,
            discrete_event_method="gillespie",
            use_discrete_event_heap=False,
            use_local_propensity_updates=bool(settings.pdmp_use_local_propensity_updates),
            local_propensity_full_recompute_fraction=float(settings.pdmp_local_propensity_full_recompute_fraction),
            heap_rebuild_factor=float(settings.pdmp_heap_rebuild_factor),
        ),
    )


def normalize_method(method: str) -> str:
    aliases = {
        "ssa": "gillespie_ssa",
        "gillespie": "gillespie_ssa",
        "gillespie_ssa": "gillespie_ssa",
        "nrm": "optimized_nrm",
        "optimized_nrm": "optimized_nrm",
        "blended": "gillespie_cle_hybrid",
        "gillespie_cle_hybrid": "gillespie_cle_hybrid",
        "nrm_blended": "nrm_cle_hybrid",
        "nrm_cle_hybrid": "nrm_cle_hybrid",
        "gillespie_pdmp_lp": "gillespie_pdmp_lp",
        "pdmp_gillespie": "gillespie_pdmp_lp",
        "nrm_pdmp_lp": "nrm_pdmp_lp",
        "pdmp_nrm": "nrm_pdmp_lp",
        "strict_2018_pdmp": STRICT_2018_PDMP_METHOD,
        "pdmp_2018_strict": STRICT_2018_PDMP_METHOD,
        "strict_pdmp_2018": STRICT_2018_PDMP_METHOD,
        "paper_2018_pdmp": STRICT_2018_PDMP_METHOD,
    }
    key = str(method).lower()
    if key not in aliases:
        raise ValueError(f"unknown method {method!r}; available methods: {list(METHOD_ORDER)}")
    return aliases[key]


def should_skip_method_network(method: str, network_name: str) -> bool:
    method_key = normalize_method(method)
    return method_key == STRICT_2018_PDMP_METHOD and str(network_name) in STRICT_2018_PDMP_SKIP_NETWORKS


def strict_2018_skip_reason(network_name: str) -> str:
    skipped = ", ".join(STRICT_2018_PDMP_SKIP_NETWORKS)
    return (
        f"{STRICT_2018_PDMP_METHOD} skips compare default polymer benchmarks "
        f"{skipped}; requested network={network_name!r}. "
        "Edit STRICT_2018_PDMP_SKIP_NETWORKS in examples/compare/common.py to include this run."
    )


def run_method(
    method: str,
    network_name: str,
    settings: RunSettings,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    write_json: bool = True,
) -> dict[str, Any]:
    method_key = normalize_method(method)
    if should_skip_method_network(method_key, network_name):
        return skipped_run_record(
            method_key,
            network_name,
            settings,
            reason=strict_2018_skip_reason(network_name),
            output_dir=output_dir,
            write_json=write_json,
        )
    build_started = perf_counter()
    network, catalysis_result, spec = build_network(network_name)
    network, catalysis_result = prepare_network_for_method(method_key, network, catalysis_result)
    build_wall_seconds = perf_counter() - build_started
    stepper, dt = make_stepper(method_key, settings, network)

    run_started = perf_counter()
    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=_runner_t_end(settings.t_end),
        seed=int(settings.seed),
        dt=dt,
        max_steps=int(settings.max_steps),
        max_runtime_seconds=float(settings.max_runtime_seconds),
        network_build_elapsed_seconds=build_wall_seconds,
    )
    run_wall_seconds = perf_counter() - run_started
    summary = result.summary
    final_state = np.asarray(summary.final_state, dtype=float)
    record = {
        "network": spec.name,
        "method": method_key,
        "script": f"{method_key}.py",
        "seed": int(settings.seed),
        "requested_t_end": settings.t_end,
        "max_runtime_seconds": float(settings.max_runtime_seconds),
        "simulation_final_time": float(summary.final_time),
        "wall_runtime_seconds": float(run_wall_seconds),
        "network_build_wall_seconds": float(build_wall_seconds),
        "n_steps": int(summary.n_steps),
        "n_events": int(summary.n_events),
        "stop_reason": summary.metadata.get("stop_reason"),
        "stepper_name": summary.metadata.get("stepper_name"),
        "n_species": int(network.n_species),
        "n_channels": int(network.n_channels),
        "final_total_abundance": float(final_state.sum()),
        "max_species_count": float(final_state.max()) if final_state.size else 0.0,
        "network_spec": asdict(spec),
        "run_settings": asdict(settings),
        "catalysis_assignment": json_ready(catalysis_result),
    }
    if write_json:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{spec.name}_{method_key}.json"
        path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
        record["result_json_path"] = str(path)
    return record


def skipped_run_record(
    method_key: str,
    network_name: str,
    settings: RunSettings,
    *,
    reason: str,
    output_dir: Path | str,
    write_json: bool,
) -> dict[str, Any]:
    record = {
        "network": str(network_name),
        "method": normalize_method(method_key),
        "script": f"{normalize_method(method_key)}.py",
        "seed": int(settings.seed),
        "requested_t_end": settings.t_end,
        "max_runtime_seconds": float(settings.max_runtime_seconds),
        "simulation_final_time": 0.0,
        "wall_runtime_seconds": 0.0,
        "network_build_wall_seconds": 0.0,
        "n_steps": 0,
        "n_events": 0,
        "stop_reason": "skipped",
        "skip_reason": str(reason),
        "stepper_name": None,
        "n_species": 0,
        "n_channels": 0,
        "final_total_abundance": 0.0,
        "max_species_count": 0.0,
        "network_spec": asdict(network_spec(network_name)),
        "run_settings": asdict(settings),
        "catalysis_assignment": {},
    }
    if write_json:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{network_name}_{normalize_method(method_key)}.json"
        path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
        record["result_json_path"] = str(path)
    return record


def run_method_profiled(
    method: str,
    network_name: str,
    settings: RunSettings,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    profile_dir: Path | str | None = None,
    profile_limit: int = 40,
    write_json: bool = False,
) -> tuple[dict[str, Any], str]:
    method_key = normalize_method(method)
    out_dir = Path(output_dir)
    prof_dir = Path(profile_dir) if profile_dir is not None else out_dir / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{network_name}_{method_key}"
    profile_path = prof_dir / f"{stem}.prof"
    report_path = prof_dir / f"{stem}_top{int(profile_limit)}.txt"

    profiler = cProfile.Profile()
    record = profiler.runcall(
        run_method,
        method_key,
        network_name,
        settings,
        output_dir=out_dir,
        write_json=bool(write_json),
    )
    profiler.dump_stats(profile_path)

    stream = io.StringIO()
    stats = pstats.Stats(str(profile_path), stream=stream)
    stats.strip_dirs().sort_stats("cumtime").print_stats(int(profile_limit))
    profile_entries = _profile_entries(stats, int(profile_limit))
    report_text = stream.getvalue()
    report_path.write_text(report_text, encoding="utf-8")

    record["profile_prof_path"] = str(profile_path)
    record["profile_report_path"] = str(report_path)
    record["profile_top"] = profile_entries
    if write_json:
        _rewrite_result_json(record)
    return record, report_text


def run_comparison(
    *,
    networks: Sequence[str] = DEFAULT_NETWORKS,
    methods: Sequence[str] = METHOD_ORDER,
    settings: RunSettings = RunSettings(),
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    profile_dir: Path | str | None = None,
    profile_limit: int = 40,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for network_name in networks:
        for method in methods:
            print(f"[compare] network={network_name} method={normalize_method(method)}")
            record, _profile_text = run_method_profiled(
                method,
                network_name,
                settings,
                output_dir=output_dir,
                profile_dir=profile_dir,
                profile_limit=int(profile_limit),
                write_json=False,
            )
            print(
                "[compare] "
                f"virtual_time={record['simulation_final_time']:.6g} "
                f"wall={record['wall_runtime_seconds']:.3f}s "
                f"steps={record['n_steps']} events={record['n_events']} "
                f"stop={record['stop_reason']}"
            )
            records.append(record)
    write_profile_report(records, output_dir, profile_limit=int(profile_limit))
    return records


def write_tables(records: Sequence[dict[str, Any]], output_dir: Path | str) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_json = out_dir / "method_run_records.json"
    records_csv = out_dir / "method_run_records.csv"
    pivot_csv = out_dir / "virtual_time_table.csv"
    pivot_md = out_dir / "virtual_time_table.md"

    records_json.write_text(json.dumps(list(records), ensure_ascii=True, indent=2), encoding="utf-8")
    if records:
        fieldnames = [
            "network",
            "method",
            "simulation_final_time",
            "wall_runtime_seconds",
            "network_build_wall_seconds",
            "n_steps",
            "n_events",
            "stop_reason",
            "stepper_name",
            "n_species",
            "n_channels",
        ]
        with records_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({key: record.get(key) for key in fieldnames})

    networks = sorted({str(record["network"]) for record in records})
    methods = [method for method in METHOD_ORDER if any(record["method"] == method for record in records)]
    by_pair = {(record["network"], record["method"]): record for record in records}
    with pivot_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["network", *methods])
        for network in networks:
            writer.writerow(
                [
                    network,
                    *[
                        _format_float(by_pair.get((network, method), {}).get("simulation_final_time"))
                        for method in methods
                    ],
                ]
            )
    pivot_md.write_text(_markdown_virtual_time_table(networks, methods, by_pair), encoding="utf-8")
    print(f"[compare] wrote records: {records_json}")
    print(f"[compare] wrote records csv: {records_csv}")
    print(f"[compare] wrote virtual-time table: {pivot_csv}")
    print(f"[compare] wrote virtual-time markdown: {pivot_md}")
    return {
        "records_json": records_json,
        "records_csv": records_csv,
        "virtual_time_csv": pivot_csv,
        "virtual_time_md": pivot_md,
    }


def write_simulation_summary_tables(records: Sequence[dict[str, Any]], output_dir: Path | str) -> dict[str, Path]:
    """Write mandatory simulation outcome tables for batch comparisons.

    The accepted input is either the raw records returned by run_method/profiled
    or the "runs" list from profile_top*_report.json.  Raw records store run
    fields at top level; profile report records store them under "simulation".
    """

    report_dir = Path(output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = report_dir / "simulation_summary_table.csv"
    summary_md = report_dir / "simulation_summary_table.md"
    summary_records = [_simulation_summary_record(record) for record in records]

    fieldnames = [
        "max_runtime_seconds",
        "network",
        "method",
        "simulation_final_time",
        "wall_runtime_seconds",
        "n_steps",
        "n_events",
        "stop_reason",
        "stepper_name",
        "n_species",
        "n_channels",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in summary_records:
            writer.writerow({key: record.get(key) for key in fieldnames})

    summary_md.write_text(_markdown_simulation_summary(summary_records), encoding="utf-8")
    print(f"[compare] wrote simulation summary csv: {summary_csv}")
    print(f"[compare] wrote simulation summary markdown: {summary_md}")
    return {"csv": summary_csv, "markdown": summary_md}


def write_profile_report(
    records: Sequence[dict[str, Any]],
    output_dir: Path | str,
    *,
    profile_limit: int = 40,
) -> dict[str, Path]:
    report_dir = Path(output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / f"profile_top{int(profile_limit)}_report.json"
    report_csv = report_dir / f"profile_top{int(profile_limit)}_report.csv"
    report_md = report_dir / f"profile_top{int(profile_limit)}_report.md"

    payload = {
        "profile_sort": "cumtime",
        "profile_limit": int(profile_limit),
        "n_runs": int(len(records)),
        "runs": [_profile_report_record(record) for record in records],
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    fieldnames = [
        "max_runtime_seconds",
        "network",
        "method",
        "rank",
        "primitive_calls",
        "total_calls",
        "tottime",
        "percall_tottime",
        "cumtime",
        "percall_cumtime",
        "function",
    ]
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for entry in record.get("profile_top", []):
                writer.writerow(
                    {
                        "network": record.get("network"),
                        "method": record.get("method"),
                        "max_runtime_seconds": record.get("max_runtime_seconds"),
                        **entry,
                    }
                )

    report_md.write_text(_markdown_profile_report(payload["runs"], int(profile_limit)), encoding="utf-8")
    print(f"[compare] wrote cProfile json report: {report_json}")
    print(f"[compare] wrote cProfile csv report: {report_csv}")
    print(f"[compare] wrote cProfile markdown report: {report_md}")
    return {
        "json": report_json,
        "csv": report_csv,
        "markdown": report_md,
    }


def main_single(method: str) -> None:
    args = _single_parser(method).parse_args()
    method_key = normalize_method(method)
    networks = tuple(args.networks) if args.networks else tuple(DEFAULT_NETWORKS)
    settings = RunSettings(
        seed=int(args.seed),
        t_end=_parse_optional_float(args.t_end),
        max_steps=int(args.max_steps),
        max_runtime_seconds=float(args.wall_seconds),
        blended_i1=float(args.blended_i1),
        blended_i2=float(args.blended_i2),
        blended_dt_cle=float(args.blended_dt_cle),
        blended_dt_macro=float(args.blended_dt_macro),
        pdmp_ode_step=float(args.pdmp_ode_step),
    )
    records: list[dict[str, Any]] = []
    print(f"[single method run] method={method_key} networks={list(networks)} wall_seconds={settings.max_runtime_seconds}")
    for network_name in networks:
        print(f"\n[run] network={network_name} method={method_key}")
        if bool(args.profile):
            record, profile_text = run_method_profiled(
                method_key,
                network_name,
                settings,
                output_dir=args.output_dir,
                profile_dir=args.profile_dir,
                profile_limit=int(args.profile_limit),
                write_json=False,
            )
        else:
            record = run_method(method_key, network_name, settings, output_dir=args.output_dir, write_json=False)
            profile_text = ""

        records.append(record)
        _print_single_run_summary(record)
        if profile_text:
            print(f"\n[profile top {int(args.profile_limit)} by cumulative time]")
            print(profile_text.rstrip())


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def _single_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {normalize_method(method)} on a shared compare network.")
    parser.add_argument(
        "networks",
        nargs="*",
        choices=sorted(NETWORK_SPECS),
        help="Networks to run. Omit this argument to run all DEFAULT_NETWORKS.",
    )
    parser.add_argument("--wall-seconds", type=float, default=DEFAULT_SETTINGS.max_runtime_seconds)
    parser.add_argument("--seed", type=int, default=DEFAULT_SETTINGS.seed)
    parser.add_argument("--t-end", default="none", help="Use 'none' to stop by wall-clock/max-steps.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_SETTINGS.max_steps)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--blended-i1", type=float, default=DEFAULT_SETTINGS.blended_i1)
    parser.add_argument("--blended-i2", type=float, default=DEFAULT_SETTINGS.blended_i2)
    parser.add_argument("--blended-dt-cle", type=float, default=DEFAULT_SETTINGS.blended_dt_cle)
    parser.add_argument("--blended-dt-macro", type=float, default=DEFAULT_SETTINGS.blended_dt_macro)
    parser.add_argument("--pdmp-ode-step", type=float, default=DEFAULT_SETTINGS.pdmp_ode_step)
    parser.add_argument("--profile", dest="profile", action="store_true", default=True)
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--profile-limit", type=int, default=40)
    return parser


def _parse_optional_float(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "inf", "infinity"}:
        return None
    return float(text)


def _runner_t_end(value: float | None) -> float:
    return float("inf") if value is None else float(value)


def _format_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def _profile_entries(stats: pstats.Stats, limit: int) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    functions = list(stats.fcn_list or [])
    for rank, func in enumerate(functions[: int(limit)], start=1):
        primitive_calls, total_calls, tottime, cumtime, _callers = stats.stats[func]
        filename, line_no, function_name = func
        primitive = int(primitive_calls)
        total = int(total_calls)
        entries.append(
            {
                "rank": int(rank),
                "primitive_calls": primitive,
                "total_calls": total,
                "tottime": float(tottime),
                "percall_tottime": None if total <= 0 else float(tottime / total),
                "cumtime": float(cumtime),
                "percall_cumtime": None if primitive <= 0 else float(cumtime / primitive),
                "function": f"{filename}:{int(line_no)}({function_name})",
            }
        )
    return entries


def _profile_report_record(record: dict[str, Any]) -> dict[str, object]:
    return {
        "network": record.get("network"),
        "method": record.get("method"),
        "profile_prof_path": record.get("profile_prof_path"),
        "profile_report_path": record.get("profile_report_path"),
        "simulation": {
            "max_runtime_seconds": record.get("max_runtime_seconds"),
            "simulation_final_time": record.get("simulation_final_time"),
            "wall_runtime_seconds": record.get("wall_runtime_seconds"),
            "network_build_wall_seconds": record.get("network_build_wall_seconds"),
            "n_steps": record.get("n_steps"),
            "n_events": record.get("n_events"),
            "stop_reason": record.get("stop_reason"),
            "stepper_name": record.get("stepper_name"),
            "n_species": record.get("n_species"),
            "n_channels": record.get("n_channels"),
        },
        "profile_top": list(record.get("profile_top", [])),
    }


def _markdown_virtual_time_table(
    networks: Sequence[str],
    methods: Sequence[str],
    by_pair: dict[tuple[str, str], dict[str, Any]],
) -> str:
    lines = [
        "| network | " + " | ".join(methods) + " |",
        "|---|" + "|".join("---" for _ in methods) + "|",
    ]
    for network in networks:
        values = [
            _format_float(by_pair.get((network, method), {}).get("simulation_final_time"))
            for method in methods
        ]
        lines.append("| " + network + " | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _markdown_simulation_summary(records: Sequence[dict[str, Any]]) -> str:
    networks = _ordered_unique(str(record["network"]) for record in records if record.get("network") is not None)
    observed_methods = _ordered_unique(str(record["method"]) for record in records if record.get("method") is not None)
    methods = [method for method in METHOD_ORDER if method in observed_methods]
    methods.extend(method for method in observed_methods if method not in methods)
    wall_budgets = _ordered_unique(
        _format_summary_value(record.get("max_runtime_seconds"))
        for record in records
        if record.get("max_runtime_seconds") is not None
    )
    if not wall_budgets:
        wall_budgets = [""]
    by_wall_pair = {
        (
            _format_summary_value(record.get("max_runtime_seconds")),
            str(record["network"]),
            str(record["method"]),
        ): record
        for record in records
        if record.get("network") is not None and record.get("method") is not None
    }

    lines = ["# Simulation Summary Table", ""]
    for wall_budget in wall_budgets:
        title = "unspecified" if wall_budget == "" else f"{wall_budget} seconds"
        scoped = {
            (network, method): record
            for (budget, network, method), record in by_wall_pair.items()
            if budget == wall_budget
        }
        lines.extend(
            [
                f"## Wall Budget: {title}",
                "",
                "### Simulation Final Time",
                "",
                _markdown_metric_pivot(networks, methods, scoped, "simulation_final_time"),
                "",
                "### Event Count",
                "",
                _markdown_metric_pivot(networks, methods, scoped, "n_events"),
                "",
                "### Step Count",
                "",
                _markdown_metric_pivot(networks, methods, scoped, "n_steps"),
                "",
            ]
        )
    lines.extend([
        "## Flat Summary",
        "",
        "| max_runtime_seconds | network | method | simulation_final_time | wall_runtime_seconds | n_steps | n_events | stop_reason |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ])
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_summary_value(record.get("max_runtime_seconds")),
                    str(record.get("network")),
                    str(record.get("method")),
                    _format_summary_value(record.get("simulation_final_time")),
                    _format_summary_value(record.get("wall_runtime_seconds")),
                    _format_summary_value(record.get("n_steps")),
                    _format_summary_value(record.get("n_events")),
                    str(record.get("stop_reason")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_metric_pivot(
    networks: Sequence[str],
    methods: Sequence[str],
    by_pair: dict[tuple[str, str], dict[str, Any]],
    metric: str,
) -> str:
    lines = [
        "| network | " + " | ".join(methods) + " |",
        "|---|" + "|".join("---:" for _ in methods) + "|",
    ]
    for network in networks:
        values = [
            _format_summary_value(by_pair.get((network, method), {}).get(metric))
            for method in methods
        ]
        lines.append("| " + network + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _simulation_summary_record(record: dict[str, Any]) -> dict[str, Any]:
    simulation = record.get("simulation", {})
    if not isinstance(simulation, dict):
        simulation = {}

    def pick(key: str) -> Any:
        if key in record:
            return record.get(key)
        return simulation.get(key)

    return {
        "network": pick("network"),
        "method": pick("method"),
        "max_runtime_seconds": pick("max_runtime_seconds"),
        "simulation_final_time": pick("simulation_final_time"),
        "wall_runtime_seconds": pick("wall_runtime_seconds"),
        "n_steps": pick("n_steps"),
        "n_events": pick("n_events"),
        "stop_reason": pick("stop_reason"),
        "stepper_name": pick("stepper_name"),
        "n_species": pick("n_species"),
        "n_channels": pick("n_channels"),
    }


def _ordered_unique(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _format_summary_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and float(value).is_integer() and abs(float(value)) >= 1000.0:
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _format_float(float(value))
    return str(value)


def _markdown_profile_report(runs: Sequence[dict[str, object]], profile_limit: int) -> str:
    lines = [f"# cProfile Top {int(profile_limit)} Report", ""]
    for run in runs:
        simulation = run.get("simulation", {})
        if not isinstance(simulation, dict):
            simulation = {}
        lines.extend(
            [
                f"## {run.get('network')} / {run.get('method')} / wall={_format_float(simulation.get('max_runtime_seconds'))}s",
                "",
                f"- max_runtime_seconds: {_format_float(simulation.get('max_runtime_seconds'))}",
                f"- simulation_final_time: {_format_float(simulation.get('simulation_final_time'))}",
                f"- wall_runtime_seconds: {_format_float(simulation.get('wall_runtime_seconds'))}",
                f"- n_steps: {simulation.get('n_steps')}",
                f"- n_events: {simulation.get('n_events')}",
                f"- stop_reason: {simulation.get('stop_reason')}",
                f"- profile_report: {run.get('profile_report_path')}",
                "",
                "| rank | primitive calls | total calls | tottime | cumtime | function |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        profile_top = run.get("profile_top", [])
        if not isinstance(profile_top, list):
            profile_top = []
        for entry in profile_top:
            if not isinstance(entry, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(entry.get("rank")),
                        str(entry.get("primitive_calls")),
                        str(entry.get("total_calls")),
                        _format_float(entry.get("tottime")),
                        _format_float(entry.get("cumtime")),
                        str(entry.get("function")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "network",
        "method",
        "simulation_final_time",
        "wall_runtime_seconds",
        "network_build_wall_seconds",
        "n_steps",
        "n_events",
        "stop_reason",
        "stepper_name",
        "n_species",
        "n_channels",
        "profile_prof_path",
        "profile_report_path",
    ]
    return {key: record.get(key) for key in keys}


def _print_single_run_summary(record: dict[str, Any]) -> None:
    print("[simulation]")
    print(f"  network: {record.get('network')}")
    print(f"  method: {record.get('method')}")
    print(f"  stepper: {record.get('stepper_name')}")
    print(f"  seed: {record.get('seed')}")
    print(f"  stop_reason: {record.get('stop_reason')}")
    if record.get("skip_reason"):
        print(f"  skip_reason: {record.get('skip_reason')}")
    print(f"  simulation_final_time: {_format_float(record.get('simulation_final_time'))}")
    print(f"  wall_runtime_seconds: {_format_float(record.get('wall_runtime_seconds'))}")
    print(f"  network_build_wall_seconds: {_format_float(record.get('network_build_wall_seconds'))}")
    print(f"  n_steps: {record.get('n_steps')}")
    print(f"  n_events: {record.get('n_events')}")
    print(f"  n_species: {record.get('n_species')}")
    print(f"  n_channels: {record.get('n_channels')}")
    if record.get("result_json_path"):
        print(f"  result_json: {record.get('result_json_path')}")
    if record.get("profile_prof_path"):
        print(f"  profile_prof: {record.get('profile_prof_path')}")
    if record.get("profile_report_path"):
        print(f"  profile_report: {record.get('profile_report_path')}")


def _rewrite_result_json(record: dict[str, Any]) -> None:
    path_value = record.get("result_json_path")
    if not path_value:
        return
    path = Path(str(path_value))
    path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
