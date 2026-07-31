from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (  # noqa: E402
    ChannelBlock,
    ElementaryExpansionConfig,
    ElementaryMassActionNetwork,
    ExperimentRunner,
    FiniteMarkovConfig,
    FiniteMarkovScalingPDMPPartitionStrategy,
    PDMPConfig,
    PDMPStepper,
    ReactionNetworkData,
    ScalingPDMPConfig,
    TrajectoryRecorder,
    build_elementary_mass_action_network,
    build_reaction_rule_tables,
    clear_all_catalysis,
    format_stepper_info,
    generate_fixed_species_space,
    save_trajectory_record,
    set_catalytic_strengths_for_channels,
)

METHOD_NAME = "strict_2018_pdmp"
DEFAULT_NETWORK = "fast_dimerization"
DEFAULT_OUTPUT_DIR = EXAMPLES_DIR


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
    cross_catalysis_rules: tuple[tuple[str, str], ...] = (("0000", "0"),)


@dataclass(frozen=True, slots=True)
class Strict2018Settings:
    seed: int = 123
    t_end: float = 400.0
    max_steps: int = 100_000_000
    max_runtime_seconds: float | None = 60.0
    pdmp_ode_step: float = 0.001
    pdmp_n0: float = 500.0
    pdmp_mu: float = 1.0
    pdmp_eta: float = 0.9
    pdmp_delta: float = 0.9
    pdmp_fast_subnetwork_threshold: float = 1.0
    pdmp_fast_subnetwork_max_size: int = 3
    pdmp_use_local_propensity_updates: bool = True
    pdmp_local_propensity_full_recompute_fraction: float = 0.5
    pdmp_heap_rebuild_factor: float = 4.0
    pdmp_finite_markov_max_states: int = 4096
    pdmp_finite_markov_max_total_internal_count: int = 10_000


DEFAULT_SETTINGS = Strict2018Settings()


NETWORK_SPECS: dict[str, NetworkSpec] = {
    "fast_dimerization": NetworkSpec(name="fast_dimerization", kind="fast_dimerization"),
    "toggle_switch": NetworkSpec(name="toggle_switch", kind="toggle_switch"),
    "repressilator": NetworkSpec(name="repressilator", kind="repressilator"),
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


def build_network(network_name: str) -> tuple[ReactionNetworkData | ElementaryMassActionNetwork, dict[str, Any], NetworkSpec]:
    spec = network_spec(network_name)
    if spec.kind == "fast_dimerization":
        return (*build_fast_dimerization_network(spec), spec)
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
        raise ValueError(f"unknown network kind: {spec.kind!r}")
    return (*build_polymer_cross_catalysis_network(spec), spec)


def prepare_strict_2018_network(
    network: ReactionNetworkData | ElementaryMassActionNetwork,
    metadata: dict[str, Any],
) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    if isinstance(network, ElementaryMassActionNetwork):
        strict_metadata = dict(metadata)
        strict_metadata["strict_2018_srn"] = {
            "source": "already_elementary_mass_action",
            "standard_zero_order_inflow": True,
            "expanded_catalysis": False,
        }
        return network, strict_metadata

    elementary_config = ElementaryExpansionConfig(
        standard_zero_order_inflow=True,
        include_uncatalyzed_channels=True,
        expand_catalysis=True,
        catalyst_binding_rate=1.0,
        catalyst_unbinding_rate=1.0,
        catalytic_turnover_rate=None,
        catalytic_turnover_scale=1.0,
    )
    elementary = build_elementary_mass_action_network(network, elementary_config)
    strict_metadata = {
        "source_catalysis_assignment": json_ready(metadata),
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
    return elementary, strict_metadata


def make_strict_2018_pdmp_stepper(settings: Strict2018Settings) -> PDMPStepper:
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
            repartition_on_event=False,
            repartition_on_bounds=True,
            discrete_event_method="gillespie",
            use_discrete_event_heap=False,
            use_local_propensity_updates=bool(settings.pdmp_use_local_propensity_updates),
            local_propensity_full_recompute_fraction=float(settings.pdmp_local_propensity_full_recompute_fraction),
            heap_rebuild_factor=float(settings.pdmp_heap_rebuild_factor),
        ),
    )


def build_fast_dimerization_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
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
    initial = {"m_A": 10.0, "p_A": 500.0, "m_B": 0.0, "p_B": 0.0, "m_C": 0.0, "p_C": 0.0}
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


def build_polymer_food_dimer_inhibition_len3_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("0", "1", "00", "11", "000", "111")
    initial = {"0": 100.0, "1": 100.0, "00": 0.0, "11": 0.0, "000": 0.0, "111": 0.0}
    reactions = [
        _reaction("R1", (), ("0",), 1.0, partition="slow", note="food inflow 0"),
        _reaction("R2", (), ("1",), 1.0, partition="slow", note="food inflow 1"),
        _reaction("R3", ("0",), (), 0.1, partition="slow", note="food outflow 0"),
        _reaction("R4", ("1",), (), 0.1, partition="slow", note="food outflow 1"),
        _reaction("R5", ("0", "0"), ("00",), 5.0, partition="fast", note="0 dimerization"),
        _reaction("R6", ("1", "1"), ("11",), 5.0, partition="fast", note="1 dimerization"),
        _reaction("R7", ("00", "0"), ("000",), 10.0, partition="fast", note="0 elongation"),
        _reaction("R8", ("11", "1"), ("111",), 10.0, partition="fast", note="1 elongation"),
        _reaction("R9", ("00", "1"), ("00",), 20.0, partition="slow", interface="S*"),
        _reaction("R10", ("11", "0"), ("11",), 20.0, partition="slow", interface="S*"),
        _reaction("R11", ("00",), (), 0.01, partition="fast"),
        _reaction("R12", ("11",), (), 0.01, partition="fast"),
        _reaction("R13", ("000",), (), 0.1, partition="fast"),
        _reaction("R14", ("111",), (), 0.1, partition="fast"),
    ]
    expected = {
        "slow_reactions": ["R1", "R2", "R3", "R4", "R9", "R10"],
        "fast_reactions": ["R5", "R6", "R7", "R8", "R11", "R12", "R13", "R14"],
        "R_star": [],
        "S_star": ["00", "11"],
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


def build_quasi_disjoint_slow_fast_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("S1", "S2", "S3", "S4", "S5")
    initial = {"S1": 20.0, "S2": 20.0, "S3": 200.0, "S4": 100.0, "S5": 100.0}
    reactions = [
        _reaction("R1", (), ("S1",), 1.0, partition="slow"),
        _reaction("R2", ("S1",), (), 0.05, partition="slow"),
        _reaction("R3", (), ("S2",), 1.0, partition="slow"),
        _reaction("R4", ("S2",), (), 0.05, partition="slow"),
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
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


def build_shared_interface_species_network(spec: NetworkSpec) -> tuple[ElementaryMassActionNetwork, dict[str, Any]]:
    species = ("A", "B", "C", "D")
    initial = {"A": 20.0, "B": 0.0, "C": 500.0, "D": 500.0}
    reactions = [
        _reaction("R1", (), ("A",), 1.0, partition="slow"),
        _reaction("R2", ("A",), (), 0.05, partition="slow"),
        _reaction("R3", ("A", "C"), ("C",), 0.001, partition="slow", interface="S*"),
        _reaction("R4", (), ("C",), 100.0, partition="fast"),
        _reaction("R5", ("C",), ("D",), 1.0, partition="fast"),
        _reaction("R6", ("D",), ("C",), 1.0, partition="fast"),
    ]
    expected = {
        "slow_reactions": ["R1", "R2", "R3"],
        "fast_reactions": ["R4", "R5", "R6"],
        "R_star": [],
        "S_star": ["C"],
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
    }
    return _build_elementary_network(spec.name, species, initial, reactions), _benchmark_metadata(spec, expected)


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
    network, _metadata = build_polymer_cross_catalysis_network(
        NetworkSpec(
            name=spec.name,
            kind="polymer_cross",
            max_len=10,
            alphabet=spec.alphabet,
            initial_food_count=spec.initial_food_count,
            food_max_count=spec.food_max_count,
            k_left_add=spec.k_left_add,
            k_right_add=spec.k_right_add,
            k_left_split=spec.k_left_split,
            k_right_split=spec.k_right_split,
            k_nonfood_outflow=spec.k_nonfood_outflow,
            food_inflow_rate=spec.food_inflow_rate,
            food_inflow_hill_coefficient=spec.food_inflow_hill_coefficient,
            use_hill_capped_food_inflow=spec.use_hill_capped_food_inflow,
            catalysis_mode=spec.catalysis_mode,
            saturation_alpha=spec.saturation_alpha,
            catalytic_gamma=spec.catalytic_gamma,
            cross_catalysis_rules=(),
        )
    )
    return network, assign_two_stage_one_polymer_catalysis(network, spec)


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
        "rules": {catalyst_name: list(source_names) for catalyst_name, source_names in rules},
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
    return ElementaryMassActionNetwork(
        species_names=species_names,
        name_to_idx=name_to_idx,
        x0=x0,
        nu_minus=np.vstack(nu_minus),
        nu_plus=np.vstack(nu_plus),
        rate_constants=np.asarray(rates, dtype=float),
        reaction_labels=labels,
    )


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
        raise ValueError(f"unknown network {network_name!r}; available: {sorted(NETWORK_SPECS)}")
    return NETWORK_SPECS[key]


def run_strict_2018_pdmp(
    network_name: str,
    settings: Strict2018Settings,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Any]:
    build_started = perf_counter()
    source_network, network_metadata, spec = build_network(network_name)
    network, strict_metadata = prepare_strict_2018_network(source_network, network_metadata)
    build_elapsed = perf_counter() - build_started

    stepper = make_strict_2018_pdmp_stepper(settings)
    recorder = TrajectoryRecorder()
    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=float(settings.t_end),
        seed=int(settings.seed),
        recorder=recorder,
        max_steps=int(settings.max_steps),
        max_runtime_seconds=settings.max_runtime_seconds,
        timing_report=True,
        timing_report_dir="timing_reports",
        timing_report_name=f"{spec.name}_{METHOD_NAME}",
        network_build_elapsed_seconds=build_elapsed,
    )
    trajectory_record = recorder.finalize()
    trajectory_record.run_metadata["example_parameters"] = {
        "method": METHOD_NAME,
        "network": spec.name,
        "network_spec": json_ready(asdict(spec)),
        "settings": json_ready(asdict(settings)),
    }
    trajectory_record.run_metadata["network_metadata"] = json_ready(strict_metadata)
    trajectory_record.run_metadata["network"] = {
        "n_species": int(network.n_species),
        "n_channels": int(network.n_channels),
        "reaction_labels": [
            json_ready(network.describe_channel(channel_id))
            for channel_id in range(network.n_channels)
        ],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{spec.name}_{METHOD_NAME}_trajectory.npz"
    save_trajectory_record(output_path, trajectory_record)

    print(f"\n{METHOD_NAME} summary:")
    print(format_stepper_info(result.summary.metadata))
    print(
        f"network={spec.name}, t={result.summary.final_time:.4f}, "
        f"steps={result.summary.n_steps}, events={result.summary.n_events}, "
        f"seed={result.summary.metadata.get('seed')}, "
        f"stop_reason={result.summary.metadata.get('stop_reason')}"
    )
    print(
        f"trajectory points={trajectory_record.times.shape[0]}, "
        f"state shape={trajectory_record.states.shape}"
    )
    print(f"trajectory saved to: {output_path}")
    return output_path, result


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


def _optional_wall_seconds(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "inf", "infinity"}:
        return None
    return float(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a strict 2018-style PDMP example and save a trajectory.")
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_SPECS),
        default=DEFAULT_NETWORK,
        help="Network to run.",
    )
    parser.add_argument("--t-end", type=float, default=DEFAULT_SETTINGS.t_end)
    parser.add_argument("--seed", type=int, default=DEFAULT_SETTINGS.seed)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_SETTINGS.max_steps)
    parser.add_argument("--max-runtime-seconds", default=str(DEFAULT_SETTINGS.max_runtime_seconds))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pdmp-ode-step", type=float, default=DEFAULT_SETTINGS.pdmp_ode_step)
    parser.add_argument("--pdmp-n0", type=float, default=DEFAULT_SETTINGS.pdmp_n0)
    parser.add_argument("--pdmp-mu", type=float, default=DEFAULT_SETTINGS.pdmp_mu)
    parser.add_argument("--pdmp-eta", type=float, default=DEFAULT_SETTINGS.pdmp_eta)
    parser.add_argument("--pdmp-delta", type=float, default=DEFAULT_SETTINGS.pdmp_delta)
    parser.add_argument("--finite-max-states", type=int, default=DEFAULT_SETTINGS.pdmp_finite_markov_max_states)
    parser.add_argument(
        "--finite-max-total-internal-count",
        type=int,
        default=DEFAULT_SETTINGS.pdmp_finite_markov_max_total_internal_count,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = Strict2018Settings(
        seed=int(args.seed),
        t_end=float(args.t_end),
        max_steps=int(args.max_steps),
        max_runtime_seconds=_optional_wall_seconds(args.max_runtime_seconds),
        pdmp_ode_step=float(args.pdmp_ode_step),
        pdmp_n0=float(args.pdmp_n0),
        pdmp_mu=float(args.pdmp_mu),
        pdmp_eta=float(args.pdmp_eta),
        pdmp_delta=float(args.pdmp_delta),
        pdmp_finite_markov_max_states=int(args.finite_max_states),
        pdmp_finite_markov_max_total_internal_count=int(args.finite_max_total_internal_count),
    )
    run_strict_2018_pdmp(args.network, settings, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

# 