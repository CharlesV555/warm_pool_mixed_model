import sys
from pathlib import Path

import numpy as np

from polymer_sim import (
    ChannelBlock,
    ElementaryMassActionNetwork,
    FiniteMarkovScalingPDMPPartitionStrategy,
    ReactionNetworkData,
    ScalingPDMPConfig,
    SystemState,
)


COMPARE_DIR = Path(__file__).resolve().parents[1] / "examples" / "compare"
if str(COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARE_DIR))

import common  # noqa: E402


def test_strict_2018_method_is_registered_and_skips_requested_default_polymer_networks():
    assert common.normalize_method("pdmp_2018_strict") == "strict_2018_pdmp"
    assert common.should_skip_method_network("strict_2018_pdmp", common.DEFAULT_NETWORKS[3])
    assert common.should_skip_method_network("strict_2018_pdmp", common.DEFAULT_NETWORKS[4])

    record = common.run_method(
        "strict_2018_pdmp",
        common.DEFAULT_NETWORKS[3],
        common.RunSettings(max_runtime_seconds=0.001, max_steps=10),
        write_json=False,
    )

    assert record["stop_reason"] == "skipped"
    assert "skip" in record["skip_reason"].lower()
    assert record["n_steps"] == 0
    assert record["n_events"] == 0


def test_strict_2018_prepares_polymer_network_as_standard_elementary_srn():
    network, catalysis_result, _spec = common.build_network("linear_cross_len3")

    elementary, metadata = common.prepare_network_for_method(
        "strict_2018_pdmp",
        network,
        catalysis_result,
    )

    assert isinstance(elementary, ElementaryMassActionNetwork)
    assert metadata["strict_2018_srn"]["target"] == "ElementaryMassActionNetwork"
    assert metadata["strict_2018_srn"]["standard_zero_order_inflow"] is True
    assert metadata["strict_2018_srn"]["expanded_catalysis"] is True
    assert elementary.n_channels >= network.n_channels
    assert np.all(elementary.reaction_order <= 2)


def test_compare_constant_food_disables_reactionnetwork_inflow_and_configures_chemostat():
    network, _catalysis_result, spec = common.build_network("linear_cross_len3", food_supply_mode="constant")

    assert isinstance(network, ReactionNetworkData)
    assert spec.food_supply_mode == "constant"
    assert network.channel_sizes[ChannelBlock.INFLOW] == 0

    restriction = common.build_compare_food_restriction(network, spec)
    assert restriction is None
    assert network.has_chemostat_species


def test_compare_explicit_food_keeps_reactionnetwork_inflow_channels():
    network, _catalysis_result, spec = common.build_network("linear_cross_len3", food_supply_mode="explicit_inflow")

    assert isinstance(network, ReactionNetworkData)
    assert spec.food_supply_mode == "explicit_inflow"
    assert network.channel_sizes[ChannelBlock.INFLOW] == len(spec.food_species)
    assert common.build_compare_food_restriction(network, spec) is None


def test_compare_constant_food_omits_manual_elementary_food_io():
    network, metadata, spec = common.build_network(
        "polymer_food_dimer_inhibition_len3",
        food_supply_mode="constant",
    )

    assert isinstance(network, ElementaryMassActionNetwork)
    assert spec.food_supply_mode == "constant"
    assert metadata["expected_partition"]["explicit_food_inflow_reactions"] is False
    notes = [str(label.get("note", "")) for label in network.reaction_labels]
    assert not any(note.startswith("food inflow") or note.startswith("food outflow") for note in notes)
    assert common.build_compare_food_restriction(network, spec) is None
    assert network.has_chemostat_species


def test_finite_markov_scaling_partition_accepts_averageable_fast_subnetwork():
    network = ElementaryMassActionNetwork(
        species_names=["A", "B", "AB"],
        name_to_idx={"A": 0, "B": 1, "AB": 2},
        x0=np.array([2.0, 1.0, 0.0]),
        nu_minus=np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        nu_plus=np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        rate_constants=np.array([100.0, 100.0]),
    )
    strategy = FiniteMarkovScalingPDMPPartitionStrategy(
        ScalingPDMPConfig(
            N0=10.0,
            use_lp=False,
            enable_fast_subnetworks=True,
            fast_subnetwork_threshold=0.0,
            fast_subnetwork_max_size=2,
        )
    )

    partition = strategy.partition(network, SystemState.from_x0(network.x0))

    assert partition.metadata["method"] == "strict_2018_scaling_lp_finite_markov"
    assert partition.metadata["finite_markov_averageable_subnetwork_count"] >= 1
    assert partition.fast_subnetworks
