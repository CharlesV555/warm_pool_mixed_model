import numpy as np

from polymer_sim import (
    ExperimentRunner,
    FoodReplenishmentRestriction,
    FoodUpperLimitRestriction,
    RestrictionController,
    SSAStepper,
    TrajectoryRecorder,
    ChannelBlock,
    build_food_supply_restriction,
    TrimerOutflowRestriction,
    build_restriction,
    build_n3_wh_network,
    normalize_food_supply_mode,
)
from polymer_sim.core.state import SystemState
from polymer_sim.simulation.restriction import RestrictionContext
from polymer_sim.simulation.stepper import StepResult


def test_food_replenishment_sets_food_back_to_target():
    network = build_n3_wh_network(initial_counts={"0": 10.0, "1": 10.0})
    restriction = FoodReplenishmentRestriction(
        {
            network.species_idx("0"): 10.0,
            network.species_idx("1"): 10.0,
        }
    )
    state = SystemState.from_x0(network.x0)
    state.x[network.species_idx("0")] = 3.0
    state.x[network.species_idx("1")] = 4.0
    restriction.apply(
        state,
        0.1,
        RestrictionContext(network=network, rng=np.random.default_rng(1)),
        StepResult(advanced_time=0.1, event_occurred=False),
    )
    assert state.x[network.species_idx("0")] == 10.0
    assert state.x[network.species_idx("1")] == 10.0


def test_food_upper_limit_caps_food_without_replenishing():
    network = build_n3_wh_network(initial_counts={"0": 10.0, "1": 10.0})
    zero_sid = network.species_idx("0")
    one_sid = network.species_idx("1")
    restriction = FoodUpperLimitRestriction(
        {
            zero_sid: 10.0,
            one_sid: 10.0,
        }
    )
    state = SystemState.from_x0(network.x0)
    state.x[zero_sid] = 15.0
    state.x[one_sid] = 4.0
    restriction.apply(
        state,
        0.1,
        RestrictionContext(network=network, rng=np.random.default_rng(1)),
        StepResult(advanced_time=0.1, event_occurred=False),
    )

    assert state.x[zero_sid] == 10.0
    assert state.x[one_sid] == 4.0
    assert restriction.metadata()["food_upper_limits"]["max_counts"] == [10.0, 10.0]


def test_trimer_outflow_removes_trimers():
    network = build_n3_wh_network()
    aaa = network.species_idx("000")
    state = SystemState.from_x0(network.x0)
    state.x[aaa] = 20.0
    restriction = TrimerOutflowRestriction(rate=0.8, species_ids=[aaa])
    restriction.apply(
        state,
        1.0,
        RestrictionContext(network=network, rng=np.random.default_rng(2)),
        StepResult(advanced_time=1.0, event_occurred=False),
    )
    assert state.x[aaa] < 20.0


def test_runner_applies_restriction_and_keeps_food_present():
    network = build_n3_wh_network(initial_counts={"0": 10.0, "1": 10.0}, k_right_add=1.0, k_nonfood_outflow=0.8)
    recorder = TrajectoryRecorder()
    result = ExperimentRunner().run_one(
        network,
        SSAStepper(),
        t_end=0.5,
        seed=3,
        recorder=recorder,
        restriction=build_restriction(network, food_count=10.0),
    )
    trajectory = recorder.finalize()
    zero_sid = network.species_idx("0")
    one_sid = network.species_idx("1")
    assert result.summary.final_time == 0.5
    assert np.allclose(trajectory.states[:, zero_sid], 10.0)
    assert np.allclose(trajectory.states[:, one_sid], 10.0)


def test_food_supply_mode_helper_supports_explicit_and_constant_modes():
    network = build_n3_wh_network(initial_counts={"0": 10.0, "1": 10.0}, k_food_inflow=1.0)

    assert normalize_food_supply_mode("inflow") == "explicit_inflow"
    assert normalize_food_supply_mode("chemostat") == "constant"
    assert build_food_supply_restriction(network, mode="explicit_inflow") is None

    constant_restriction = build_food_supply_restriction(
        network,
        mode="constant",
        food_species=("0", "1"),
        food_count=10.0,
    )
    assert constant_restriction is None
    assert network.has_chemostat_species
    assert set(network.chemostat_species_ids.tolist()) == {network.species_idx("0"), network.species_idx("1")}


def test_constant_food_mode_runs_without_explicit_inflow_channels():
    network = build_n3_wh_network(
        initial_counts={"0": 10.0, "1": 10.0},
        k_right_add=1.0,
        k_nonfood_outflow=0.8,
        k_food_inflow=0.0,
    )
    assert network.channel_sizes[ChannelBlock.INFLOW] == 0

    restriction = build_food_supply_restriction(
        network,
        mode="constant",
        food_species=("0", "1"),
        food_count=10.0,
    )
    assert restriction is None

    recorder = TrajectoryRecorder()
    result = ExperimentRunner().run_one(
        network,
        SSAStepper(),
        t_end=0.5,
        seed=33,
        recorder=recorder,
        restriction=restriction,
    )
    trajectory = recorder.finalize()
    zero_sid = network.species_idx("0")
    one_sid = network.species_idx("1")

    assert result.summary.final_time == 0.5
    assert np.allclose(trajectory.states[:, zero_sid], 10.0)
    assert np.allclose(trajectory.states[:, one_sid], 10.0)
    assert "food_replenishment" not in trajectory.run_metadata


def test_constant_food_is_network_level_chemostat_not_state_projection():
    network = build_n3_wh_network(
        initial_counts={"0": 10.0, "1": 10.0},
        k_right_add=1.0,
        k_nonfood_outflow=0.0,
        k_food_inflow=0.0,
    )
    restriction = build_food_supply_restriction(
        network,
        mode="constant",
        food_species=("0", "1"),
        food_count=10.0,
    )
    assert restriction is None

    zero = network.species_idx("0")
    dimer = network.species_idx("00")
    channel_id = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[zero, zero]))
    state = SystemState.from_x0(network.x0)
    network.apply_channel_update(state, channel_id)

    assert state.x[zero] == 10.0
    assert state.x[dimer] == 1.0
    assert zero not in network.get_channel_changed_species(channel_id).tolist()
    assert network.affected_channels_for_species([zero]).size == 0

    state.x[zero] = 0.0
    assert network.compute_base_propensity(channel_id, state) == 45.0


def test_runner_can_stop_by_max_runtime_seconds():
    network = build_n3_wh_network(initial_counts={"0": 10.0, "1": 10.0}, k_right_add=1.0, k_nonfood_outflow=0.8)
    result = ExperimentRunner().run_one(
        network,
        SSAStepper(),
        t_end=10.0,
        seed=4,
        restriction=build_restriction(network, food_count=10.0),
        max_steps=1_000_000,
        max_runtime_seconds=0.001,
    )
    assert result.summary.metadata["stop_reason"] == "max_runtime_seconds"
    assert result.summary.final_time < 10.0
