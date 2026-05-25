import numpy as np

from polymer_sim import (
    ExperimentRunner,
    FoodReplenishmentRestriction,
    RestrictionController,
    SSAStepper,
    TrajectoryRecorder,
    TrimerOutflowRestriction,
    build_restriction,
    build_n3_wh_network,
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
