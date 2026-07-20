import numpy as np
import pytest

from polymer_sim import (
    ChannelBlock,
    ExperimentRunner,
    OptimizedNRMStepper,
    ReactionNetworkData,
    StepperContext,
    SystemState,
    build_reaction_rule_tables,
    generate_fixed_species_space,
)


def test_optimized_nrm_first_event_matches_independent_exponential_schedule():
    space = generate_fixed_species_space(["A", "B"], max_len=1)
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_inflow=[2.0, 3.0],
        inflow_species_ids=[space.idx("A"), space.idx("B")],
    )
    manual_rng = np.random.default_rng(7)
    expected_times = np.asarray(
        [
            manual_rng.exponential(1.0 / 2.0),
            manual_rng.exponential(1.0 / 3.0),
        ],
        dtype=float,
    )
    expected_local = int(np.argmin(expected_times))
    expected_channel = network.channel_id(ChannelBlock.INFLOW, expected_local)

    state = SystemState.from_x0(network.x0)
    stepper = OptimizedNRMStepper()
    result = stepper.step(
        state,
        10.0,
        StepperContext(network=network, rng=np.random.default_rng(7)),
    )

    assert result.event_occurred
    assert result.channel_id == expected_channel
    assert result.tau == pytest.approx(float(expected_times[expected_local]))
    assert state.t == pytest.approx(float(expected_times[expected_local]))
    assert state.x[expected_local] == 1.0


def test_optimized_nrm_runner_uses_dependency_graph_and_local_propensity_update():
    space = generate_fixed_species_space(["A", "B"], max_len=3, initial_counts={"A": 20, "B": 20})
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.002,
        k_poly_right=0.002,
        k_frag_left=0.02,
        k_frag_right=0.02,
    )
    state = SystemState.from_x0(network.x0)
    stepper = OptimizedNRMStepper()
    result = stepper.step(
        state,
        100.0,
        StepperContext(network=network, rng=np.random.default_rng(11)),
    )

    assert result.event_occurred
    assert result.details["mode"] == "optimized_nrm"
    assert result.details["dependency_graph_used"] is True
    assert result.details["full_recompute"] is False
    assert stepper._propensities is not None
    assert np.allclose(stepper._propensities, network.compute_all_propensities(state))


def test_optimized_nrm_runs_through_experiment_runner():
    space = generate_fixed_species_space(["A", "B"], max_len=3, initial_counts={"A": 20, "B": 20})
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.002,
        k_poly_right=0.002,
        k_frag_left=0.02,
        k_frag_right=0.02,
    )

    result = ExperimentRunner().run_one(
        network,
        OptimizedNRMStepper(),
        t_end=0.5,
        seed=5,
    )

    assert result.state.t == 0.5
    assert result.summary.n_steps >= 1
    assert result.summary.metadata["stop_reason"] == "reached_t_end"
