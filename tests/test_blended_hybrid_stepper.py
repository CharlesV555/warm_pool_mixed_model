import numpy as np

from polymer_sim import (
    BlendedHybridConfig,
    BlendedHybridStepper,
    ChannelBlock,
    ExperimentRunner,
    NRMBlendedHybridStepper,
    ReactionNetworkData,
    StepperContext,
    SystemState,
    TrajectoryRecorder,
    build_reaction_rule_tables,
    generate_fixed_species_space,
)
from polymer_sim.simulation.stepper import _species_beta


def make_network(initial_count: float = 20.0) -> ReactionNetworkData:
    space = generate_fixed_species_space(
        ["A", "B"],
        max_len=3,
        initial_counts={"A": initial_count, "B": initial_count},
    )
    tables = build_reaction_rule_tables(space)
    return ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.002,
        k_poly_right=0.002,
        k_frag_left=0.02,
        k_frag_right=0.02,
    )


def test_species_beta_piecewise_linear():
    assert _species_beta(5.0, 10.0, 30.0) == 1.0
    assert _species_beta(30.0, 10.0, 30.0) == 0.0
    assert _species_beta(20.0, 10.0, 30.0) == 0.5


def test_blended_hybrid_pure_ssa_branch_keeps_state_valid():
    network = make_network()
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(BlendedHybridConfig(i1=1_000.0, i2=2_000.0, dt_cle=0.01))
    result = stepper.step(
        state,
        0.01,
        StepperContext(network=network, rng=np.random.default_rng(1)),
    )
    assert result.details["mode"] == "ssa"
    assert np.all(np.isfinite(state.x))
    assert np.all(state.x >= 0.0)


def test_blended_hybrid_pure_cle_branch_has_no_discrete_event():
    network = make_network()
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(BlendedHybridConfig(i1=-2.0, i2=-1.0, dt_cle=0.01))
    result = stepper.step(
        state,
        0.01,
        StepperContext(network=network, rng=np.random.default_rng(2)),
    )
    assert result.details["mode"] == "cle"
    assert result.channel_id is None
    assert not result.event_occurred
    assert np.all(np.isfinite(state.x))
    assert np.all(state.x >= 0.0)


def test_blended_hybrid_halves_cle_dt_when_trial_goes_negative():
    space = generate_fixed_species_space(["A"], max_len=1, initial_counts={"A": 1.0})
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_outflow=1_000.0,
        outflow_species_ids=[space.idx("A")],
    )
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(
        BlendedHybridConfig(
            i1=-2.0,
            i2=-1.0,
            dt_cle=1.0,
            cle_dt_min=1e-9,
            round_low_counts_after_cle=False,
        )
    )

    result = stepper.step(
        state,
        1.0,
        StepperContext(network=network, rng=np.random.default_rng(123)),
    )

    assert result.details["mode"] == "cle"
    assert result.details["cle_rejected_attempts"] > 0
    assert result.details["cle_accepted_dt"] < result.details["cle_requested_dt"]
    assert result.advanced_time == result.details["cle_accepted_dt"]
    assert state.t == result.advanced_time
    assert np.all(np.isfinite(state.x))
    assert np.all(state.x >= 0.0)


def test_blended_hybrid_doubles_cle_dt_without_discrete_interruption():
    space = generate_fixed_species_space(["A"], max_len=1, initial_counts={"A": 0.0})
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(space, tables)
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(
        BlendedHybridConfig(
            i1=-2.0,
            i2=-1.0,
            dt_cle=0.01,
            dt_macro=0.01,
            cle_dt_max=0.04,
            round_low_counts_after_cle=False,
        )
    )

    result = stepper.step(
        state,
        1.0,
        StepperContext(network=network, rng=np.random.default_rng(124)),
    )

    assert result.details["mode"] == "cle"
    assert result.details["cle_rejected_attempts"] == 0
    assert np.isclose(result.details["cle_accepted_dt"], 0.01)
    assert np.isclose(result.details["cle_dt_after"], 0.02)
    assert np.all(np.isfinite(state.x))
    assert np.all(state.x >= 0.0)


def test_blended_hybrid_mixed_branch_smoke():
    network = make_network(initial_count=20.0)
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.01))
    context = StepperContext(network=network, rng=np.random.default_rng(3))
    modes = set()
    for _ in range(5):
        result = stepper.step(state, 0.01, context)
        modes.add(result.details["mode"])
        assert np.all(np.isfinite(state.x))
        assert np.all(state.x >= 0.0)
    assert modes <= {"mixed_cle", "mixed_jump", "ssa", "cle"}


def test_nrm_blended_pure_cle_branch_has_no_discrete_event():
    network = make_network()
    state = SystemState.from_x0(network.x0)
    stepper = NRMBlendedHybridStepper(BlendedHybridConfig(i1=-2.0, i2=-1.0, dt_cle=0.01))
    result = stepper.step(
        state,
        0.01,
        StepperContext(network=network, rng=np.random.default_rng(20)),
    )
    assert result.details["mode"] == "nrm_blended_cle"
    assert result.channel_id is None
    assert not result.event_occurred
    assert np.all(np.isfinite(state.x))
    assert np.all(state.x >= 0.0)


def test_nrm_blended_pure_cle_uses_adaptive_dt_guard():
    space = generate_fixed_species_space(["A"], max_len=1, initial_counts={"A": 1.0})
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_outflow=1_000.0,
        outflow_species_ids=[space.idx("A")],
    )
    state = SystemState.from_x0(network.x0)
    stepper = NRMBlendedHybridStepper(
        BlendedHybridConfig(
            i1=-2.0,
            i2=-1.0,
            dt_cle=1.0,
            cle_dt_min=1e-9,
            round_low_counts_after_cle=False,
        )
    )

    result = stepper.step(
        state,
        1.0,
        StepperContext(network=network, rng=np.random.default_rng(125)),
    )

    assert result.details["mode"] == "nrm_blended_cle"
    assert result.details["cle_rejected_attempts"] > 0
    assert result.details["cle_accepted_dt"] < result.details["cle_requested_dt"]
    assert np.all(np.isfinite(state.x))
    assert np.all(state.x >= 0.0)


def test_nrm_blended_pure_nrm_branch_keeps_state_valid():
    network = make_network()
    state = SystemState.from_x0(network.x0)
    stepper = NRMBlendedHybridStepper(BlendedHybridConfig(i1=1_000.0, i2=2_000.0, dt_cle=0.01))
    result = stepper.step(
        state,
        0.01,
        StepperContext(network=network, rng=np.random.default_rng(21)),
    )
    assert result.details["mode"] == "nrm_blended_nrm"
    assert np.all(np.isfinite(state.x))
    assert np.all(state.x >= 0.0)


def test_nrm_blended_mixed_branch_schedules_multiple_discrete_events():
    space = generate_fixed_species_space(
        ["A"],
        max_len=1,
        initial_counts={"A": 20.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_outflow=200.0,
        outflow_species_ids=[space.idx("A")],
    )
    state = SystemState.from_x0(network.x0)
    stepper = NRMBlendedHybridStepper(BlendedHybridConfig(i1=10.0, i2=110.0, dt_cle=0.01))

    result = stepper.step(
        state,
        0.01,
        StepperContext(network=network, rng=np.random.default_rng(22)),
    )

    assert result.details["mode"] == "nrm_blended_mixed"
    assert result.details["n_scheduled_discrete_events"] > 1
    assert result.details["n_applied_discrete_events"] == len(result.details["discrete_event_ids"])
    assert result.details["n_applied_discrete_events"] == len(result.details["discrete_event_times"])
    assert state.event_count == result.details["n_applied_discrete_events"]
    assert state.t == 0.01
    assert np.all(np.isfinite(state.x))
    assert np.all(state.x >= 0.0)


def test_nrm_blended_runner_records_multiple_discrete_events():
    space = generate_fixed_species_space(
        ["A"],
        max_len=1,
        initial_counts={"A": 20.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_outflow=200.0,
        outflow_species_ids=[space.idx("A")],
    )
    recorder = TrajectoryRecorder()
    result = ExperimentRunner().run_one(
        network,
        NRMBlendedHybridStepper(BlendedHybridConfig(i1=10.0, i2=110.0, dt_cle=0.01)),
        t_end=0.01,
        seed=22,
        recorder=recorder,
        max_steps=1,
    )
    record = recorder.finalize()

    assert result.summary.n_events > 1
    assert len(record.run_metadata["channel_event_ids"]) == result.summary.n_events
    assert len(record.run_metadata["channel_event_times"]) == result.summary.n_events
    assert sum(record.run_metadata["channel_trigger_counts"]) == result.summary.n_events


def test_blended_hybrid_splits_outflow_channels_like_reactions():
    space = generate_fixed_species_space(
        ["A"],
        max_len=2,
        initial_counts={"AA": 20.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_outflow=2.0,
        outflow_species_ids=[space.idx("AA")],
    )
    source = network.species_idx("AA")
    outflow_channel = network.channel_id(ChannelBlock.OUTFLOW, int(network.outflow_local_id_by_source[source]))
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.01))

    beta = stepper._channel_betas(network, state.x)
    propensities = network.compute_all_propensities(state)
    nu = stepper._stoichiometry_matrix(network)

    assert network.get_channel_block(outflow_channel) == ChannelBlock.OUTFLOW
    assert propensities[outflow_channel] == 40.0
    assert beta[outflow_channel] == 0.5
    assert beta[outflow_channel] * propensities[outflow_channel] == 20.0
    assert (1.0 - beta[outflow_channel]) * propensities[outflow_channel] == 20.0
    assert nu[outflow_channel, source] == -1.0


def test_blended_hybrid_keeps_inflow_continuous():
    space = generate_fixed_species_space(
        ["A"],
        max_len=1,
        initial_counts={"A": 200.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_inflow=5.0,
        inflow_species_ids=[space.idx("A")],
    )
    target = network.species_idx("A")
    inflow_channel = network.channel_id(ChannelBlock.INFLOW, int(network.inflow_local_id_by_target[target]))
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.01))

    beta = stepper._channel_betas(network, state.x)
    propensities = network.compute_all_propensities(state)
    nu = stepper._stoichiometry_matrix(network)

    assert network.get_channel_block(inflow_channel) == ChannelBlock.INFLOW
    assert propensities[inflow_channel] == 5.0
    assert beta[inflow_channel] == 0.0
    assert beta[inflow_channel] * propensities[inflow_channel] == 0.0
    assert (1.0 - beta[inflow_channel]) * propensities[inflow_channel] == 5.0
    assert nu[inflow_channel, target] == 1.0


def test_blended_hybrid_rounds_changed_low_count_species_after_cle():
    space = generate_fixed_species_space(
        ["A"],
        max_len=1,
        initial_counts={"A": 0.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_inflow=40.0,
        inflow_species_ids=[space.idx("A")],
    )
    target = network.species_idx("A")
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.1))

    result = stepper.step(
        state,
        1.0,
        StepperContext(network=network, rng=np.random.default_rng(0)),
    )

    assert result.details["mode"] == "cle"
    assert result.details["n_low_count_rounded"] == 1
    assert state.x[target] <= 10.0
    assert state.x[target] == np.rint(state.x[target])


def test_blended_hybrid_can_keep_low_count_cle_values_continuous():
    space = generate_fixed_species_space(
        ["A"],
        max_len=1,
        initial_counts={"A": 0.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_inflow=40.0,
        inflow_species_ids=[space.idx("A")],
    )
    target = network.species_idx("A")
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(
        BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.1, round_low_counts_after_cle=False)
    )

    result = stepper.step(
        state,
        1.0,
        StepperContext(network=network, rng=np.random.default_rng(0)),
    )

    assert result.details["mode"] == "cle"
    assert result.details["n_low_count_rounded"] == 0
    assert state.x[target] <= 10.0
    assert not np.isclose(state.x[target], np.rint(state.x[target]))


def test_blended_hybrid_beta_considers_products_by_default():
    space = generate_fixed_species_space(
        ["A", "B"],
        max_len=2,
        initial_counts={"A": 100.0, "B": 100.0, "AB": 0.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=1.0,
        k_poly_right=1.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
    )
    source = network.species_idx("A")
    monomer = network.species_idx("B")
    product = network.species_idx("AB")
    channel = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[source, monomer]))
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.01))

    beta = stepper._channel_betas(network, state.x)

    assert network.get_channel_products(channel) == (product,)
    assert state.x[source] > 30.0
    assert state.x[monomer] > 30.0
    assert state.x[product] == 0.0
    assert beta[channel] == 1.0


def test_blended_hybrid_beta_can_use_reactants_only():
    space = generate_fixed_species_space(
        ["A", "B"],
        max_len=2,
        initial_counts={"A": 100.0, "B": 100.0, "AB": 0.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(space, tables)
    source = network.species_idx("A")
    monomer = network.species_idx("B")
    channel = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[source, monomer]))
    state = SystemState.from_x0(network.x0)
    stepper = BlendedHybridStepper(
        BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.01, beta_species_mode="reactants")
    )

    beta = stepper._channel_betas(network, state.x)

    assert beta[channel] == 0.0


def test_blended_hybrid_runner_compatibility():
    network = make_network(initial_count=20.0)
    result = ExperimentRunner().run_one(
        network,
        BlendedHybridStepper(BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.01)),
        t_end=0.05,
        seed=4,
        max_steps=100,
    )
    assert result.summary.metadata["stop_reason"] == "reached_t_end"
    assert np.isclose(result.state.t, 0.05)
    assert np.all(np.isfinite(result.state.x))
    assert np.all(result.state.x >= 0.0)
