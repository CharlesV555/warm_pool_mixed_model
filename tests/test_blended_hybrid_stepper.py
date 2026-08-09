from pathlib import Path

import numpy as np

import polymer_sim.simulation.stepper as stepper_module
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
from polymer_sim.simulation.stepper import _lookup_beta_affected_channels, _species_beta


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
    assert network.nu_csr is network.nu_csr
    assert np.array_equal(network.nu_csr.toarray(), nu)
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
    assert network.nu_csr is network.nu_csr
    assert np.array_equal(network.nu_csr.toarray(), nu)
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


def test_blended_hybrid_beta_reverse_lookup_updates_small_affected_set():
    class LocalBetaStepper(BlendedHybridStepper):
        def _beta_local_update_limit(self, n_channels: int) -> int:
            return int(n_channels)

    network = make_network(initial_count=100.0)
    x = np.full(network.n_species, 100.0, dtype=float)
    stepper = LocalBetaStepper(
        BlendedHybridConfig(
            i1=10.0,
            i2=30.0,
            dt_cle=0.01,
            beta_compute_mode="beta_compute_by_state_difference",
        )
    )

    beta0 = stepper._channel_betas(network, x).copy()
    assert stepper._last_beta_full_recompute

    sid = network.species_idx("AAA")
    x_changed = x.copy()
    x_changed[sid] = 20.0
    lookup = stepper._channel_beta_lookup(network)
    affected = _lookup_beta_affected_channels(lookup, np.asarray([sid], dtype=np.int64))

    beta1 = stepper._channel_betas(network, x_changed)
    expected = BlendedHybridStepper(stepper.config)._channel_betas(network, x_changed)

    assert 0 < affected.size <= stepper._beta_local_update_limit(network.n_channels)
    assert not stepper._last_beta_full_recompute
    assert stepper._last_beta_affected_updates == affected.size
    assert np.allclose(beta1, expected)
    assert np.any(beta0[affected] != beta1[affected])


def test_blended_hybrid_beta_state_difference_local_update_recomputes_only_affected_species(monkeypatch):
    class LocalBetaStepper(BlendedHybridStepper):
        def _beta_local_update_limit(self, n_channels: int) -> int:
            return int(n_channels)

    network = make_network(initial_count=100.0)
    x = np.full(network.n_species, 100.0, dtype=float)
    sid = network.species_idx("AAA")
    x_changed = x.copy()
    x_changed[sid] = 20.0
    expected = BlendedHybridStepper(BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.01))._channel_betas(
        network,
        x_changed,
    )
    stepper = LocalBetaStepper(
        BlendedHybridConfig(
            i1=10.0,
            i2=30.0,
            dt_cle=0.01,
            beta_compute_mode="beta_compute_by_state_difference",
        )
    )

    calls: list[int] = []
    original = stepper_module._species_beta_array

    def record_species_beta_array(values: np.ndarray, i1: float, i2: float) -> np.ndarray:
        calls.append(np.asarray(values).size)
        return original(values, i1, i2)

    monkeypatch.setattr(stepper_module, "_species_beta_array", record_species_beta_array)

    stepper._channel_betas(network, x)
    calls.clear()
    beta = stepper._channel_betas(network, x_changed)
    lookup = stepper._channel_beta_lookup(network)
    affected = _lookup_beta_affected_channels(lookup, np.asarray([sid], dtype=np.int64))
    affected_species = lookup.relevant_species[affected][lookup.relevant_mask[affected]]

    assert not stepper._last_beta_full_recompute
    assert np.allclose(beta, expected)
    assert affected_species.size < network.n_species
    assert calls == [np.unique(affected_species).size]


def test_blended_hybrid_beta_reverse_lookup_updates_large_affected_set_when_limit_disabled():
    space = generate_fixed_species_space(["A", "B"], max_len=5, initial_counts={"A": 100.0, "B": 100.0})
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(space, tables)
    x = np.full(network.n_species, 100.0, dtype=float)
    stepper = BlendedHybridStepper(
        BlendedHybridConfig(
            i1=10.0,
            i2=30.0,
            dt_cle=0.01,
            beta_compute_mode="beta_compute_by_state_difference",
        )
    )

    stepper._channel_betas(network, x)
    x_changed = x.copy()
    x_changed[network.species_idx("A")] = 20.0
    lookup = stepper._channel_beta_lookup(network)
    affected = _lookup_beta_affected_channels(lookup, np.asarray([network.species_idx("A")], dtype=np.int64))

    beta = stepper._channel_betas(network, x_changed)
    expected = BlendedHybridStepper(stepper.config)._channel_betas(network, x_changed)

    assert affected.size > stepper._beta_local_update_limit(network.n_channels)
    if affected.size >= network.n_channels:
        assert stepper._last_beta_full_recompute
        assert stepper._last_beta_affected_updates == 0
    else:
        assert not stepper._last_beta_full_recompute
        assert stepper._last_beta_affected_updates == affected.size
    assert np.allclose(beta, expected)


def test_blended_hybrid_beta_fully_compute_skips_local_update_path():
    class LocalBetaStepper(BlendedHybridStepper):
        def _beta_local_update_limit(self, n_channels: int) -> int:
            return int(n_channels)

    network = make_network(initial_count=100.0)
    x = np.full(network.n_species, 100.0, dtype=float)
    stepper = LocalBetaStepper(BlendedHybridConfig(i1=10.0, i2=30.0, dt_cle=0.01))

    stepper._channel_betas(network, x)
    x_changed = x.copy()
    x_changed[network.species_idx("AAA")] = 20.0
    beta = stepper._channel_betas(network, x_changed)
    expected = BlendedHybridStepper(stepper.config)._channel_betas(network, x_changed)

    assert stepper.config.beta_compute_mode == "beta_fully_compute"
    assert stepper._last_beta_full_recompute
    assert stepper._last_beta_affected_updates == 0
    assert np.allclose(beta, expected)


def test_blended_hybrid_strict_int_for_cle_uses_rounded_propensities():
    space = generate_fixed_species_space(["A"], max_len=1, initial_counts={"A": 0.0})
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_outflow=10.0,
        outflow_species_ids=[space.idx("A")],
    )
    state = SystemState(t=0.0, x=np.asarray([0.4], dtype=float))
    stepper = BlendedHybridStepper(
        BlendedHybridConfig(
            i1=-2.0,
            i2=-1.0,
            dt_cle=0.01,
            adaptive_cle_dt=False,
            round_low_counts_after_cle=False,
            strict_int_for_CLE=True,
        )
    )

    result = stepper.step(
        state,
        0.01,
        StepperContext(network=network, rng=np.random.default_rng(31)),
    )

    assert result.details["mode"] == "cle"
    assert result.details["total_cle_propensity"] == 0.0
    assert np.isclose(state.x[0], 0.4)


def test_blended_hybrid_strict_int_for_cle_reuses_mixed_jump_propensities():
    class CountingBlendedHybridStepper(BlendedHybridStepper):
        def __init__(self, config: BlendedHybridConfig):
            super().__init__(config)
            self.propensity_calls = 0

        def _propensities_for_x(self, network, x, t):
            self.propensity_calls += 1
            return super()._propensities_for_x(network, x, t)

    space = generate_fixed_species_space(
        ["A", "B"],
        max_len=1,
        initial_counts={"A": 20.0, "B": 20.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_outflow=0.01,
        outflow_species_ids=[space.idx("A"), space.idx("B")],
    )
    state = SystemState.from_x0(network.x0)
    stepper = CountingBlendedHybridStepper(
        BlendedHybridConfig(
            i1=10.0,
            i2=30.0,
            dt_cle=1e-6,
            adaptive_cle_dt=False,
            strict_int_for_CLE=True,
            local_propensity_calculation=True,
        )
    )

    context = StepperContext(network=network, rng=np.random.default_rng(32))

    result = stepper.step(state, 1e-6, context)
    assert result.details["mode"] == "mixed_cle"
    assert stepper.propensity_calls == 1

    result = stepper.step(state, 1e-6, context)
    assert result.details["mode"] == "mixed_cle"
    assert stepper.propensity_calls == 1

    state.x[:] = stepper._observed_propensity_state
    state.x[space.idx("A")] += 1.0
    result = stepper.step(state, 1e-6, context)
    assert result.details["mode"] == "mixed_cle"
    assert np.isclose(result.details["total_jump_propensity"], 0.1945)
    assert stepper.propensity_calls == 1


def test_blended_hybrid_observed_propensity_reuses_beta_affected_sets(monkeypatch):
    class LocalObservedPropensityStepper(BlendedHybridStepper):
        def _observed_propensity_local_update_limit(self, n_channels: int) -> int:
            return int(n_channels)

    network = make_network(initial_count=100.0)
    x = np.full(network.n_species, 100.0, dtype=float)
    sid = network.species_idx("AAA")
    stepper = LocalObservedPropensityStepper(
        BlendedHybridConfig(
            i1=10.0,
            i2=30.0,
            dt_cle=0.01,
            strict_int_for_CLE=True,
            local_propensity_calculation=True,
            beta_compute_mode="beta_compute_by_state_difference",
        )
    )

    stepper._channel_betas(network, x)
    stepper._propensities_for_observed_cached(network, x, 0.0, "cached propensities")

    x_changed = x.copy()
    x_changed[sid] = 20.0
    stepper._channel_betas(network, x_changed)
    beta_affected_channels = stepper._last_beta_reuse_beta_channels.copy()
    beta_affected_species = stepper._last_beta_reuse_affected_species.copy()
    assert beta_affected_channels.size > 0
    assert beta_affected_species.size > 0

    def fail_affected_channels_for_species(self, species_ids):
        raise AssertionError("propensity should reuse beta affected channels")

    monkeypatch.setattr(ReactionNetworkData, "affected_channels_for_species", fail_affected_channels_for_species)

    propensities = stepper._propensities_for_observed_cached(network, x_changed, 0.0, "cached propensities")
    expected = network.compute_all_propensities(SystemState(t=0.0, x=x_changed))

    assert np.allclose(propensities, expected)
    assert stepper._last_observed_propensity_reused_beta_affected
    assert stepper._last_observed_propensity_beta_affected_species == beta_affected_species.size
    assert stepper._last_observed_propensity_beta_affected_channels == beta_affected_channels.size
    assert stepper._last_observed_propensity_affected_updates == stepper._last_beta_reuse_affected_channels.size
    assert stepper._last_observed_propensity_update_path == "local_update"


def test_blended_hybrid_local_propensity_calculation_without_strict_int():
    class CountingLocalPropensityStepper(BlendedHybridStepper):
        def __init__(self, config: BlendedHybridConfig):
            super().__init__(config)
            self.full_propensity_calls = 0

        def _propensities_for_x(self, network, x, t):
            self.full_propensity_calls += 1
            return super()._propensities_for_x(network, x, t)

        def _observed_propensity_local_update_limit(self, n_channels: int) -> int:
            return int(n_channels)

    network = make_network(initial_count=100.0)
    x = np.full(network.n_species, 100.0, dtype=float)
    sid = network.species_idx("AAA")
    stepper = CountingLocalPropensityStepper(
        BlendedHybridConfig(
            i1=10.0,
            i2=30.0,
            dt_cle=0.01,
            strict_int_for_CLE=False,
            local_propensity_calculation=True,
        )
    )

    prop0 = stepper._propensities_for_state(network, x, 0.0, "cached propensities").copy()
    assert stepper.full_propensity_calls == 1
    assert stepper._last_observed_propensity_update_path == "full_recompute"

    x_changed = x.copy()
    x_changed[sid] = 20.0
    prop1 = stepper._propensities_for_state(network, x_changed, 0.0, "cached propensities")
    expected = network.compute_all_propensities(SystemState(t=0.0, x=x_changed))

    assert np.allclose(prop0, network.compute_all_propensities(SystemState(t=0.0, x=x)))
    assert np.allclose(prop1, expected)
    assert stepper.full_propensity_calls == 1
    assert stepper._last_observed_propensity_update_path == "local_update"
    assert stepper._last_observed_propensity_affected_updates > 0


def test_blended_hybrid_observed_propensity_reuse_adds_changed_catalyst_channels(monkeypatch):
    class LocalObservedPropensityStepper(BlendedHybridStepper):
        def _observed_propensity_local_update_limit(self, n_channels: int) -> int:
            return int(n_channels)

    network = make_network(initial_count=100.0)
    source = network.species_idx("A")
    monomer = network.species_idx("B")
    product = network.species_idx("AB")
    catalyst = network.species_idx("AAA")
    channel = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[source, monomer]))
    network.set_catalytic_strength(channel, catalyst_sid=catalyst, strength=2.0, mirror_reverse=False)
    x = np.full(network.n_species, 100.0, dtype=float)
    stepper = LocalObservedPropensityStepper(
        BlendedHybridConfig(
            i1=10.0,
            i2=30.0,
            dt_cle=0.01,
            strict_int_for_CLE=True,
            local_propensity_calculation=True,
            beta_compute_mode="beta_compute_by_state_difference",
        )
    )

    stepper._channel_betas(network, x)
    stepper._propensities_for_observed_cached(network, x, 0.0, "cached propensities")

    x_changed = x.copy()
    x_changed[catalyst] = 20.0
    stepper._channel_betas(network, x_changed)
    lookup = stepper._channel_beta_lookup(network)
    beta_affected = _lookup_beta_affected_channels(lookup, np.asarray([catalyst], dtype=np.int64))

    assert channel not in set(int(cid) for cid in beta_affected)
    assert stepper._last_beta_reuse_extra_channels.size == 0
    assert stepper._last_beta_reuse_changed_catalyst_species.size == 0
    assert stepper._last_beta_reuse_catalyst_channels.size == 0

    def fail_affected_channels_for_species(self, species_ids):
        raise AssertionError("propensity should reuse beta affected channels plus catalyst channels")

    monkeypatch.setattr(ReactionNetworkData, "affected_channels_for_species", fail_affected_channels_for_species)

    propensities = stepper._propensities_for_observed_cached(network, x_changed, 0.0, "cached propensities")
    expected = network.compute_all_propensities(SystemState(t=0.0, x=x_changed))

    assert np.allclose(propensities, expected)
    assert propensities[channel] == expected[channel]
    assert product not in network.get_channel_reactants(channel)
    assert channel in set(int(cid) for cid in stepper._last_beta_reuse_catalyst_channels)
    assert channel in set(int(cid) for cid in stepper._last_beta_reuse_extra_channels)
    assert catalyst in set(int(sid) for sid in stepper._last_beta_reuse_changed_catalyst_species)
    assert stepper._last_observed_propensity_reused_beta_affected
    assert stepper._last_observed_propensity_changed_catalyst_species == 1
    assert stepper._last_observed_propensity_catalyst_affected_channels >= 1
    assert stepper._last_observed_propensity_beta_extra_channels >= 1


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


def test_blended_hybrid_cle_sparsity_sampler_writes_summary_not_trajectory():
    network = make_network(initial_count=20.0)
    plot_path = Path("tests") / "_cle_sparsity_sampler_plot_tmp.png"
    plot_path.unlink(missing_ok=True)
    result = ExperimentRunner().run_one(
        network,
        BlendedHybridStepper(
            BlendedHybridConfig(
                i1=-2.0,
                i2=-1.0,
                dt_cle=0.01,
                adaptive_cle_dt=False,
                cle_sparsity_sampling=True,
                cle_sparsity_sample_interval=1,
                cle_sparsity_plot_path=str(plot_path),
            )
        ),
        t_end=0.02,
        seed=41,
        max_steps=10,
    )

    metadata = result.summary.metadata["cle_sparsity_sampling"]
    assert metadata["enabled"] is True
    assert metadata["sample_interval"] == 1
    assert metadata["cle_increment_calls"] >= 1
    assert metadata["n_samples"] == metadata["cle_increment_calls"]
    assert metadata["plot_path"] == str(plot_path)
    assert plot_path.exists()
    plot_path.unlink(missing_ok=True)

    sample = metadata["samples"][0]
    assert sample["amounts_shape"] == [network.n_channels]
    assert sample["stoichiometry_shape"] == [network.n_channels, network.n_species]
    assert sample["csr_stoichiometry_shape"] == [network.n_channels, network.n_species]
    assert 0.0 <= sample["amounts_zero_fraction"] <= 1.0
    assert 0.0 <= sample["amounts_nonzero_fraction"] <= 1.0
    assert 0.0 <= sample["stoichiometry_zero_fraction"] <= 1.0
    assert 0.0 <= sample["stoichiometry_nonzero_fraction"] <= 1.0
