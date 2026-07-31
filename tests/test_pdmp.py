from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from polymer_sim import (
    ChannelBlock,
    ElementaryExpansionConfig,
    ElementaryMassActionNetwork,
    ExperimentRunner,
    FastNetworkReportRecorder,
    FixedPDMPPartitionStrategy,
    FastSubnetworkSelector,
    FastSubnetwork,
    FiniteMarkovConfig,
    FiniteMarkovSubnetworkAnalyzer,
    LinearCatalysisScalingPDMPPartitionStrategy,
    PDMPConfig,
    PDMPStepper,
    ReactionNetworkData,
    ScalingPDMPConfig,
    ScalingPDMPPartitionStrategy,
    StepperContext,
    SummaryRecorder,
    SystemState,
    analyze_fast_network,
    build_elementary_mass_action_network,
    build_reaction_rule_tables,
    generate_fixed_species_space,
)
from polymer_sim.core.kernels import NUMBA_AVAILABLE


def make_polymer_network(**kwargs):
    space = generate_fixed_species_space(["A", "B"], max_len=2, initial_counts={"A": 10, "B": 10})
    tables = build_reaction_rule_tables(space)
    return ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=kwargs.pop("k_poly_left", 0.1),
        k_poly_right=kwargs.pop("k_poly_right", 0.1),
        k_frag_left=kwargs.pop("k_frag_left", 0.0),
        k_frag_right=kwargs.pop("k_frag_right", 0.0),
        **kwargs,
    )


def test_elementary_expansion_introduces_complex_for_catalyzed_ligation():
    network = make_polymer_network()
    a = network.species_idx("A")
    b = network.species_idx("B")
    aa = network.species_idx("AA")
    source = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[a, b]))
    network.set_catalytic_strength(source, catalyst_sid=aa, strength=2.0, mirror_reverse=False)

    elementary = build_elementary_mass_action_network(
        network,
        ElementaryExpansionConfig(
            catalyst_binding_rate=3.0,
            catalyst_unbinding_rate=4.0,
            catalytic_turnover_scale=5.0,
        ),
    )

    assert elementary.polymer_species_count == network.n_species
    assert any(name.startswith("complex:AA|A") for name in elementary.species_names)
    assert np.all(np.sum(elementary.nu_minus, axis=1) <= 2.0)

    roles = [label["role"] for label in elementary.reaction_labels]
    assert "complex_binding" in roles
    assert "complex_unbinding" in roles
    assert "catalytic_turnover" in roles

    turnover = [
        channel_id
        for channel_id, label in enumerate(elementary.reaction_labels)
        if label["role"] == "catalytic_turnover" and label["source_channel"] == source
    ]
    assert len(turnover) == 1
    reactant_names = [elementary.species_names[sid] for sid in elementary.get_channel_reactants(turnover[0])]
    product_names = [elementary.species_names[sid] for sid in elementary.get_channel_products(turnover[0])]
    assert "B" in reactant_names
    assert "AA" in product_names
    assert "AB" in product_names


def test_elementary_expansion_can_map_strength_to_binding_and_fixed_k2():
    network = make_polymer_network()
    a = network.species_idx("A")
    b = network.species_idx("B")
    aa = network.species_idx("AA")
    source = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[a, b]))
    gamma = 7.0
    network.set_catalytic_strength(source, catalyst_sid=aa, strength=gamma, mirror_reverse=False)

    elementary = build_elementary_mass_action_network(
        network,
        ElementaryExpansionConfig(
            catalyst_binding_rate_per_strength=2.0,
            catalyst_unbinding_rate=20.0,
            catalytic_turnover_rate=1.0,
        ),
    )

    binding = [
        channel_id
        for channel_id, label in enumerate(elementary.reaction_labels)
        if label["role"] == "complex_binding"
    ]
    unbinding = [
        channel_id
        for channel_id, label in enumerate(elementary.reaction_labels)
        if label["role"] == "complex_unbinding"
    ]
    turnover = [
        channel_id
        for channel_id, label in enumerate(elementary.reaction_labels)
        if label["role"] == "catalytic_turnover" and label["source_channel"] == source
    ]

    assert len(binding) == 1
    assert len(unbinding) == 1
    assert len(turnover) == 1
    assert elementary.rate_constants[binding[0]] == pytest.approx(2.0 * gamma)
    assert elementary.rate_constants[unbinding[0]] == pytest.approx(20.0)
    assert elementary.rate_constants[turnover[0]] == pytest.approx(1.0)


def test_elementary_inflow_is_standard_zero_order():
    network = make_polymer_network(
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_inflow=2.5,
        inflow_species_ids=[0],
        inflow_capacity=10.0,
        inflow_hill_coefficient=2.0,
    )
    elementary = build_elementary_mass_action_network(network)
    inflow_channels = [
        channel_id
        for channel_id, label in enumerate(elementary.reaction_labels)
        if label["block_type"] == "INFLOW"
    ]
    assert len(inflow_channels) == 1

    state = SystemState.from_x0(elementary.x0)
    state.x[network.species_idx("A")] = 100.0
    assert elementary.get_channel_reactants(inflow_channels[0]) == ()
    assert elementary.compute_propensity(inflow_channels[0], state) == 2.5


def test_elementary_inflow_rejects_nonstandard_mode():
    network = make_polymer_network(k_poly_left=0.0, k_poly_right=0.0, k_inflow=2.5, inflow_species_ids=[0])
    with pytest.raises(ValueError, match="standard_zero_order_inflow"):
        build_elementary_mass_action_network(
            network,
            ElementaryExpansionConfig(standard_zero_order_inflow=False),
        )


def test_elementary_propensity_precompute_vectorized_and_subset_order():
    network = ElementaryMassActionNetwork(
        species_names=["S0", "S1", "S2"],
        name_to_idx={"S0": 0, "S1": 1, "S2": 2},
        x0=np.array([5.0, 4.0, 0.0]),
        nu_minus=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        nu_plus=np.zeros((4, 3), dtype=float),
        rate_constants=np.array([2.0, 3.0, 0.5, 1.0], dtype=float),
    )
    state = SystemState.from_x0(network.x0)

    assert np.array_equal(network.reaction_order, np.array([0, 1, 2, 2], dtype=np.int8))
    assert np.array_equal(network.reactant1, np.array([-1, 0, 0, 0], dtype=np.int64))
    assert np.array_equal(network.reactant2, np.array([-1, -1, 1, 0], dtype=np.int64))
    assert np.array_equal(network.homo_second_order, np.array([False, False, False, True]))

    expected = np.array([2.0, 15.0, 10.0, 10.0])
    assert np.allclose(network.compute_all_propensities(state), expected)

    channels = np.array([3, 1, 0, 2], dtype=np.int64)
    out = np.empty(channels.shape, dtype=float)
    returned = network.compute_propensities_for_channels(channels, state, out=out)

    assert returned is out
    assert np.allclose(out, expected[channels])


def test_scaling_pdmp_partition_runs_and_partitions_all_channels():
    network = make_polymer_network()
    elementary = build_elementary_mass_action_network(network)
    state = SystemState.from_x0(elementary.x0)
    strategy = ScalingPDMPPartitionStrategy(
        ScalingPDMPConfig(N0=10.0, use_lp=False, enable_fast_subnetworks=True)
    )

    partition = strategy.partition(elementary, state)

    assigned = np.union1d(partition.continuous_channels, partition.discrete_channels)
    assert np.array_equal(assigned, np.arange(elementary.n_channels))
    assert partition.alpha.shape == (elementary.n_species,)
    assert partition.beta.shape == (elementary.n_channels,)
    assert partition.zeta.shape == (elementary.n_channels,)


def test_scaling_pdmp_algorithm3_mu_eta_delta_controls_partition_bounds():
    network = make_polymer_network(k_poly_left=0.0, k_poly_right=0.0, k_inflow=1.0, inflow_species_ids=[0])
    elementary = build_elementary_mass_action_network(network)
    state = SystemState.from_x0(elementary.x0)
    state.x[elementary.species_idx("A")] = 100.0
    strategy = ScalingPDMPPartitionStrategy(
        ScalingPDMPConfig(
            N0=10.0,
            continuous_copy_number_scale_threshold_mu=1.0,
            adaptation_scale_threshold_eta=0.25,
            reaction_relaxation_delta=0.75,
            use_lp=False,
        )
    )

    partition = strategy.partition(elementary, state)
    a_sid = elementary.species_idx("A")

    assert a_sid in partition.continuous_species
    assert partition.metadata["continuous_copy_number_scale_threshold_mu"] == 1.0
    assert partition.metadata["adaptation_scale_threshold_eta"] == 0.25
    assert partition.metadata["reaction_relaxation_delta"] == 0.75
    assert partition.lower_bounds[a_sid] == np.power(10.0, partition.alpha[a_sid] - 0.25)
    assert partition.upper_bounds[a_sid] == np.power(10.0, partition.alpha[a_sid] + 0.25)


def test_linear_catalysis_scaling_zeta_includes_catalyst_copy_and_strength_scale():
    network = make_polymer_network(k_poly_left=0.0, k_poly_right=0.1, catalysis_mode="linear")
    a = network.species_idx("A")
    b = network.species_idx("B")
    aa = network.species_idx("AA")
    channel = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[a, b]))
    network.set_catalytic_strength(channel, catalyst_sid=aa, strength=10.0, mirror_reverse=False)

    state = SystemState.from_x0(network.x0)
    state.x[a] = 10.0
    state.x[b] = 10.0
    state.x[aa] = 100.0
    strategy = LinearCatalysisScalingPDMPPartitionStrategy(
        ScalingPDMPConfig(N0=10.0, use_lp=False)
    )

    partition = strategy.partition(network, state)

    assert partition.metadata["method"] == "linear_catalysis_scaling_lp"
    assert partition.metadata["linear_catalysis_term_count"] >= 1
    assert partition.alpha[aa] == pytest.approx(2.0)
    assert partition.beta[channel] == pytest.approx(-1.0)
    assert partition.zeta[channel] == pytest.approx(4.0)


def test_linear_catalysis_scaling_zeta_includes_saturation_alpha_for_saturating_mode():
    network = make_polymer_network(
        k_poly_left=0.0,
        k_poly_right=0.1,
        catalysis_mode="substrate_saturating",
        saturation_alpha=0.1,
    )
    a = network.species_idx("A")
    b = network.species_idx("B")
    aa = network.species_idx("AA")
    channel = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[a, b]))
    network.set_catalytic_strength(channel, catalyst_sid=aa, strength=10.0, mirror_reverse=False)

    state = SystemState.from_x0(network.x0)
    state.x[a] = 1000.0
    state.x[b] = 100.0
    state.x[aa] = 10000.0
    strategy = LinearCatalysisScalingPDMPPartitionStrategy(
        ScalingPDMPConfig(N0=10.0, use_lp=False)
    )

    partition = strategy.partition(network, state)

    base_exponent = -1.0 + 3.0 + 2.0
    saturated_factor_exponent = min(
        1.0 + 3.0,
        1.0 + 2.0,
        1.0 + 4.0 - (-1.0),
    )
    assert partition.metadata["catalysis_mode"] == "substrate_saturating"
    assert partition.metadata["saturation_alpha_exponent"] == pytest.approx(-1.0)
    assert partition.metadata["saturating_catalysis_term_count"] >= 1
    assert partition.zeta[channel] == pytest.approx(base_exponent + saturated_factor_exponent)


def test_fast_subnetwork_selector_finds_timescale_separated_component():
    network = ElementaryMassActionNetwork(
        species_names=["S0", "S1", "S2"],
        name_to_idx={"S0": 0, "S1": 1, "S2": 2},
        x0=np.array([10.0, 10.0, 10.0]),
        nu_minus=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        nu_plus=np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        rate_constants=np.array([100.0, 100.0, 1.0]),
    )
    selected = FastSubnetworkSelector(threshold=1.0, max_size=2).select(
        network,
        zeta=np.array([3.0, 3.0, 0.0]),
    )

    assert selected
    assert set(selected[0].channels) == {0, 1}
    assert selected[0].delta_zeta >= 1.0


def make_reversible_binding_elementary_network():
    return ElementaryMassActionNetwork(
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
        rate_constants=np.array([3.0, 2.0]),
    )


def make_zero_order_inflow_elementary_network():
    return ElementaryMassActionNetwork(
        species_names=["S0"],
        name_to_idx={"S0": 0},
        x0=np.array([0.0]),
        nu_minus=np.array([[0.0]], dtype=float),
        nu_plus=np.array([[1.0]], dtype=float),
        rate_constants=np.array([1.0], dtype=float),
    )


def test_finite_markov_analyzer_accepts_reversible_binding_subnetwork():
    network = make_reversible_binding_elementary_network()
    state = SystemState.from_x0(network.x0)
    subnetwork = FastSubnetwork(
        channels=np.array([0, 1], dtype=np.int64),
        changed_species=np.array([0, 1, 2], dtype=np.int64),
        catalyst_species=np.empty(0, dtype=np.int64),
        surrounding_channels=np.empty(0, dtype=np.int64),
        delta_zeta=2.0,
    )

    result = FiniteMarkovSubnetworkAnalyzer(FiniteMarkovConfig(max_states=16)).analyze(
        network,
        state,
        subnetwork,
    )

    assert result.finite
    assert result.averageable
    assert result.state_count == 2
    assert result.transition_count == 2
    assert result.irreducible
    assert result.stationary_distribution is not None
    assert np.sum(result.stationary_distribution) == pytest.approx(1.0)


def test_fast_network_report_counts_averageable_finite_markov_subnetworks():
    network = make_reversible_binding_elementary_network()
    state = SystemState.from_x0(network.x0)
    partition = FixedPDMPPartitionStrategy(continuous_channels=[0, 1]).partition(network, state)

    report = analyze_fast_network(
        network,
        state,
        selector=FastSubnetworkSelector(threshold=0.0, max_size=2),
        zeta=partition.zeta,
        finite_config=FiniteMarkovConfig(max_states=16),
    )

    assert report.total_reaction_count == 2
    assert report.found_subnetwork_count >= 1
    assert report.averageable_subnetwork_count >= 1
    assert report.averageable_reaction_count == 2


def test_fast_network_report_recorder_writes_text_at_event_interval():
    network = make_reversible_binding_elementary_network()
    path = Path("tests") / "_fast_network_report_test.txt"
    if path.exists():
        path.unlink()
    recorder = FastNetworkReportRecorder(
        SummaryRecorder(),
        network=network,
        partition_strategy=FixedPDMPPartitionStrategy(continuous_channels=[0, 1]),
        output_path=path,
        interval_events=1000,
        selector=FastSubnetworkSelector(threshold=0.0, max_size=2),
        finite_config=FiniteMarkovConfig(max_states=16),
    )

    try:
        recorder.initialize(network.species_names, network.x0, {"n_channels": network.n_channels})
        recorder.record_step(
            time=0.1,
            state=network.x0,
            step_count=1000,
            event_count=1000,
            event_time=0.1,
            metadata={"channel_id": 0},
        )
        summary = recorder.finalize()

        text = path.read_text(encoding="utf-8")
        assert "event_count" in text
        assert "\n1000\t" in text
        assert summary.metadata["fast_network_report_count"] == 1
    finally:
        if path.exists():
            path.unlink()


def test_pdmp_stepper_runs_through_experiment_runner():
    network = make_polymer_network(
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_inflow=1.0,
        inflow_species_ids=[0],
    )
    elementary = build_elementary_mass_action_network(network)
    inflow_channels = [
        channel_id
        for channel_id, label in enumerate(elementary.reaction_labels)
        if label["block_type"] == "INFLOW"
    ]
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(continuous_channels=inflow_channels),
        config=PDMPConfig(ode_step=0.01),
    )

    result = ExperimentRunner().run_one(elementary, stepper, t_end=0.05, seed=123, dt=0.05)

    assert result.state.t == 0.05
    assert result.summary.n_steps >= 1
    assert result.summary.metadata["stepper_name"] == "PDMPStepper"
    assert result.state.x[elementary.species_idx("A")] > elementary.x0[elementary.species_idx("A")]


def test_pdmp_infinite_dt_returns_after_one_ode_step():
    network = make_zero_order_inflow_elementary_network()
    state = SystemState.from_x0(network.x0)
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(continuous_channels=[0]),
        config=PDMPConfig(ode_step=0.01, discrete_event_method="gillespie"),
    )

    result = stepper.step(
        state,
        float("inf"),
        StepperContext(network=network, rng=np.random.default_rng(123)),
    )

    assert result.advanced_time == pytest.approx(0.01)
    assert state.t == pytest.approx(0.01)
    assert state.x[0] == pytest.approx(0.01)


def test_pdmp_runner_wall_clock_only_run_stops_by_runtime():
    network = make_zero_order_inflow_elementary_network()
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(continuous_channels=[0]),
        config=PDMPConfig(ode_step=0.01, discrete_event_method="gillespie"),
    )

    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=float("inf"),
        seed=123,
        dt=None,
        max_steps=1_000_000,
        max_runtime_seconds=0.001,
    )

    assert result.summary.metadata["stop_reason"] == "max_runtime_seconds"
    assert np.isfinite(result.summary.final_time)


def test_pdmp_stepper_observes_expired_wall_deadline_inside_step():
    network = make_zero_order_inflow_elementary_network()
    state = SystemState.from_x0(network.x0)
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(continuous_channels=[0]),
        config=PDMPConfig(ode_step=0.01, discrete_event_method="gillespie"),
    )

    result = stepper.step(
        state,
        10.0,
        StepperContext(
            network=network,
            rng=np.random.default_rng(123),
            wall_deadline=perf_counter() - 1.0,
        ),
    )

    assert result.details["wall_deadline_reached"] is True
    assert result.advanced_time == 0.0


def test_pdmp_stepper_can_use_heap_discrete_event_locator():
    network = make_polymer_network(
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_inflow=1000.0,
        inflow_species_ids=[0],
    )
    state = SystemState.from_x0(network.x0)
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(continuous_channels=[]),
        config=PDMPConfig(
            ode_step=0.1,
            use_discrete_event_heap=True,
            use_local_propensity_updates=True,
        ),
    )

    result = stepper.step(
        state,
        0.1,
        StepperContext(network=network, rng=np.random.default_rng(123)),
    )

    assert result.event_occurred
    assert result.details["discrete_event_locator"] == "heap"
    assert result.details["heap_entries"] >= 0
    assert state.x[network.species_idx("A")] > network.x0[network.species_idx("A")]


def test_pdmp_stepper_can_use_gillespie_discrete_event_locator():
    network = make_polymer_network(
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_inflow=1000.0,
        inflow_species_ids=[0],
    )
    state = SystemState.from_x0(network.x0)
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(continuous_channels=[]),
        config=PDMPConfig(
            ode_step=0.1,
            discrete_event_method="gillespie",
            use_discrete_event_heap=True,
            use_local_propensity_updates=True,
        ),
    )

    result = stepper.step(
        state,
        0.1,
        StepperContext(network=network, rng=np.random.default_rng(123)),
    )

    assert result.event_occurred
    assert stepper.config.discrete_event_method == "gillespie"
    assert stepper.config.use_discrete_event_heap is False
    assert result.details["discrete_event_method"] == "gillespie"
    assert result.details["discrete_event_locator"] == "gillespie"
    assert result.details["discrete_hazard_mode"] == "scalar_integrated"
    assert state.x[network.species_idx("A")] > network.x0[network.species_idx("A")]


def test_fixed_pdmp_partition_marks_rq_discrete_channels_that_change_continuous_species():
    network = ElementaryMassActionNetwork(
        species_names=["S0", "S1"],
        name_to_idx={"S0": 0, "S1": 1},
        x0=np.array([10.0, 0.0]),
        nu_minus=np.array([[1.0, 0.0]], dtype=float),
        nu_plus=np.array([[0.0, 1.0]], dtype=float),
        rate_constants=np.array([1.0], dtype=float),
    )

    partition = FixedPDMPPartitionStrategy(
        continuous_channels=[],
        continuous_species=[1],
    ).partition(network, SystemState.from_x0(network.x0))

    assert partition.rq_channels.tolist() == [0]


def test_pdmp_gillespie_rq_event_forces_adaptation():
    network = ElementaryMassActionNetwork(
        species_names=["S0", "S1"],
        name_to_idx={"S0": 0, "S1": 1},
        x0=np.array([10.0, 0.0]),
        nu_minus=np.array([[1.0, 0.0]], dtype=float),
        nu_plus=np.array([[0.0, 1.0]], dtype=float),
        rate_constants=np.array([1000.0], dtype=float),
    )
    state = SystemState.from_x0(network.x0)
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(
            continuous_channels=[],
            continuous_species=[1],
        ),
        config=PDMPConfig(
            ode_step=0.1,
            discrete_event_method="gillespie",
            repartition_on_event=False,
        ),
    )

    result = stepper.step(
        state,
        0.1,
        StepperContext(network=network, rng=np.random.default_rng(123)),
    )

    assert result.event_occurred
    assert result.details["rq_event_count"] == 1
    assert result.details["n_repartitions"] == 2
    assert state.event_count == 1
    assert state.x[0] == 9.0
    assert state.x[1] == 1.0


def test_pdmp_gillespie_discrete_locator_checks_event_state_reactant_availability():
    network = make_polymer_network(
        k_poly_left=1000.0,
        k_poly_right=1000.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_outflow=600.0,
        outflow_species_ids=[0],
    )
    a = network.species_idx("A")
    outflow_channel = network.channel_id(ChannelBlock.OUTFLOW, int(network.outflow_local_id_by_source[a]))
    x0 = np.zeros(network.n_species, dtype=float)
    x0[a] = 2.0
    state = SystemState.from_x0(x0)
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(continuous_channels=[outflow_channel]),
        config=PDMPConfig(
            ode_step=0.0005,
            discrete_event_method="gillespie",
            validate_nonnegative=True,
        ),
    )

    result = stepper.step(
        state,
        0.0005,
        StepperContext(network=network, rng=np.random.default_rng(123)),
    )

    assert result.details["discrete_event_method"] == "gillespie"
    assert np.all(state.x >= 0.0)
    assert state.x[a] < 2.0


def test_pdmp_available_channel_mask_uses_precomputed_reactant_terms():
    network = make_polymer_network(k_poly_left=1.0, k_poly_right=1.0, k_frag_left=1.0, k_frag_right=1.0)
    a = network.species_idx("A")
    b = network.species_idx("B")
    aa = network.species_idx("AA")
    same = network.channel_id(ChannelBlock.LEFT_ADD, int(network.left_add_local_id[a, a]))
    hetero = network.channel_id(ChannelBlock.LEFT_ADD, int(network.left_add_local_id[a, b]))
    split = network.channel_id(ChannelBlock.LEFT_SPLIT, int(network.left_split_local_id_by_source[aa]))

    x = np.zeros(network.n_species, dtype=float)
    x[a] = 1.5
    x[b] = 1.0
    x[aa] = 0.5
    state = SystemState.from_x0(x)
    stepper = PDMPStepper(config=PDMPConfig(discrete_event_method="gillespie"))

    mask = stepper._available_channel_mask(
        network,
        state,
        np.asarray([same, hetero, split], dtype=np.int64),
    )

    assert mask.tolist() == [False, True, False]


def test_pdmp_gillespie_discrete_propensity_cache_updates_locally():
    network = make_polymer_network(k_poly_left=1.0, k_poly_right=0.0, k_frag_left=0.0, k_frag_right=0.0)
    a = network.species_idx("A")
    fired_channel = network.channel_id(ChannelBlock.LEFT_ADD, int(network.left_add_local_id[a, a]))
    x = np.zeros(network.n_species, dtype=float)
    x[a] = 2.0
    state = SystemState.from_x0(x)
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(),
        config=PDMPConfig(discrete_event_method="gillespie", adaptive=False),
    )
    rng = np.random.default_rng(123)
    context = StepperContext(network=network, rng=rng)
    store = stepper._ensure_runtime_store(state, network)
    propensities = stepper._get_propensities(network, state, store, reason="test")
    partition = stepper._compute_partition(network, state, propensities, context)
    stepper._install_partition(
        store,
        partition,
        rng,
        current_time=float(state.t),
        propensities=propensities,
        reset_all=True,
    )

    cached_total = stepper._available_discrete_propensity_total(
        network,
        state,
        partition.discrete_channels,
        propensities,
        store=store,
    )
    direct_total = stepper._available_discrete_propensity_total(
        network,
        state,
        partition.discrete_channels,
        propensities,
    )
    assert store["gillespie_cache_valid"]
    assert cached_total == pytest.approx(direct_total)

    changed_species = network.get_channel_changed_species(fired_channel)
    network.apply_channel_update(state, fired_channel)
    stepper._refresh_propensities_after_state_change(
        network,
        state,
        store,
        changed_species,
        rng,
        reason="test local jump",
        fired_channel=fired_channel,
    )

    propensities = store["propensities"]
    cached_total = stepper._available_discrete_propensity_total(
        network,
        state,
        partition.discrete_channels,
        propensities,
        store=store,
    )
    direct_total = stepper._available_discrete_propensity_total(
        network,
        state,
        partition.discrete_channels,
        propensities,
    )
    assert store["gillespie_cache_valid"]
    assert cached_total == pytest.approx(direct_total)


def test_elementary_dependency_csr_matches_species_to_channels():
    network = ElementaryMassActionNetwork(
        species_names=["A", "B", "AB"],
        name_to_idx={"A": 0, "B": 1, "AB": 2},
        x0=np.asarray([2.0, 1.0, 0.0], dtype=float),
        nu_minus=np.asarray(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        nu_plus=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        rate_constants=np.asarray([2.0, 0.5, 0.1], dtype=float),
    )

    for sid, expected in enumerate(network.species_to_channels):
        start = int(network.species_to_channels_indptr[sid])
        end = int(network.species_to_channels_indptr[sid + 1])
        assert network.species_to_channels_indices[start:end].tolist() == expected.tolist()


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is not installed")
def test_pdmp_gillespie_numba_backend_refreshes_elementary_cache_locally():
    network = ElementaryMassActionNetwork(
        species_names=["A", "B", "AB"],
        name_to_idx={"A": 0, "B": 1, "AB": 2},
        x0=np.asarray([1.0, 1.0, 0.0], dtype=float),
        nu_minus=np.asarray(
            [
                [1.0, 1.0, 0.0],  # A + B -> AB
                [1.0, 0.0, 0.0],  # A -> B
                [0.0, 0.0, 0.0],  # inflow-like zero-order test channel
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        nu_plus=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        rate_constants=np.asarray([2.0, 0.5, 0.1, 0.2], dtype=float),
    )
    state = SystemState.from_x0(network.x0)
    stepper = PDMPStepper(
        partition_strategy=FixedPDMPPartitionStrategy(),
        config=PDMPConfig(
            discrete_event_method="gillespie",
            adaptive=False,
            local_propensity_full_recompute_fraction=1.0,
            kernel_backend="numba",
        ),
    )
    rng = np.random.default_rng(123)
    context = StepperContext(network=network, rng=rng)
    store = stepper._ensure_runtime_store(state, network)
    propensities = stepper._get_propensities(network, state, store, reason="test")
    partition = stepper._compute_partition(network, state, propensities, context)
    stepper._install_partition(
        store,
        partition,
        rng,
        current_time=float(state.t),
        propensities=propensities,
        reset_all=True,
    )

    cached_total = stepper._available_discrete_propensity_total(
        network,
        state,
        partition.discrete_channels,
        propensities,
        store=store,
    )
    direct_total = stepper._available_discrete_propensity_total(
        network,
        state,
        partition.discrete_channels,
        propensities,
    )
    assert cached_total == pytest.approx(direct_total)
    assert store["numba_cache_rebuilds"] >= 1

    changed_species = network.get_channel_changed_species(0)
    network.apply_channel_update(state, 0)
    stepper._refresh_propensities_after_state_change(
        network,
        state,
        store,
        changed_species,
        rng,
        reason="test numba local jump",
        fired_channel=0,
    )

    propensities = store["propensities"]
    cached_total = stepper._available_discrete_propensity_total(
        network,
        state,
        partition.discrete_channels,
        propensities,
        store=store,
    )
    direct_total = stepper._available_discrete_propensity_total(
        network,
        state,
        partition.discrete_channels,
        propensities,
    )
    assert store["propensity_update_mode"] == "local_numba"
    assert store["numba_local_refreshes"] >= 1
    assert cached_total == pytest.approx(direct_total)


def test_pdmp_stepper_can_select_linear_catalysis_partition_method_on_polymer_network():
    network = make_polymer_network(
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_inflow=1.0,
        inflow_species_ids=[0],
        catalysis_mode="substrate_saturating",
        saturation_alpha=0.1,
    )
    stepper = PDMPStepper(
        partition_method="linear_catalysis_scaling",
        partition_config=ScalingPDMPConfig(N0=10.0, use_lp=False),
        config=PDMPConfig(ode_step=0.01),
    )

    result = ExperimentRunner().run_one(network, stepper, t_end=0.05, seed=123, dt=0.05)

    assert result.state.t == 0.05
    assert result.summary.n_steps >= 1
    assert result.state.x[network.species_idx("A")] > network.x0[network.species_idx("A")]
