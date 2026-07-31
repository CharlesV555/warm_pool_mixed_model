import numpy as np
import pytest

from polymer_sim import (
    ChannelBlock,
    ReactionNetworkData,
    SSAStepper,
    StepperContext,
    SystemState,
    build_reaction_rule_tables,
    generate_fixed_species_space,
)


def make_network(k_poly=1.0, k_frag=1.0, **kwargs):
    space = generate_fixed_species_space(["A", "B"], max_len=3)
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=k_poly,
        k_poly_right=k_poly,
        k_frag_left=k_frag,
        k_frag_right=k_frag,
        **kwargs,
    )
    return network


def test_left_add_update():
    network = make_network()
    x = np.zeros(network.n_species)
    x[network.species_idx("A")] = 2
    x[network.species_idx("BA")] = 1
    state = SystemState.from_x0(x)
    local = int(network.left_add_local_id[network.species_idx("A"), network.species_idx("BA")])
    channel = network.channel_id(ChannelBlock.LEFT_ADD, local)
    network.apply_channel_update(state, channel)
    assert state.x[network.species_idx("A")] == 1
    assert state.x[network.species_idx("BA")] == 0
    assert state.x[network.species_idx("ABA")] == 1


def test_right_add_update():
    network = make_network()
    x = np.zeros(network.n_species)
    x[network.species_idx("BA")] = 1
    x[network.species_idx("A")] = 2
    state = SystemState.from_x0(x)
    local = int(network.right_add_local_id[network.species_idx("BA"), network.species_idx("A")])
    channel = network.channel_id(ChannelBlock.RIGHT_ADD, local)
    network.apply_channel_update(state, channel)
    assert state.x[network.species_idx("BA")] == 0
    assert state.x[network.species_idx("A")] == 1
    assert state.x[network.species_idx("BAA")] == 1


def test_left_split_update():
    network = make_network()
    x = np.zeros(network.n_species)
    x[network.species_idx("ABA")] = 1
    state = SystemState.from_x0(x)
    local = int(network.left_split_local_id_by_source[network.species_idx("ABA")])
    channel = network.channel_id(ChannelBlock.LEFT_SPLIT, local)
    network.apply_channel_update(state, channel)
    assert state.x[network.species_idx("ABA")] == 0
    assert state.x[network.species_idx("A")] == 1
    assert state.x[network.species_idx("BA")] == 1


def test_right_split_update():
    network = make_network()
    x = np.zeros(network.n_species)
    x[network.species_idx("ABA")] = 1
    state = SystemState.from_x0(x)
    local = int(network.right_split_local_id_by_source[network.species_idx("ABA")])
    channel = network.channel_id(ChannelBlock.RIGHT_SPLIT, local)
    network.apply_channel_update(state, channel)
    assert state.x[network.species_idx("ABA")] == 0
    assert state.x[network.species_idx("AB")] == 1
    assert state.x[network.species_idx("A")] == 1


def test_channel_reactant_terms_match_channel_reactants():
    network = make_network(
        k_poly=1.0,
        k_frag=1.0,
        k_outflow=1.0,
        outflow_species_ids=[0],
        k_inflow=1.0,
        inflow_species_ids=[1],
    )

    assert network.reaction_order.shape == (network.n_channels,)
    assert network.reactant1.shape == (network.n_channels,)
    assert network.reactant2.shape == (network.n_channels,)
    assert network.homo_second_order.shape == (network.n_channels,)

    for channel_id in range(network.n_channels):
        reactants = network.get_channel_reactants(channel_id)
        assert int(network.reaction_order[channel_id]) == len(reactants)
        if len(reactants) == 0:
            assert int(network.reactant1[channel_id]) == -1
            assert int(network.reactant2[channel_id]) == -1
        elif len(reactants) == 1:
            assert int(network.reactant1[channel_id]) == int(reactants[0])
            assert int(network.reactant2[channel_id]) == -1
        else:
            assert int(network.reactant1[channel_id]) == int(reactants[0])
            assert int(network.reactant2[channel_id]) == int(reactants[1])
            assert bool(network.homo_second_order[channel_id]) == (int(reactants[0]) == int(reactants[1]))


def test_inflow_update_and_fixed_propensity():
    network = make_network(k_inflow=3.5, inflow_species_ids=[0])
    target = network.species_idx("A")
    local = int(network.inflow_local_id_by_target[target])
    channel = network.channel_id(ChannelBlock.INFLOW, local)
    x = np.zeros(network.n_species)
    state = SystemState.from_x0(x)

    assert network.get_channel_reactants(channel) == ()
    assert network.get_channel_products(channel) == (target,)
    assert network.compute_base_propensity(channel, state) == 3.5

    state.x[target] = 100.0
    state.x[network.species_idx("B")] = 2.0
    assert network.compute_base_propensity(channel, state) == 3.5
    network.set_catalytic_strength(channel, catalyst_sid=network.species_idx("B"), strength=10.0)
    assert network.compute_propensity(channel, state) == 3.5
    network.apply_channel_update(state, channel)
    assert state.x[target] == 101.0


def test_inflow_capacity_reduces_propensity_to_zero_at_upper_limit():
    network = make_network(
        k_inflow=10.0,
        inflow_species_ids=[0],
        inflow_capacity=100.0,
        inflow_hill_coefficient=2.0,
    )
    target = network.species_idx("A")
    channel = network.channel_id(ChannelBlock.INFLOW, int(network.inflow_local_id_by_target[target]))

    state = SystemState.from_x0(np.zeros(network.n_species))
    assert network.compute_base_propensity(channel, state) == 10.0
    assert network.compute_propensity(channel, state) == 10.0

    state.x[target] = 50.0
    assert network.compute_base_propensity(channel, state) == pytest.approx(7.5)

    state.x[target] = 100.0
    assert network.compute_base_propensity(channel, state) == 0.0
    assert network.compute_propensity(channel, state) == 0.0

    propensities = network.compute_all_propensities(state)
    assert propensities[channel] == 0.0
    assert channel in network.species_to_channels[target]


def test_propensity_without_and_with_catalysis():
    network = make_network(k_poly=2.0)
    a = network.species_idx("A")
    b = network.species_idx("B")
    local = int(network.left_add_local_id[a, a])
    channel = network.channel_id(ChannelBlock.LEFT_ADD, local)
    x = np.zeros(network.n_species)
    x[a] = 5
    x[b] = 3
    state = SystemState.from_x0(x)
    assert network.compute_base_propensity(channel, state) == 20.0
    assert network.compute_propensity(channel, state) == 20.0
    network.set_catalytic_strength(channel, catalyst_sid=b, strength=0.5)
    assert network.get_catalytic_strength(channel, b) == 0.5
    assert network.get_catalytic_factor(channel, state) == 2.5
    assert network.compute_propensity(channel, state) == 50.0
    assert b in network.channel_to_catalysts[channel]
    assert channel in network.species_to_channels[b]


def test_block_vectorized_propensities_match_scalar_channels():
    network = make_network(
        k_poly=2.0,
        k_frag=0.5,
        k_outflow=0.25,
        outflow_species_ids=[8],
        k_inflow=0.75,
        inflow_species_ids=[0],
    )
    a = network.species_idx("A")
    b = network.species_idx("B")
    ab = network.species_idx("AB")
    aba = network.species_idx("ABA")

    x = np.arange(network.n_species, dtype=float) + 1.0
    state = SystemState.from_x0(x)

    left_add = network.channel_id(ChannelBlock.LEFT_ADD, int(network.left_add_local_id[a, b]))
    left_split = network.channel_id(ChannelBlock.LEFT_SPLIT, int(network.left_split_local_id_by_source[ab]))
    outflow = network.channel_id(ChannelBlock.OUTFLOW, int(network.outflow_local_id_by_source[aba]))
    inflow = network.channel_id(ChannelBlock.INFLOW, int(network.inflow_local_id_by_target[a]))
    network.set_catalytic_strength(left_add, catalyst_sid=ab, strength=0.25, rebuild=False, mirror_reverse=False)
    network.set_catalytic_strength(left_split, catalyst_sid=b, strength=0.5, rebuild=False, mirror_reverse=False)
    network.set_catalytic_strength(outflow, catalyst_sid=a, strength=0.75, rebuild=False, mirror_reverse=False)
    partial_vectorized = network.compute_all_propensities(state)
    partial_scalar = np.asarray([network.compute_propensity(channel_id, state) for channel_id in range(network.n_channels)])
    assert network.dependency_indices_dirty
    assert np.allclose(partial_vectorized, partial_scalar)

    network.set_catalytic_strength(inflow, catalyst_sid=b, strength=100.0, rebuild=True, mirror_reverse=False)

    vectorized = network.compute_all_propensities(state)
    scalar = np.asarray([network.compute_propensity(channel_id, state) for channel_id in range(network.n_channels)])

    assert np.allclose(vectorized, scalar)
    assert vectorized[inflow] == network.inflow_rates[0]


def test_block_vectorized_substrate_saturating_propensities_match_scalar_channels():
    network = make_network(k_poly=2.0, catalysis_mode="substrate_saturating", saturation_alpha=0.25)
    a = network.species_idx("A")
    b = network.species_idx("B")
    ab = network.species_idx("AB")
    ba = network.species_idx("BA")
    aa = network.species_idx("AA")

    x = np.zeros(network.n_species)
    x[a] = 1.5
    x[b] = 4.0
    x[ab] = 3.0
    x[ba] = 2.0
    state = SystemState.from_x0(x)

    same_species_channel = network.channel_id(ChannelBlock.LEFT_ADD, int(network.left_add_local_id[a, a]))
    mixed_channel = network.channel_id(ChannelBlock.LEFT_ADD, int(network.left_add_local_id[a, b]))
    network.set_catalytic_strength(same_species_channel, catalyst_sid=ab, strength=0.5, rebuild=False)
    network.set_catalytic_strength(mixed_channel, catalyst_sid=ba, strength=0.25, rebuild=True)

    vectorized = network.compute_all_propensities(state)
    scalar = np.asarray([network.compute_propensity(channel_id, state) for channel_id in range(network.n_channels)])

    assert np.allclose(vectorized, scalar)
    assert vectorized[same_species_channel] == 0.0
    assert vectorized[mixed_channel] > 0.0
    assert aa in network.get_channel_products(same_species_channel)


def test_sparse_catalysis_cache_matches_dense_fallback():
    network = make_network(k_poly=2.0, catalysis_mode="substrate_saturating", saturation_alpha=0.25)
    a = network.species_idx("A")
    b = network.species_idx("B")
    ab = network.species_idx("AB")
    ba = network.species_idx("BA")
    bb = network.species_idx("BB")
    left_channel = network.channel_id(ChannelBlock.LEFT_ADD, int(network.left_add_local_id[a, b]))
    right_channel = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[ba, a]))
    network.set_catalytic_strength(left_channel, catalyst_sid=ab, strength=0.5, rebuild=False, mirror_reverse=False)
    network.set_catalytic_strength(left_channel, catalyst_sid=bb, strength=0.25, rebuild=False, mirror_reverse=False)
    network.set_catalytic_strength(right_channel, catalyst_sid=ab, strength=0.75, rebuild=True, mirror_reverse=False)

    state = SystemState.from_x0(np.arange(network.n_species, dtype=float) + 3.0)
    sparse = network.compute_all_propensities(state)
    assert network._sparse_catalysis_ready
    assert network._block_local_ids_cache[ChannelBlock.LEFT_ADD] is network._local_ids_for_block(ChannelBlock.LEFT_ADD, None)
    assert bool(network._block_any_catalysts_cache[ChannelBlock.LEFT_ADD])

    network._sparse_catalysis_ready = False
    dense = network.compute_all_propensities(state)

    assert np.allclose(sparse, dense)


def test_local_propensity_update_matches_full_recompute():
    network = make_network(k_poly=2.0, k_frag=0.5)
    a = network.species_idx("A")
    b = network.species_idx("B")
    ab = network.species_idx("AB")
    ba = network.species_idx("BA")
    x = np.zeros(network.n_species)
    x[a] = 10.0
    x[b] = 8.0
    x[ab] = 2.0
    x[ba] = 3.0
    state = SystemState.from_x0(x)

    catalyzed = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[ba, a]))
    network.set_catalytic_strength(catalyzed, catalyst_sid=ab, strength=0.4)

    propensities = network.compute_all_propensities(state)
    fired = network.channel_id(ChannelBlock.RIGHT_ADD, int(network.right_add_local_id[a, b]))
    changed_species = network.get_channel_changed_species(fired)
    network.apply_channel_update(state, fired)

    affected = network.update_propensities_for_species(propensities, state, changed_species)
    full = network.compute_all_propensities(state)

    assert affected.size > 0
    assert np.allclose(propensities, full)


def test_ssa_local_propensity_cache_matches_full_recompute_after_event():
    network = make_network(k_poly=1.0, k_frag=0.1)
    state = SystemState.from_x0(network.x0)
    state.x[network.species_idx("A")] = 20.0
    state.x[network.species_idx("B")] = 20.0
    stepper = SSAStepper()
    result = stepper.step(
        state,
        100.0,
        StepperContext(network=network, rng=np.random.default_rng(123)),
    )

    assert result.event_occurred
    assert stepper._propensity_cache is not None
    assert np.allclose(stepper._propensity_cache, network.compute_all_propensities(state))


def test_catalysis_assignment_mirrors_reverse_reaction():
    network = make_network()
    catalyst = network.species_idx("B")
    source = network.species_idx("BA")
    monomer = network.species_idx("A")
    product = network.species_idx("BAA")
    local = int(network.right_add_local_id[source, monomer])
    channel = network.channel_id(ChannelBlock.RIGHT_ADD, local)
    reverse_channel = network.channel_id(
        ChannelBlock.RIGHT_SPLIT,
        int(network.right_split_local_id_by_source[product]),
    )

    assert reverse_channel in network.get_reverse_channel_ids(channel)
    network.set_catalytic_strength(channel, catalyst_sid=catalyst, strength=0.75)

    assert network.get_catalytic_strength(channel, catalyst) == 0.75
    assert network.get_catalytic_strength(reverse_channel, catalyst) == 0.75
    assert catalyst in network.get_channel_catalysts(reverse_channel)
    assert reverse_channel in network.species_to_channels[catalyst]


def test_substrate_saturating_propensity_uses_per_catalyst_capacity():
    network = make_network(k_poly=2.0, catalysis_mode="substrate_saturating", saturation_alpha=0.25)
    a = network.species_idx("A")
    b = network.species_idx("B")
    ba = network.species_idx("BA")
    bb = network.species_idx("BB")
    local = int(network.left_add_local_id[a, b])
    channel = network.channel_id(ChannelBlock.LEFT_ADD, local)

    x = np.zeros(network.n_species)
    x[a] = 10.0
    x[b] = 4.0
    x[ba] = 4.0
    x[bb] = 4.0
    state = SystemState.from_x0(x)

    network.set_catalytic_strength(channel, catalyst_sid=ba, strength=0.5, rebuild=False)
    network.set_catalytic_strength(channel, catalyst_sid=bb, strength=0.25, rebuild=True)

    substrate_capacity = 4.0
    effective_per_catalyst = substrate_capacity * 4.0 / (0.25 * substrate_capacity + 4.0)
    expected_factor = 1.0 + 0.5 * effective_per_catalyst + 0.25 * effective_per_catalyst
    assert network.compute_base_propensity(channel, state) == 80.0
    assert network.get_catalytic_factor(channel, state) == pytest.approx(expected_factor)
    assert network.compute_propensity(channel, state) == pytest.approx(80.0 * expected_factor)


def test_substrate_saturating_same_species_capacity_floor_can_zero_propensity():
    network = make_network(k_poly=2.0, catalysis_mode="substrate_saturating", saturation_alpha=0.25)
    a = network.species_idx("A")
    b = network.species_idx("B")
    local = int(network.left_add_local_id[a, a])
    channel = network.channel_id(ChannelBlock.LEFT_ADD, local)

    x = np.zeros(network.n_species)
    x[a] = 1.5
    x[b] = 3.0
    state = SystemState.from_x0(x)
    network.set_catalytic_strength(channel, catalyst_sid=b, strength=0.5)

    assert network.compute_base_propensity(channel, state) > 0.0
    assert network.compute_propensity(channel, state) == 0.0


def test_saturation_alpha_must_be_positive():
    with pytest.raises(ValueError, match="saturation_alpha"):
        make_network(catalysis_mode="substrate_saturating", saturation_alpha=0.0)
