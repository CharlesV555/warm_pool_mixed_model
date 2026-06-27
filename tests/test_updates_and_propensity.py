import numpy as np
import pytest

from polymer_sim import ChannelBlock, ReactionNetworkData, SystemState, build_reaction_rule_tables, generate_fixed_species_space


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
