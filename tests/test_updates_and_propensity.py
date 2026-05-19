import numpy as np

from polymer_sim import ChannelBlock, ReactionNetworkData, SystemState, build_reaction_rule_tables, generate_fixed_species_space


def make_network(k_poly=1.0, k_frag=1.0):
    space = generate_fixed_species_space(["A", "B"], max_len=3)
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=k_poly,
        k_poly_right=k_poly,
        k_frag_left=k_frag,
        k_frag_right=k_frag,
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
