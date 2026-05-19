import numpy as np

from polymer_sim import (
    ReactionNetworkData,
    assign_random_longest_catalyst_to_all_channels,
    assign_random_longest_catalysts_to_distinct_channels,
    build_reaction_rule_tables,
    clear_all_catalysis,
    generate_fixed_species_space,
    longest_polymer_species_ids,
)


def make_network():
    space = generate_fixed_species_space(["A", "B"], max_len=3)
    tables = build_reaction_rule_tables(space)
    return ReactionNetworkData.from_species_space(space, tables)


def test_longest_polymer_species_ids():
    network = make_network()
    longest = longest_polymer_species_ids(network)
    names = [network.species_names[int(sid)] for sid in longest]
    assert names == ["AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"]


def test_method1_assigns_one_longest_catalyst_to_all_channels():
    network = make_network()
    rng = np.random.default_rng(123)
    result = assign_random_longest_catalyst_to_all_channels(
        network,
        rng=rng,
        log_mean=0.0,
        log_sigma=0.25,
    )

    catalyst_sid = int(result["catalyst_sid"])
    longest = set(int(sid) for sid in longest_polymer_species_ids(network))
    assert catalyst_sid in longest
    assert result["channel_ids"].shape == (network.n_channels,)
    assert result["strengths"].shape == (network.n_channels,)
    assert np.all(result["strengths"] > 0.0)

    for channel_id in range(network.n_channels):
        cats = network.get_channel_catalysts(channel_id)
        assert cats.tolist() == [catalyst_sid]
        assert network.get_catalytic_strength(channel_id, catalyst_sid) > 0.0


def test_method2_assigns_distinct_catalysts_to_distinct_channels():
    network = make_network()
    rng = np.random.default_rng(456)
    result = assign_random_longest_catalysts_to_distinct_channels(
        network,
        4,
        rng=rng,
        log_mean=-1.0,
        log_sigma=0.5,
    )

    catalyst_sids = np.asarray(result["catalyst_sids"], dtype=np.int64)
    channel_ids = np.asarray(result["channel_ids"], dtype=np.int64)
    strengths = np.asarray(result["strengths"], dtype=float)

    assert catalyst_sids.shape == (4,)
    assert channel_ids.shape == (4,)
    assert strengths.shape == (4,)
    assert len(set(int(v) for v in catalyst_sids)) == 4
    assert len(set(int(v) for v in channel_ids)) == 4
    assert np.all(strengths > 0.0)

    for channel_id, catalyst_sid, strength in zip(channel_ids, catalyst_sids, strengths):
        cats = network.get_channel_catalysts(int(channel_id))
        assert cats.tolist() == [int(catalyst_sid)]
        assert np.isclose(network.get_catalytic_strength(int(channel_id), int(catalyst_sid)), float(strength))

    untouched = set(range(network.n_channels)) - set(int(cid) for cid in channel_ids)
    for channel_id in untouched:
        assert network.get_channel_catalysts(channel_id).size == 0


def test_clear_all_catalysis():
    network = make_network()
    rng = np.random.default_rng(789)
    assign_random_longest_catalyst_to_all_channels(network, rng=rng)
    clear_all_catalysis(network)
    for channel_id in range(network.n_channels):
        assert network.get_channel_catalysts(channel_id).size == 0
