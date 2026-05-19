from polymer_sim import ChannelBlock, ReactionNetworkData, build_reaction_rule_tables, generate_fixed_species_space


def make_network():
    space = generate_fixed_species_space(["A", "B"], max_len=3)
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(space, tables)
    return space, tables, network


def test_join_tables():
    space, tables, _ = make_network()
    a = space.idx("A")
    b = space.idx("B")
    ab = space.idx("AB")
    ba = space.idx("BA")
    assert tables.left_join[a, b] == ab
    assert tables.right_join[b, a] == ba
    assert tables.left_join[a, space.idx("AB")] == space.idx("AAB")
    assert tables.right_join[space.idx("AB"), b] == space.idx("ABB")
    assert tables.left_join[a, space.idx("AAA")] == -1


def test_split_tables():
    space, tables, _ = make_network()
    ab = space.idx("AB")
    aba = space.idx("ABA")
    assert tables.split_left_monomer[ab] == space.idx("A")
    assert tables.split_left_rest[ab] == space.idx("B")
    assert tables.split_right_rest[ab] == space.idx("A")
    assert tables.split_right_monomer[ab] == space.idx("B")
    assert tables.split_left_rest[aba] == space.idx("BA")
    assert tables.split_right_rest[aba] == space.idx("AB")
    assert tables.split_left_monomer[space.idx("A")] == -1


def test_channel_mapping_is_block_contiguous():
    _, _, network = make_network()
    assert network.channel_offsets[ChannelBlock.LEFT_ADD] == 0
    assert network.channel_offsets[ChannelBlock.RIGHT_ADD] == network.channel_sizes[ChannelBlock.LEFT_ADD]
    start = network.channel_offsets[ChannelBlock.LEFT_SPLIT]
    assert network.get_channel_block(start) == ChannelBlock.LEFT_SPLIT
    assert network.get_channel_local_id(start) == 0


def test_length_two_split_is_merged_into_left_split():
    space, _, network = make_network()
    ab = space.idx("AB")
    assert network.left_split_local_id_by_source[ab] >= 0
    assert network.right_split_local_id_by_source[ab] == -1
    local = int(network.left_split_local_id_by_source[ab])
    assert network.left_split_multiplicity[local] == 2.0
