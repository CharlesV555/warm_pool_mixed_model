from polymer_sim import (
    assign_paper_minimal_catalysis,
    build_n3_wh_network,
    build_n3_wh_reactions,
    build_n3_wh_species,
    compute_max_raf,
    enumerate_irr_rafs,
)


def _reaction_ids(reactions):
    return {reaction.reaction_id for reaction in reactions}


def _catalysis_map(network, reactions):
    return {
        reaction.channel_id: set(int(sid) for sid in network.get_channel_catalysts(reaction.channel_id))
        for reaction in reactions
    }


def test_build_n3_wh_species():
    space = build_n3_wh_species()
    assert space.n_species == 14
    assert space.idx("01") != space.idx("10")


def test_build_n3_wh_reactions():
    network = build_n3_wh_network()
    reaction_ids = _reaction_ids(build_n3_wh_reactions(network))
    assert "0+0->00" in reaction_ids
    assert "0+1->01" in reaction_ids
    assert "1+0->10" in reaction_ids
    assert "1+1->11" in reaction_ids
    assert "00+0->000" in reaction_ids
    assert "11+1->111" in reaction_ids


def test_assign_paper_minimal_catalysis():
    network = build_n3_wh_network()
    reactions = build_n3_wh_reactions(network)
    result = assign_paper_minimal_catalysis(network)

    sid_000 = network.species_idx("000")
    sid_111 = network.species_idx("111")
    assert int(result["catalyst_000"]) == sid_000
    assert int(result["catalyst_111"]) == sid_111

    for reaction in reactions:
        cats = set(int(sid) for sid in network.get_channel_catalysts(reaction.channel_id))
        if reaction.category == "R1":
            assert cats == {sid_000}
        elif reaction.category == "R4":
            assert cats == {sid_111}
        else:
            assert cats == set()


def test_compute_max_raf():
    network = build_n3_wh_network()
    reactions = build_n3_wh_reactions(network)
    assign_paper_minimal_catalysis(network)
    food = {network.species_idx("0"), network.species_idx("1")}
    max_raf = compute_max_raf(food, reactions, _catalysis_map(network, reactions))

    assert _reaction_ids(max_raf) == {
        "0+0->00",
        "00+0->000",
        "1+1->11",
        "11+1->111",
    }


def test_enumerate_irr_raf():
    network = build_n3_wh_network()
    reactions = build_n3_wh_reactions(network)
    assign_paper_minimal_catalysis(network)
    food = {network.species_idx("0"), network.species_idx("1")}
    catalysis_map = _catalysis_map(network, reactions)
    max_raf = compute_max_raf(food, reactions, catalysis_map)
    irr_rafs = enumerate_irr_rafs(food, max_raf, catalysis_map)

    irr_sets = {frozenset(_reaction_ids(subset)) for subset in irr_rafs}
    assert irr_sets == {
        frozenset({"0+0->00", "00+0->000"}),
        frozenset({"1+1->11", "11+1->111"}),
    }
