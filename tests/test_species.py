import numpy as np

from polymer_sim import generate_fixed_species_space


def test_species_order_length_then_lexicographic():
    space = generate_fixed_species_space(["B", "A"], max_len=3)
    assert space.species_names == [
        "A",
        "B",
        "AA",
        "AB",
        "BA",
        "BB",
        "AAA",
        "AAB",
        "ABA",
        "ABB",
        "BAA",
        "BAB",
        "BBA",
        "BBB",
    ]
    assert np.array_equal(space.lengths[:2], np.array([1, 1]))
    assert space.n_monomers == 2
    assert space.name_to_idx["A"] == 0
    assert space.name_to_idx["B"] == 1


def test_initial_counts_mapping():
    space = generate_fixed_species_space(["A", "B"], max_len=2, initial_counts={"A": 3, "AB": 2})
    assert space.x0[space.idx("A")] == 3
    assert space.x0[space.idx("AB")] == 2
    assert space.x0[space.idx("B")] == 0
