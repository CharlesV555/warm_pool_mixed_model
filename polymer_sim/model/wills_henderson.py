from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from polymer_sim.core.enums import ChannelBlock
from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.model.catalysis import clear_all_catalysis
from polymer_sim.model.rules import build_reaction_rule_tables
from polymer_sim.model.species import SpeciesSpace, generate_fixed_species_space


@dataclass(slots=True)
class WHReaction:
    channel_id: int
    reaction_id: str
    reactants: tuple[int, int]
    products: tuple[int]
    category: str
    rate_constant: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def build_n3_wh_species(
    initial_counts: dict[str, float] | None = None,
) -> SpeciesSpace:
    return generate_fixed_species_space(["0", "1"], max_len=3, initial_counts=initial_counts)


def build_n3_wh_network(
    *,
    initial_counts: dict[str, float] | None = None,
    k_right_add: float = 1.0,
    k_trimer_outflow: float = 0.0,
    k_nonfood_outflow: float | None = None,
    food_species: tuple[str, ...] = ("0", "1"),
    catalysis_mode: str = "linear",
    saturation_alpha: float = 0.25,
) -> ReactionNetworkData:
    """Build the n=3 Wills-Henderson network.

    ``k_trimer_outflow`` is retained as a compatibility alias. New code should
    use ``k_nonfood_outflow`` because outflow now applies to every non-food
    species, not only trimers. ``food_species`` controls which species are
    excluded from natural outflow.
    """

    space = build_n3_wh_species(initial_counts=initial_counts)
    tables = build_reaction_rule_tables(space)
    if k_nonfood_outflow is not None and k_trimer_outflow != 0.0:
        if not np.isclose(float(k_nonfood_outflow), float(k_trimer_outflow)):
            raise ValueError("k_trimer_outflow and k_nonfood_outflow specify different rates")

    outflow_rate = float(k_trimer_outflow if k_nonfood_outflow is None else k_nonfood_outflow)
    food_species_ids = np.asarray([space.idx(name) for name in food_species], dtype=np.int64)
    nonfood_species_ids = np.setdiff1d(
        np.arange(space.n_species, dtype=np.int64),
        food_species_ids,
        assume_unique=True,
    )
    return ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.0,
        k_poly_right=k_right_add,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_outflow=outflow_rate,
        outflow_species_ids=nonfood_species_ids if outflow_rate > 0.0 else None,
        catalysis_mode=catalysis_mode,
        saturation_alpha=saturation_alpha,
    )


def build_n3_wh_reactions(network: ReactionNetworkData) -> list[WHReaction]:
    reactions: list[WHReaction] = []
    for local_id in range(network.channel_sizes[ChannelBlock.RIGHT_ADD]):
        channel_id = network.channel_id(ChannelBlock.RIGHT_ADD, local_id)
        reactants = network.get_channel_reactants(channel_id)
        products = network.get_channel_products(channel_id)
        category = classify_wh_reaction_category(network, channel_id)
        reaction_id = (
            f"{network.species_names[reactants[0]]}"
            f"+{network.species_names[reactants[1]]}"
            f"->{network.species_names[products[0]]}"
        )
        reactions.append(
            WHReaction(
                channel_id=channel_id,
                reaction_id=reaction_id,
                reactants=(int(reactants[0]), int(reactants[1])),
                products=(int(products[0]),),
                category=category,
                rate_constant=float(network.right_add_rates[local_id]),
                metadata={"block_type": network.get_channel_block_name(channel_id)},
            )
        )
    return reactions


def classify_wh_reaction_category(network: ReactionNetworkData, channel_id: int) -> str:
    block = network.get_channel_block(channel_id)
    if block != ChannelBlock.RIGHT_ADD:
        raise ValueError("Wills-Henderson category labels only apply to RIGHT_ADD channels")

    source_sid, monomer_sid = network.get_channel_reactants(channel_id)
    source_name = network.species_names[int(source_sid)]
    monomer_name = network.species_names[int(monomer_sid)]
    last_char = source_name[-1]
    if last_char == "0" and monomer_name == "0":
        return "R1"
    if last_char == "0" and monomer_name == "1":
        return "R2"
    if last_char == "1" and monomer_name == "0":
        return "R3"
    if last_char == "1" and monomer_name == "1":
        return "R4"
    raise ValueError("unexpected source/monomer combination")


def build_n3_wh_reaction_index(network: ReactionNetworkData) -> dict[int, WHReaction]:
    return {reaction.channel_id: reaction for reaction in build_n3_wh_reactions(network)}


def assign_paper_minimal_catalysis(
    network: ReactionNetworkData,
    *,
    strength: float = 1.0,
    reset_existing: bool = True,
) -> dict[str, object]:
    if reset_existing:
        clear_all_catalysis(network, rebuild=False)

    catalyst_000 = network.species_idx("000")
    catalyst_111 = network.species_idx("111")
    channel_ids_r1: list[int] = []
    channel_ids_r4: list[int] = []

    for reaction in build_n3_wh_reactions(network):
        if reaction.category == "R1":
            network.set_catalytic_strength(
                reaction.channel_id,
                catalyst_sid=catalyst_000,
                strength=float(strength),
                rebuild=False,
            )
            channel_ids_r1.append(reaction.channel_id)
        elif reaction.category == "R4":
            network.set_catalytic_strength(
                reaction.channel_id,
                catalyst_sid=catalyst_111,
                strength=float(strength),
                rebuild=False,
            )
            channel_ids_r4.append(reaction.channel_id)

    network.rebuild_dependency_indices()
    return {
        "catalyst_000": catalyst_000,
        "catalyst_111": catalyst_111,
        "channel_ids_r1": np.asarray(channel_ids_r1, dtype=np.int64),
        "channel_ids_r4": np.asarray(channel_ids_r4, dtype=np.int64),
    }
