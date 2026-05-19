from __future__ import annotations

from itertools import combinations

from polymer_sim.model.wills_henderson import WHReaction


def closure(food_sids: set[int], reaction_subset: list[WHReaction]) -> set[int]:
    reachable = set(int(sid) for sid in food_sids)
    changed = True
    while changed:
        changed = False
        for reaction in reaction_subset:
            if all(int(reactant) in reachable for reactant in reaction.reactants):
                before = len(reachable)
                reachable.update(int(product) for product in reaction.products)
                if len(reachable) > before:
                    changed = True
    return reachable


def is_raf_subset(
    food_sids: set[int],
    reaction_subset: list[WHReaction],
    catalysis_map: dict[int, set[int]],
) -> bool:
    if not reaction_subset:
        return False
    reachable = closure(food_sids, reaction_subset)
    for reaction in reaction_subset:
        if not all(int(reactant) in reachable for reactant in reaction.reactants):
            return False
        catalysts = catalysis_map.get(int(reaction.channel_id), set())
        if not catalysts:
            return False
        if not any(int(catalyst_sid) in reachable for catalyst_sid in catalysts):
            return False
    return True


def compute_max_raf(
    food_sids: set[int],
    reactions: list[WHReaction],
    catalysis_map: dict[int, set[int]],
) -> list[WHReaction]:
    active = list(reactions)
    changed = True
    while changed:
        changed = False
        reachable = closure(food_sids, active)
        kept: list[WHReaction] = []
        for reaction in active:
            reactants_ok = all(int(reactant) in reachable for reactant in reaction.reactants)
            catalysts = catalysis_map.get(int(reaction.channel_id), set())
            catalyst_ok = any(int(catalyst_sid) in reachable for catalyst_sid in catalysts)
            if reactants_ok and catalyst_ok:
                kept.append(reaction)
            else:
                changed = True
        active = kept
    return active


def enumerate_irr_rafs(
    food_sids: set[int],
    max_raf: list[WHReaction],
    catalysis_map: dict[int, set[int]],
) -> list[list[WHReaction]]:
    raf_subsets: list[list[WHReaction]] = []
    for size in range(1, len(max_raf) + 1):
        for combo in combinations(max_raf, size):
            subset = list(combo)
            if is_raf_subset(food_sids, subset, catalysis_map):
                raf_subsets.append(subset)

    irr_rafs: list[list[WHReaction]] = []
    for subset in raf_subsets:
        subset_ids = {reaction.channel_id for reaction in subset}
        has_smaller_raf = False
        for other in raf_subsets:
            other_ids = {reaction.channel_id for reaction in other}
            if other_ids < subset_ids:
                has_smaller_raf = True
                break
        if not has_smaller_raf:
            irr_rafs.append(subset)
    return irr_rafs
