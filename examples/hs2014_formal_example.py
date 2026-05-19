from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (
    assign_paper_minimal_catalysis,
    build_n3_wh_network,
    build_n3_wh_reactions,
    compute_max_raf,
    enumerate_irr_rafs,
)


def build_catalysis_map(network, reactions):
    return {
        reaction.channel_id: set(int(sid) for sid in network.get_channel_catalysts(reaction.channel_id))
        for reaction in reactions
    }


def main() -> None:
    network = build_n3_wh_network()
    reactions = build_n3_wh_reactions(network)
    assign_paper_minimal_catalysis(network)
    food = {network.species_idx("0"), network.species_idx("1")}
    catalysis_map = build_catalysis_map(network, reactions)

    max_raf = compute_max_raf(food, reactions, catalysis_map)
    irr_rafs = enumerate_irr_rafs(food, max_raf, catalysis_map)

    print("maxRAF:")
    for reaction in max_raf:
        print(f"  {reaction.reaction_id} [{reaction.category}]")

    print("\nirrRAFs:")
    for idx, subset in enumerate(irr_rafs):
        labels = ", ".join(reaction.reaction_id for reaction in subset)
        print(f"  irrRAF {idx}: {labels}")


if __name__ == "__main__":
    main()
