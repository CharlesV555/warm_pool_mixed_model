from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (
    ChannelBlock,
    ExperimentRunner,
    FixedPartitionStrategy,
    FoodUpperLimitRestriction,
    HybridStepper,
    ReactionNetworkData,
    SSAStepper,
    TrajectoryRecorder,
    assign_random_longest_catalyst_to_all_channels,
    build_reaction_rule_tables,
    generate_fixed_species_space,
)


ALPHABET = ("A", "B")
INITIAL_FOOD_COUNT = 100.0
FOOD_INFLOW_RATE = 5000.0
FOOD_MAX_COUNT = 100.0
INITIAL_COUNTS = {
    name: min(INITIAL_FOOD_COUNT, FOOD_MAX_COUNT)
    for name in ALPHABET
}


def build_network() -> ReactionNetworkData:
    space = generate_fixed_species_space(
        ALPHABET,
        max_len=3,
        initial_counts=INITIAL_COUNTS,
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.001,
        k_poly_right=0.001,
        k_frag_left=0.05,
        k_frag_right=0.05,
        k_inflow=FOOD_INFLOW_RATE,
        inflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name in ALPHABET
        ],
    )

    assign_random_longest_catalyst_to_all_channels(
        network,
        rng=np.random.default_rng(2026),
        log_mean=-4.0,
        log_sigma=0.5,
    )
    return network


def build_food_upper_limit_restriction(network: ReactionNetworkData) -> FoodUpperLimitRestriction:
    return FoodUpperLimitRestriction(
        {
            network.species_idx(name): FOOD_MAX_COUNT
            for name in ALPHABET
        }
    )


def print_summary(label: str, summary) -> None:
    print(f"\n{label}")
    print(f"t={summary.final_time:.4f}, steps={summary.n_steps}, events={summary.n_events}")
    print(f"seed={summary.metadata.get('seed')}, final_state={summary.final_state}")


def main() -> None:
    network = build_network()
    restriction = build_food_upper_limit_restriction(network)
    runner = ExperimentRunner()

    ssa_recorder = TrajectoryRecorder()
    ssa = runner.run_one(
        network,
        SSAStepper(),
        t_end=2.0,
        seed=123,
        recorder=ssa_recorder,
        restriction=restriction,
    )
    print_summary("SSA summary", ssa.summary)

    fast_local = int(network.left_add_local_id[network.species_idx("A"), network.species_idx("B")])
    fast_channel = network.channel_id(ChannelBlock.LEFT_ADD, fast_local)
    hybrid_recorder = TrajectoryRecorder()
    hybrid = runner.run_one(
        network,
        HybridStepper(),
        t_end=2.0,
        seed=456,
        dt=0.05,
        recorder=hybrid_recorder,
        restriction=restriction,
        partition_strategy=FixedPartitionStrategy([fast_channel]),
    )
    print_summary("Hybrid skeleton summary", hybrid.summary)


if __name__ == "__main__":
    main()
