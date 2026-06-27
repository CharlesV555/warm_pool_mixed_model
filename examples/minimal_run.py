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
    HybridStepper,
    ReactionNetworkData,
    SSAStepper,
    TrajectoryRecorder,
    assign_random_longest_catalyst_to_all_channels,
    build_reaction_rule_tables,
    generate_fixed_species_space,
)


def build_network() -> ReactionNetworkData:
    space = generate_fixed_species_space(
        ["A", "B"],
        max_len=3,
        initial_counts={"A": 40, "B": 30},
    )
    # This minimal smoke example has no external food reservoir. To test a
    # finite food reservoir here, add formal INFLOW channels in
    # ReactionNetworkData.from_species_space(...) and pass
    # FoodUpperLimitRestriction to ExperimentRunner.run_one(...).
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.001,
        k_poly_right=0.001,
        k_frag_left=0.05,
        k_frag_right=0.05,
    )

    assign_random_longest_catalyst_to_all_channels(
        network,
        rng=np.random.default_rng(2026),
        log_mean=-4.0,
        log_sigma=0.5,
    )
    return network


def print_summary(label: str, summary) -> None:
    print(f"\n{label}")
    print(f"t={summary.final_time:.4f}, steps={summary.n_steps}, events={summary.n_events}")
    print(f"seed={summary.metadata.get('seed')}, final_state={summary.final_state}")


def main() -> None:
    network = build_network()
    runner = ExperimentRunner()

    ssa_recorder = TrajectoryRecorder()
    ssa = runner.run_one(
        network,
        SSAStepper(),
        t_end=2.0,
        seed=123,
        recorder=ssa_recorder,
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
        partition_strategy=FixedPartitionStrategy([fast_channel]),
    )
    print_summary("Hybrid skeleton summary", hybrid.summary)


if __name__ == "__main__":
    main()
