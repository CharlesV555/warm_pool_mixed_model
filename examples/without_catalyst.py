from __future__ import annotations

import sys
from pathlib import Path

from polymer_sim.simulation.restriction import build_restriction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (
    ExperimentRunner,
    ReactionNetworkData,
    SSAStepper,
    TrajectoryRecorder,
    build_reaction_rule_tables,
    clear_all_catalysis,
    generate_fixed_species_space,
    save_trajectory_record,
)

MAX_LEN = 5
ALPHABET = ("A", "B")
T_END = 1000.0
SEED = 123
MAX_STEPS = 100_000_000
MAX_TIMES = 30.0  # Set to None to disable runtime cutoff
K_RIGHT_ADD = 0.1
K_NONFOOD_OUTFLOW = 0.5
FOOD_COUNT = 100.0
# Optional finite-reservoir variant:
# - add formal INFLOW channels in ReactionNetworkData.from_species_space(...)
# - add FOOD_INFLOW_RATE and FOOD_MAX_COUNT hyperparameters
# - pass FoodUpperLimitRestriction({network.species_idx(name): FOOD_MAX_COUNT, ...})
#   to ExperimentRunner.run_one(...).
CATALYSIS_MODE = "substrate_saturating"  # "linear" or "substrate_saturating"
SATURATION_ALPHA = 0.01
INITIAL_COUNTS = {"A": FOOD_COUNT, "B": FOOD_COUNT}


def build_without_catalyst_network() -> ReactionNetworkData:
    space = generate_fixed_species_space(
        ALPHABET,
        max_len=MAX_LEN,
        initial_counts=INITIAL_COUNTS,
    )
    tables = build_reaction_rule_tables(space)
    
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.0,
        k_poly_right=K_RIGHT_ADD,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_outflow=K_NONFOOD_OUTFLOW,
        outflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name not in ALPHABET
        ],
        catalysis_mode=CATALYSIS_MODE,
        saturation_alpha=SATURATION_ALPHA,
    )
    clear_all_catalysis(network)
    return network


def assert_no_catalysts(network: ReactionNetworkData) -> None:
    for channel_id in range(network.n_channels):
        if network.get_channel_catalysts(channel_id).size:
            raise AssertionError(f"unexpected catalyst on channel {channel_id}")


def print_run_summary(run_result, trajectory_record) -> None:
    print("\nSSA summary:")
    print(
        f"t={run_result.summary.final_time:.4f}, "
        f"steps={run_result.summary.n_steps}, "
        f"events={run_result.summary.n_events}, "
        f"seed={run_result.summary.metadata.get('seed')}, "
        f"stop_reason={run_result.summary.metadata.get('stop_reason')}"
    )
    print(
        f"trajectory points={trajectory_record.times.shape[0]}, "
        f"state shape={trajectory_record.states.shape}"
    )


def main() -> None:
    network = build_without_catalyst_network()
    assert_no_catalysts(network)

    print("Without-catalyst reaction system")
    print(f"alphabet={ALPHABET}, max_len={MAX_LEN}")
    print(f"n_species={network.n_species}, n_channels={network.n_channels}")
    print(f"catalysis_mode={network.catalysis_mode}, saturation_alpha={network.saturation_alpha}, catalysts=0")

    recorder = TrajectoryRecorder()
    result = ExperimentRunner().run_one(
        network,
        SSAStepper(),
        t_end=T_END,
        seed=SEED,
        recorder=recorder,
        max_steps=MAX_STEPS,
        max_runtime_seconds=MAX_TIMES,
        # restriction=FoodUpperLimitRestriction(...),
        # Use this comment-only hook when food is modeled as formal INFLOW
        # and should be capped without being replenished to a fixed count.
    )

    trajectory_record = recorder.finalize()
    output_path = PROJECT_ROOT / "examples" / "without_catalyst_trajectory.npz"
    save_trajectory_record(output_path, trajectory_record)

    print_run_summary(result, trajectory_record)
    print(f"trajectory saved to: {output_path}")


if __name__ == "__main__":
    main()
