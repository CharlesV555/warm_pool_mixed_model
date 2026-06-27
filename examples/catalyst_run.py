from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (
    ExperimentRunner,
    FoodUpperLimitRestriction,
    ReactionNetworkData,
    SSAStepper,
    TrajectoryRecorder,
    assign_random_longest_catalyst_to_all_channels,
    assign_random_longest_catalysts_to_distinct_channels,
    build_reaction_rule_tables,
    generate_fixed_species_space,
    save_trajectory_record,
)

MAX_LEN = 5
ALPHABET = ("A", "B")
T_END = 0.2
SEED = 124
MAX_STEPS = 100_000_000
MAX_TIMES = 60.0  # Set to None to disable runtime cutoff
K_LEFT_ADD = 0.01
K_RIGHT_ADD = 0.01
K_LEFT_SPLIT = 0.01
K_RIGHT_SPLIT = 0.01
K_NONFOOD_OUTFLOW = 0.5
INITIAL_FOOD_COUNT = 100.0
FOOD_INFLOW_RATE = 5000.0
FOOD_MAX_COUNT = 100.0
CATALYSIS_MODE = "substrate_saturating"  # "linear" or "substrate_saturating"
SATURATION_ALPHA = 0.01
CATALYST_SEED = 2026
CATALYST_ASSIGNMENT_MODE = "single_longest_all_channels"
N_RANDOM_CATALYSTS = 16
CATALYST_LOG_MEAN = 0.0
CATALYST_LOG_SIGMA = 1.0
INITIAL_COUNTS = {
    name: min(INITIAL_FOOD_COUNT, FOOD_MAX_COUNT)
    for name in ALPHABET
}
OUTPUT_FILENAME = "catalyst_run_trajectory.npz"


def build_random_catalyst_network() -> tuple[ReactionNetworkData, dict]:
    space = generate_fixed_species_space(
        ALPHABET,
        max_len=MAX_LEN,
        initial_counts=INITIAL_COUNTS,
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=K_LEFT_ADD,
        k_poly_right=K_RIGHT_ADD,
        k_frag_left=K_LEFT_SPLIT,
        k_frag_right=K_RIGHT_SPLIT,
        k_outflow=K_NONFOOD_OUTFLOW,
        outflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name not in ALPHABET
        ],
        k_inflow=FOOD_INFLOW_RATE,
        inflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name in ALPHABET
        ],
        catalysis_mode=CATALYSIS_MODE,
        saturation_alpha=SATURATION_ALPHA,
    )
    rng = np.random.default_rng(CATALYST_SEED)
    catalysis_result = assign_random_catalysis(network, rng)  # modes: "single_longest_all_channels" or "distinct_longest_channels"
    return network, catalysis_result


def assign_random_catalysis(network: ReactionNetworkData, rng: np.random.Generator) -> dict:
    if CATALYST_ASSIGNMENT_MODE == "single_longest_all_channels":
        return assign_random_longest_catalyst_to_all_channels(
            network,
            rng=rng,
            log_mean=CATALYST_LOG_MEAN,
            log_sigma=CATALYST_LOG_SIGMA,
        )
    if CATALYST_ASSIGNMENT_MODE == "distinct_longest_channels":
        return assign_random_longest_catalysts_to_distinct_channels(
            network,
            N_RANDOM_CATALYSTS,
            rng=rng,
            log_mean=CATALYST_LOG_MEAN,
            log_sigma=CATALYST_LOG_SIGMA,
        )
    raise ValueError(
        "CATALYST_ASSIGNMENT_MODE must be "
        "'single_longest_all_channels' or 'distinct_longest_channels'"
    )


def catalyzed_channel_count(network: ReactionNetworkData) -> int:
    return sum(
        1
        for channel_id in range(network.n_channels)
        if network.get_channel_catalysts(channel_id).size > 0
    )


def catalyst_species_names(network: ReactionNetworkData) -> list[str]:
    catalyst_sids = []
    for channel_id in range(network.n_channels):
        catalyst_sids.extend(int(sid) for sid in network.get_channel_catalysts(channel_id))
    return [
        network.species_names[int(sid)]
        for sid in sorted(set(catalyst_sids))
    ]


def build_food_upper_limit_restriction(network: ReactionNetworkData) -> FoodUpperLimitRestriction:
    return FoodUpperLimitRestriction(
        {
            network.species_idx(name): FOOD_MAX_COUNT
            for name in ALPHABET
        }
    )


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def example_parameters() -> dict:
    return {
        "max_len": MAX_LEN,
        "alphabet": list(ALPHABET),
        "t_end": T_END,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "max_times": MAX_TIMES,
        "k_left_add": K_LEFT_ADD,
        "k_right_add": K_RIGHT_ADD,
        "k_left_split": K_LEFT_SPLIT,
        "k_right_split": K_RIGHT_SPLIT,
        "k_nonfood_outflow": K_NONFOOD_OUTFLOW,
        "initial_food_count": INITIAL_FOOD_COUNT,
        "effective_initial_counts": dict(INITIAL_COUNTS),
        "food_inflow_rate": FOOD_INFLOW_RATE,
        "food_max_count": FOOD_MAX_COUNT,
        "catalysis_mode": CATALYSIS_MODE,
        "saturation_alpha": SATURATION_ALPHA,
        "catalyst_seed": CATALYST_SEED,
        "catalyst_assignment_mode": CATALYST_ASSIGNMENT_MODE,
        "n_random_catalysts": N_RANDOM_CATALYSTS,
        "catalyst_log_mean": CATALYST_LOG_MEAN,
        "catalyst_log_sigma": CATALYST_LOG_SIGMA,
    }


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
    network, catalysis_result = build_random_catalyst_network()

    print("Random-catalyst reaction system")
    print(f"alphabet={ALPHABET}, max_len={MAX_LEN}")
    print(f"n_species={network.n_species}, n_channels={network.n_channels}")
    print(f"catalysis_mode={network.catalysis_mode}, saturation_alpha={network.saturation_alpha}")
    print(f"catalyst_assignment_mode={CATALYST_ASSIGNMENT_MODE}")
    print(
        f"initial_food_count={INITIAL_FOOD_COUNT}, "
        f"food_inflow_rate={FOOD_INFLOW_RATE}, "
        f"food_max_count={FOOD_MAX_COUNT}"
    )
    print(f"catalyst species={catalyst_species_names(network)}")
    print(f"catalyzed channels={catalyzed_channel_count(network)}")

    recorder = TrajectoryRecorder()
    restriction = build_food_upper_limit_restriction(network)
    t0 = perf_counter()
    build_elapsed = perf_counter() - t0

    result = ExperimentRunner().run_one(
        network,
        SSAStepper(),
        t_end=T_END,
        seed=SEED,
        recorder=recorder,
        max_steps=MAX_STEPS,
        restriction=restriction,
        max_runtime_seconds=MAX_TIMES,
        timing_report=True,
        timing_report_dir="timing_reports",
        network_build_elapsed_seconds=build_elapsed,
    )

    trajectory_record = recorder.finalize()
    trajectory_record.run_metadata["example_parameters"] = example_parameters()
    trajectory_record.run_metadata["catalysis_assignment"] = json_ready(catalysis_result)
    trajectory_record.run_metadata["catalyst_species_names"] = catalyst_species_names(network)

    output_path = PROJECT_ROOT / "examples" / OUTPUT_FILENAME
    save_trajectory_record(output_path, trajectory_record)

    print_run_summary(result, trajectory_record)
    print(f"trajectory saved to: {output_path}")


if __name__ == "__main__":
    main()
