from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (
    BlendedHybridConfig,
    BlendedHybridStepper,
    ChannelBlock,
    ExperimentRunner,
    FoodUpperLimitRestriction,
    ReactionNetworkData,
    TrajectoryRecorder,
    build_reaction_rule_tables,
    clear_all_catalysis,
    generate_fixed_species_space,
    save_trajectory_record,
)


MAX_LEN = 5
ALPHABET = ("0", "1")
T_END = 20.0
SEED = 123
MAX_STEPS = 100_000_000
MAX_TIMES = 120.0

BACKGROUND_RATE = 0.01
CATALYTIC_STRENGTH = 1.0
K_NONFOOD_OUTFLOW = 1.5
CATALYSIS_MODE = "linear"
SATURATION_ALPHA = 0.01
INITIAL_FOOD_COUNT = 100.0
FOOD_INFLOW_RATE = 5000.0
# Per-food-species upper bound enforced after each simulation step. INFLOW
# remains a formal reaction channel; this cap only prevents the food inventory
# from accumulating above the configured system concentration.
FOOD_MAX_COUNT = 100.0
INITIAL_COUNTS = {
    name: min(INITIAL_FOOD_COUNT, FOOD_MAX_COUNT)
    for name in ALPHABET
}

K_LEFT_ADD = BACKGROUND_RATE
K_RIGHT_ADD = BACKGROUND_RATE
K_LEFT_SPLIT = BACKGROUND_RATE
K_RIGHT_SPLIT = BACKGROUND_RATE

CROSS_CATALYSIS_RULES = {
    "11111": "0",
    "00000": "1",
}

BLENDED_I1 = 110.0
BLENDED_I2 = 150.0
BLENDED_DT_CLE = 0.0001
BLENDED_DT_MACRO = 0.01

OUTPUT_PATH = EXAMPLES_DIR / "cross_catalysis_trajectory.npz"


def build_cross_catalysis_network() -> tuple[ReactionNetworkData, dict]:
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
    catalysis_result = assign_cross_terminal_catalysis(network)
    return network, catalysis_result


def assign_cross_terminal_catalysis(network: ReactionNetworkData) -> dict:
    clear_all_catalysis(network, rebuild=False)
    channels_by_catalyst: dict[str, list[int]] = {}
    primary_channels_by_catalyst: dict[str, list[int]] = {}

    for catalyst_name, added_monomer_name in CROSS_CATALYSIS_RULES.items():
        catalyst_sid = network.species_idx(catalyst_name)
        added_monomer_sid = network.species_idx(added_monomer_name)
        catalyzed_channels = _terminal_matched_addition_channels(
            network,
            added_monomer_sid,
            added_monomer_name,
        )
        for channel_id in catalyzed_channels:
            network.set_catalytic_strength(
                int(channel_id),
                catalyst_sid=catalyst_sid,
                strength=CATALYTIC_STRENGTH,
                rebuild=False,
                mirror_reverse=True,
            )
        primary_channels = [int(channel_id) for channel_id in catalyzed_channels]
        mirrored_channels = [
            int(reverse_channel_id)
            for channel_id in primary_channels
            for reverse_channel_id in network.get_reverse_channel_ids(channel_id)
        ]
        primary_channels_by_catalyst[catalyst_name] = primary_channels
        channels_by_catalyst[catalyst_name] = sorted(set(primary_channels + mirrored_channels))

    network.rebuild_dependency_indices()
    return {
        "method": "cross_terminal_matched_addition",
        "rules": dict(CROSS_CATALYSIS_RULES),
        "strength": CATALYTIC_STRENGTH,
        "mirror_reverse": True,
        "primary_channels_by_catalyst": primary_channels_by_catalyst,
        "channels_by_catalyst": channels_by_catalyst,
    }


def _terminal_matched_addition_channels(
    network: ReactionNetworkData,
    added_monomer_sid: int,
    added_monomer_name: str,
) -> np.ndarray:
    channels: list[int] = []

    for local_id, monomer_sid in enumerate(network.left_add_monomer):
        if int(monomer_sid) != int(added_monomer_sid):
            continue
        polymer_sid = int(network.left_add_species[int(local_id)])
        if not network.species_names[polymer_sid].startswith(added_monomer_name):
            continue
        channels.append(network.channel_id(ChannelBlock.LEFT_ADD, int(local_id)))

    for local_id, monomer_sid in enumerate(network.right_add_monomer):
        if int(monomer_sid) != int(added_monomer_sid):
            continue
        polymer_sid = int(network.right_add_species[int(local_id)])
        if not network.species_names[polymer_sid].endswith(added_monomer_name):
            continue
        channels.append(network.channel_id(ChannelBlock.RIGHT_ADD, int(local_id)))

    return np.asarray(channels, dtype=np.int64)


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
    # This restriction only caps food from above. It does not replenish food
    # when reactions consume it, so the actual input remains the INFLOW channel.
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
        "background_rate": BACKGROUND_RATE,
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
        "catalytic_strength": CATALYTIC_STRENGTH,
        "cross_catalysis_rules": dict(CROSS_CATALYSIS_RULES),
        "blended_i1": BLENDED_I1,
        "blended_i2": BLENDED_I2,
        "blended_dt_cle": BLENDED_DT_CLE,
        "blended_dt_macro": BLENDED_DT_MACRO,
    }


def print_run_summary(run_result, trajectory_record) -> None:
    print("\nBlended hybrid summary:")
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
    network, catalysis_result = build_cross_catalysis_network()
    restriction = build_food_upper_limit_restriction(network)
    stepper = BlendedHybridStepper(
        BlendedHybridConfig(
            i1=BLENDED_I1,
            i2=BLENDED_I2,
            dt_cle=BLENDED_DT_CLE,
            dt_macro=BLENDED_DT_MACRO,
            use_reaction_interval_dt=False,
            reaction_interval_update_steps=1,
        )
    )

    print("Cross-catalysis reaction system")
    print(f"alphabet={ALPHABET}, max_len={MAX_LEN}")
    print(f"n_species={network.n_species}, n_channels={network.n_channels}")
    print(f"catalysis_mode={network.catalysis_mode}, saturation_alpha={network.saturation_alpha}")
    print(f"background_rate={BACKGROUND_RATE}, catalytic_strength={CATALYTIC_STRENGTH}")
    print(
        f"initial_food_count={INITIAL_FOOD_COUNT}, "
        f"food_inflow_rate={FOOD_INFLOW_RATE}, "
        f"food_max_count={FOOD_MAX_COUNT}"
    )
    print(f"catalyst species={catalyst_species_names(network)}")
    print(f"catalyzed channels={catalyzed_channel_count(network)}")

    recorder = TrajectoryRecorder()
    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=T_END,
        seed=SEED,
        recorder=recorder,
        restriction=restriction,
        max_steps=MAX_STEPS,
        max_runtime_seconds=MAX_TIMES,
    )

    trajectory_record = recorder.finalize()
    trajectory_record.run_metadata["example_parameters"] = example_parameters()
    trajectory_record.run_metadata["catalysis_assignment"] = json_ready(catalysis_result)
    trajectory_record.run_metadata["catalyst_species_names"] = catalyst_species_names(network)
    save_trajectory_record(OUTPUT_PATH, trajectory_record)

    print_run_summary(result, trajectory_record)
    print(f"trajectory saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
