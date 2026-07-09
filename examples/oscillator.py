from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

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


MAX_LEN = 3
ALPHABET = ("A", "B")
T_END = 200.0
SEED = 123
MAX_STEPS = 100_000_000
MAX_TIMES = 60.0

BACKGROUND_RATE = 0.001
K_NONFOOD_OUTFLOW = 1.0
CATALYSIS_MODE = "linear"
SATURATION_ALPHA = 0.01
INITIAL_FOOD_COUNT = 50.0
FOOD_INFLOW_RATE = 500.0
FOOD_MAX_COUNT = INITIAL_FOOD_COUNT
INITIAL_COUNTS = {
    name: min(INITIAL_FOOD_COUNT, FOOD_MAX_COUNT)
    for name in ALPHABET
}

K_LEFT_ADD = BACKGROUND_RATE
K_RIGHT_ADD = BACKGROUND_RATE
K_LEFT_SPLIT = BACKGROUND_RATE
K_RIGHT_SPLIT = BACKGROUND_RATE

A_CATALYST_NAME = "A" * MAX_LEN
B_CATALYST_NAME = "B" * MAX_LEN
A_PROMOTES_B_STRENGTH = 10.0
A_SELF_PROMOTION_STRENGTH = 2.0
B_INHIBITS_A_STRENGTH = 1000.0 # 促进裂解
MIRROR_REVERSE = True

BLENDED_I1 = 110.0
BLENDED_I2 = 150.0
BLENDED_DT_CLE = 0.003981
BLENDED_DT_MACRO = 0.01

OUTPUT_PATH = EXAMPLES_DIR / "oscillator_trajectory.npz"


def build_oscillator_network() -> tuple[ReactionNetworkData, dict]:
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
    catalysis_result = assign_oscillator_catalysis(network)
    return network, catalysis_result


def assign_oscillator_catalysis(network: ReactionNetworkData) -> dict:
    clear_all_catalysis(network, rebuild=False)
    a_catalyst_sid = network.species_idx(A_CATALYST_NAME)
    b_catalyst_sid = network.species_idx(B_CATALYST_NAME)
    a_growth_channels = _homopolymer_growth_channels(network, "A")
    b_growth_channels = _homopolymer_growth_channels(network, "B", max_target_len=MAX_LEN - 1)
    a_n_split_channels = _source_split_channels(network, A_CATALYST_NAME)

    assignments = [
        {
            "name": "A_n_promotes_B_series_growth",
            "catalyst": A_CATALYST_NAME,
            "target": "B1..B{n-1}",
            "catalyst_sid": a_catalyst_sid,
            "strength": A_PROMOTES_B_STRENGTH,
            "channels": b_growth_channels,
        },
        {
            "name": "B_n_promotes_A_n_depolymerization/inhibits_A_n_growth",
            "catalyst": B_CATALYST_NAME,
            "target": A_CATALYST_NAME,
            "catalyst_sid": b_catalyst_sid,
            "strength": B_INHIBITS_A_STRENGTH,
            "channels": a_n_split_channels,
        },
        {
            "name": "A_n_self_promotes_A_series_growth",
            "catalyst": A_CATALYST_NAME,
            "target": "A1..An",
            "catalyst_sid": a_catalyst_sid,
            "strength": A_SELF_PROMOTION_STRENGTH,
            "channels": a_growth_channels,
        },
    ]

    result_assignments = []
    for assignment in assignments:
        primary_channels = [int(channel_id) for channel_id in assignment["channels"]]
        for channel_id in primary_channels:
            network.set_catalytic_strength(
                channel_id,
                catalyst_sid=int(assignment["catalyst_sid"]),
                strength=float(assignment["strength"]),
                rebuild=False,
                mirror_reverse=MIRROR_REVERSE,
            )
        mirrored_channels = [
            int(reverse_channel_id)
            for channel_id in primary_channels
            for reverse_channel_id in network.get_reverse_channel_ids(channel_id)
        ] if MIRROR_REVERSE else []
        result_assignments.append(
            {
                "name": assignment["name"],
                "catalyst": assignment["catalyst"],
                "target": assignment["target"],
                "strength": float(assignment["strength"]),
                "primary_channels": primary_channels,
                "mirrored_channels": sorted(set(mirrored_channels)),
                "all_channels": sorted(set(primary_channels + mirrored_channels)),
            }
        )

    network.rebuild_dependency_indices()
    return {
        "method": "longest_chain_oscillator",
        "a_catalyst": A_CATALYST_NAME,
        "b_catalyst": B_CATALYST_NAME,
        "mirror_reverse": MIRROR_REVERSE,
        "assignments": result_assignments,
    }


def _homopolymer_growth_channels(
    network: ReactionNetworkData,
    monomer_name: str,
    *,
    max_target_len: int = MAX_LEN,
) -> np.ndarray:
    monomer_sid = network.species_idx(monomer_name)
    channels: list[int] = []
    for local_id, monomer in enumerate(network.left_add_monomer):
        if int(monomer) != monomer_sid:
            continue
        source_sid = int(network.left_add_species[int(local_id)])
        target_sid = int(network.left_add_target[int(local_id)])
        if _is_homopolymer_extension(network, source_sid, target_sid, monomer_name, max_target_len=max_target_len):
            channels.append(network.channel_id(ChannelBlock.LEFT_ADD, int(local_id)))

    for local_id, monomer in enumerate(network.right_add_monomer):
        if int(monomer) != monomer_sid:
            continue
        source_sid = int(network.right_add_species[int(local_id)])
        target_sid = int(network.right_add_target[int(local_id)])
        if _is_homopolymer_extension(network, source_sid, target_sid, monomer_name, max_target_len=max_target_len):
            channels.append(network.channel_id(ChannelBlock.RIGHT_ADD, int(local_id)))

    return np.asarray(sorted(set(channels)), dtype=np.int64)


def _source_split_channels(network: ReactionNetworkData, source_name: str) -> np.ndarray:
    source_sid = network.species_idx(source_name)
    channels: list[int] = []
    for local_id, source in enumerate(network.left_split_source):
        if int(source) == source_sid:
            channels.append(network.channel_id(ChannelBlock.LEFT_SPLIT, int(local_id)))
    for local_id, source in enumerate(network.right_split_source):
        if int(source) == source_sid:
            channels.append(network.channel_id(ChannelBlock.RIGHT_SPLIT, int(local_id)))
    return np.asarray(sorted(set(channels)), dtype=np.int64)


def _is_homopolymer_extension(
    network: ReactionNetworkData,
    source_sid: int,
    target_sid: int,
    monomer_name: str,
    *,
    max_target_len: int = MAX_LEN,
) -> bool:
    source = network.species_names[int(source_sid)]
    target = network.species_names[int(target_sid)]
    return (
        source == monomer_name * len(source)
        and target == monomer_name * len(target)
        and len(target) == len(source) + 1
        and len(target) <= int(max_target_len)
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
            network.species_idx(name): INITIAL_FOOD_COUNT
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
        "a_catalyst": A_CATALYST_NAME,
        "b_catalyst": B_CATALYST_NAME,
        "a_promotes_b_strength": A_PROMOTES_B_STRENGTH,
        "a_self_promotion_strength": A_SELF_PROMOTION_STRENGTH,
        "b_inhibits_a_strength": B_INHIBITS_A_STRENGTH,
        "mirror_reverse": MIRROR_REVERSE,
        "blended_i1": BLENDED_I1,
        "blended_i2": BLENDED_I2,
        "blended_dt_cle": BLENDED_DT_CLE,
        "blended_dt_macro": BLENDED_DT_MACRO,
    }


def print_run_summary(run_result, trajectory_record) -> None:
    print("\nOscillator blended hybrid summary:")
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
    network, catalysis_result = build_oscillator_network()
    restriction = build_food_upper_limit_restriction(network)
    stepper = BlendedHybridStepper(
        BlendedHybridConfig(
            i1=BLENDED_I1,
            i2=BLENDED_I2,
            dt_cle=BLENDED_DT_CLE,
            dt_macro=BLENDED_DT_MACRO,
            use_reaction_interval_dt=False,
            reaction_interval_update_steps=1,
            beta_species_mode="reactants",
        )
    )
    print("Oscillator reaction system")
    print(f"alphabet={ALPHABET}, max_len={MAX_LEN}")
    print(f"n_species={network.n_species}, n_channels={network.n_channels}")
    print(f"catalysis_mode={network.catalysis_mode}, saturation_alpha={network.saturation_alpha}")
    print(f"catalyst species={catalyst_species_names(network)}")
    print(f"catalyzed channels={catalyzed_channel_count(network)}")

    recorder = TrajectoryRecorder()
    t0 = perf_counter()
    build_elapsed = perf_counter() - t0
    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=T_END,
        seed=SEED,
        recorder=recorder,
        restriction=restriction,
        max_steps=MAX_STEPS,
        max_runtime_seconds=MAX_TIMES,
        network_build_elapsed_seconds=build_elapsed,
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
