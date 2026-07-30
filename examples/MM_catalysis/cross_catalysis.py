from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

from time import perf_counter

import numpy as np

from polymer_sim import (
    ChannelBlock,
    ElementaryExpansionConfig,
    ExperimentRunner,
    FastNetworkReportRecorder,
    FiniteMarkovConfig,
    PDMPConfig,
    PDMPStepper,
    ReactionNetworkData,
    ScalingPDMPConfig,
    ScalingPDMPPartitionStrategy,
    TrajectoryRecorder,
    build_elementary_mass_action_network,
    build_reaction_rule_tables,
    clear_all_catalysis,
    format_stepper_info,
    generate_fixed_species_space,
    save_trajectory_record,
)

# 其它模拟模式保留为注释，便于对照时手动切换。
# from polymer_sim import (
#     BlendedHybridConfig,
#     BlendedHybridStepper,
#     NRMBlendedHybridStepper,
#     OptimizedNRMStepper,
#     SSAStepper,
# )

MAX_LEN = 4
ALPHABET = ("0", "1")
T_END = 200.0
SEED = 123
MAX_STEPS = 100_000_000
MAX_TIMES = 100.0

BACKGROUND_RATE = 2
K_NONFOOD_OUTFLOW = 1.5
CATALYSIS_MODE = "linear"
SATURATION_ALPHA = 0.01
INITIAL_FOOD_COUNT = 1000.0
K_FOOD_ZERO_ORDER_INFLOW = 1000000.0
FOOD_INFLOW_RATE = K_FOOD_ZERO_ORDER_INFLOW
FOOD_INFLOW_HILL_COEFFICIENT = 2.0
# PDMP 论文算法要求 elementary mass-action。启用该开关时，food inflow
# 在 elementary 预计算网络中会变成标准零阶反应：∅ -> food。
# 如果要回到原来的 Hill-like capped inflow 对照，应使用原 ReactionNetworkData
# 配合 SSA/NRM/Hybrid，而不是 PDMPStepper。
USE_STANDARD_ZERO_ORDER_INFLOW = True
# Per-food-species soft upper bound used only by the original polymer network
# when USE_STANDARD_ZERO_ORDER_INFLOW=False.
FOOD_MAX_COUNT = INITIAL_FOOD_COUNT
INITIAL_COUNTS = {
    name: min(INITIAL_FOOD_COUNT, FOOD_MAX_COUNT)
    for name in ALPHABET
}

K_LEFT_ADD = 0.01
K_RIGHT_ADD = 0.01
K_LEFT_SPLIT = 0.011
K_RIGHT_SPLIT = 0.011

CROSS_CATALYSIS_RULES = {
    # "1111": "1",
    "0000": "0",
}

# Catalytic quasi-steady-state parameterization used by the PDMP elementary view:
#   C + S <-> C:S       with k1, k_-1
#   C:S + M -> C + P   with k2
#
# For two symmetric product exits with the same monomer M,
#   a_each = k1*k2*C*S*M / (k_-1 + 2*k2*M)
# Define gamma = k1 / 2 and Ksat = k_-1 / (2*k2), so:
#   k1 = 2*gamma
#   k_-1 = 2*k2*Ksat
#
# Only gamma is intended to vary across catalytic assignments.  The current
# test/example setting uses a uniform gamma=1000.  To randomize later, change
# CATALYTIC_GAMMA_ASSIGNMENT_MODE and _sample_catalytic_gamma(...); keep gamma
# assigned per catalyst-substrate complex pair, not independently per outgoing
# channel, because one explicit complex has one binding rate.
CATALYTIC_GAMMA = 1e5
CATALYTIC_GAMMA_ASSIGNMENT_MODE = "constant"
CATALYTIC_GAMMA_LOGNORMAL_MEAN = float(np.log(CATALYTIC_GAMMA))
CATALYTIC_GAMMA_LOGNORMAL_SIGMA = 0.5
CATALYTIC_GAMMA_SEED = SEED

# Backward-compatible alias for older metadata/plotting code.  In this example
# "strength" means gamma, not the explicit catalytic turnover rate.
CATALYTIC_STRENGTH = CATALYTIC_GAMMA

BLENDED_I1 = 100.0
BLENDED_I2 = 150.0
BLENDED_DT_CLE = 0.00033981
BLENDED_DT_MACRO = 0.01

ELEMENTARY_CATALYTIC_TURNOVER_RATE_K2 = 5.0
ELEMENTARY_COMPLEX_KSAT = 1000.0
ELEMENTARY_CATALYST_BINDING_RATE_PER_GAMMA = 2.0
ELEMENTARY_CATALYST_BINDING_RATE = ELEMENTARY_CATALYST_BINDING_RATE_PER_GAMMA * CATALYTIC_GAMMA
ELEMENTARY_CATALYST_UNBINDING_RATE = (
    2.0 * ELEMENTARY_CATALYTIC_TURNOVER_RATE_K2 * ELEMENTARY_COMPLEX_KSAT
)
ELEMENTARY_CATALYTIC_TURNOVER_SCALE = 1.0

# Algorithm 2 example settings.  PDMPStepper executes the adaptive PDMP loop;
# ScalingPDMPPartitionStrategy provides the Algorithm-3 scaling partition used
# inside that loop.  Algorithm 4 subnetwork selection can be enabled here for a
# later comparison, but the default below keeps this file focused on Algorithm 3.
PDMP_ODE_STEP = 0.01
PDMP_N0 = 100.0

# Algorithm 3 hyperparameters:
# mu: continuous copy-number scale threshold.  Species with alpha_i >= mu are
# treated as continuous species.
# eta: adaptation scale threshold.  Continuous species keep bounds
# [N0^(alpha_i - eta), N0^(alpha_i + eta)] before repartitioning.
# delta: reaction relaxation/slack threshold.  A reaction is continuous only
# when every species it changes has alpha_i > delta.
PDMP_CONTINUOUS_COPY_NUMBER_SCALE_THRESHOLD_MU = 1.0
PDMP_ADAPTATION_SCALE_THRESHOLD_ETA = 0.9
PDMP_REACTION_RELAXATION_DELTA = 0.9

# Backward-compatible aliases used by older metadata/plotting code.
PDMP_SPECIES_EXPONENT_THRESHOLD = PDMP_CONTINUOUS_COPY_NUMBER_SCALE_THRESHOLD_MU
PDMP_REACTION_EXPONENT_THRESHOLD = PDMP_REACTION_RELAXATION_DELTA
PDMP_BOUND_FACTOR = PDMP_N0 ** PDMP_ADAPTATION_SCALE_THRESHOLD_ETA
PDMP_USE_LP = True
PDMP_REPARTITION_ON_EVENT = False
PDMP_REPARTITION_ON_BOUNDS = True
PDMP_ENABLE_FAST_SUBNETWORKS = False
PDMP_FAST_SUBNETWORK_THRESHOLD = 1.0
PDMP_FAST_SUBNETWORK_MAX_SIZE = 3

OUTPUT_PATH = EXAMPLES_DIR / "cross_catalysis_trajectory.npz"
FAST_NETWORK_REPORT_PATH = EXAMPLES_DIR / "cross_catalysis_fast_network_report.txt"

# Set this to False to disable the finite-Markov fast-subnetwork diagnostic.
# The report is observational only: it does not change the PDMP stepper state,
# partition, or propensities.
fast_network_report = False
FAST_NETWORK_REPORT_INTERVAL_EVENTS = 1000
FAST_NETWORK_REPORT_MAX_STATES = 512
FAST_NETWORK_REPORT_MAX_TOTAL_INTERNAL_COUNT = 10_000


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
        k_inflow=K_FOOD_ZERO_ORDER_INFLOW if USE_STANDARD_ZERO_ORDER_INFLOW else FOOD_INFLOW_RATE,
        inflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name in ALPHABET
        ],
        inflow_capacity=None if USE_STANDARD_ZERO_ORDER_INFLOW else FOOD_MAX_COUNT,
        inflow_hill_coefficient=FOOD_INFLOW_HILL_COEFFICIENT,
        catalysis_mode=CATALYSIS_MODE,
        saturation_alpha=SATURATION_ALPHA,
    )
    gamma_rng = np.random.default_rng(CATALYTIC_GAMMA_SEED)
    catalysis_result = assign_cross_terminal_catalysis(network, rng=gamma_rng)
    return network, catalysis_result


def assign_cross_terminal_catalysis(network: ReactionNetworkData, *, rng: np.random.Generator | None = None) -> dict:
    clear_all_catalysis(network, rebuild=False)
    channels_by_catalyst: dict[str, list[int]] = {}
    primary_channels_by_catalyst: dict[str, list[int]] = {}
    gamma_by_complex_pair: dict[tuple[str, str], float] = {}
    gamma_by_primary_channel: dict[int, float] = {}

    for catalyst_name, added_monomer_name in CROSS_CATALYSIS_RULES.items():
        catalyst_sid = network.species_idx(catalyst_name)
        added_monomer_sid = network.species_idx(added_monomer_name)
        catalyzed_channels = _terminal_matched_addition_channels(
            network,
            added_monomer_sid,
            added_monomer_name,
        )
        for channel_id in catalyzed_channels:
            substrate_sid = int(network.get_channel_main_species(int(channel_id)))
            substrate_name = network.species_names[substrate_sid]
            pair_key = (catalyst_name, substrate_name)
            if pair_key not in gamma_by_complex_pair:
                gamma_by_complex_pair[pair_key] = _sample_catalytic_gamma(rng)
            gamma = float(gamma_by_complex_pair[pair_key])
            network.set_catalytic_strength(
                int(channel_id),
                catalyst_sid=catalyst_sid,
                strength=gamma,
                rebuild=False,
                mirror_reverse=True,
            )
            gamma_by_primary_channel[int(channel_id)] = gamma
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
        "gamma_assignment_mode": CATALYTIC_GAMMA_ASSIGNMENT_MODE,
        "gamma_constant": CATALYTIC_GAMMA,
        "gamma_lognormal_mean": CATALYTIC_GAMMA_LOGNORMAL_MEAN,
        "gamma_lognormal_sigma": CATALYTIC_GAMMA_LOGNORMAL_SIGMA,
        "gamma_by_complex_pair": {
            f"{catalyst}|{substrate}": float(gamma)
            for (catalyst, substrate), gamma in gamma_by_complex_pair.items()
        },
        "gamma_by_primary_channel": {int(channel_id): float(gamma) for channel_id, gamma in gamma_by_primary_channel.items()},
        "strength": CATALYTIC_STRENGTH,
        "mirror_reverse": True,
        "primary_channels_by_catalyst": primary_channels_by_catalyst,
        "channels_by_catalyst": channels_by_catalyst,
    }


def _sample_catalytic_gamma(rng: np.random.Generator | None) -> float:
    mode = str(CATALYTIC_GAMMA_ASSIGNMENT_MODE).lower()
    if mode == "constant":
        return float(CATALYTIC_GAMMA)
    if mode == "lognormal":
        generator = rng if rng is not None else np.random.default_rng(CATALYTIC_GAMMA_SEED)
        return float(
            generator.lognormal(
                mean=float(CATALYTIC_GAMMA_LOGNORMAL_MEAN),
                sigma=float(CATALYTIC_GAMMA_LOGNORMAL_SIGMA),
            )
        )
    raise ValueError("CATALYTIC_GAMMA_ASSIGNMENT_MODE must be 'constant' or 'lognormal'")


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


# Legacy restriction entry point.
# This PDMP example does not pass a runner restriction.  Food supply is part of
# the reaction network itself.  With USE_STANDARD_ZERO_ORDER_INFLOW=True, the
# elementary network uses paper-style zero-order inflow; with it disabled, use
# the original polymer network plus SSA/NRM/Hybrid for Hill-like capped inflow.


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
        "k_food_zero_order_inflow": K_FOOD_ZERO_ORDER_INFLOW,
        "food_inflow_rate": FOOD_INFLOW_RATE,
        "food_inflow_hill_coefficient": FOOD_INFLOW_HILL_COEFFICIENT,
        "use_standard_zero_order_inflow": USE_STANDARD_ZERO_ORDER_INFLOW,
        "food_max_count": FOOD_MAX_COUNT,
        "catalysis_mode": CATALYSIS_MODE,
        "saturation_alpha": SATURATION_ALPHA,
        "catalytic_strength": CATALYTIC_STRENGTH,
        "catalytic_gamma": CATALYTIC_GAMMA,
        "catalytic_gamma_assignment_mode": CATALYTIC_GAMMA_ASSIGNMENT_MODE,
        "catalytic_gamma_lognormal_mean": CATALYTIC_GAMMA_LOGNORMAL_MEAN,
        "catalytic_gamma_lognormal_sigma": CATALYTIC_GAMMA_LOGNORMAL_SIGMA,
        "catalytic_gamma_seed": CATALYTIC_GAMMA_SEED,
        "cross_catalysis_rules": dict(CROSS_CATALYSIS_RULES),
        "elementary_catalyst_binding_rate": ELEMENTARY_CATALYST_BINDING_RATE,
        "elementary_catalyst_binding_rate_per_gamma": ELEMENTARY_CATALYST_BINDING_RATE_PER_GAMMA,
        "elementary_catalyst_unbinding_rate": ELEMENTARY_CATALYST_UNBINDING_RATE,
        "elementary_catalytic_turnover_rate_k2": ELEMENTARY_CATALYTIC_TURNOVER_RATE_K2,
        "elementary_complex_Ksat": ELEMENTARY_COMPLEX_KSAT,
        "elementary_catalytic_turnover_scale": ELEMENTARY_CATALYTIC_TURNOVER_SCALE,
        "pdmp_ode_step": PDMP_ODE_STEP,
        "pdmp_N0": PDMP_N0,
        "pdmp_continuous_copy_number_scale_threshold_mu": PDMP_CONTINUOUS_COPY_NUMBER_SCALE_THRESHOLD_MU,
        "pdmp_adaptation_scale_threshold_eta": PDMP_ADAPTATION_SCALE_THRESHOLD_ETA,
        "pdmp_reaction_relaxation_delta": PDMP_REACTION_RELAXATION_DELTA,
        "pdmp_species_exponent_threshold": PDMP_SPECIES_EXPONENT_THRESHOLD,
        "pdmp_reaction_exponent_threshold": PDMP_REACTION_EXPONENT_THRESHOLD,
        "pdmp_bound_factor": PDMP_BOUND_FACTOR,
        "pdmp_use_lp": PDMP_USE_LP,
        "pdmp_repartition_on_event": PDMP_REPARTITION_ON_EVENT,
        "pdmp_repartition_on_bounds": PDMP_REPARTITION_ON_BOUNDS,
        "pdmp_enable_fast_subnetworks": PDMP_ENABLE_FAST_SUBNETWORKS,
        "pdmp_fast_subnetwork_threshold": PDMP_FAST_SUBNETWORK_THRESHOLD,
        "pdmp_fast_subnetwork_max_size": PDMP_FAST_SUBNETWORK_MAX_SIZE,
        "blended_i1": BLENDED_I1,
        "blended_i2": BLENDED_I2,
        "blended_dt_cle": BLENDED_DT_CLE,
        "blended_dt_macro": BLENDED_DT_MACRO,
        "fast_network_report": bool(fast_network_report),
        "fast_network_report_path": str(FAST_NETWORK_REPORT_PATH),
        "fast_network_report_interval_events": FAST_NETWORK_REPORT_INTERVAL_EVENTS,
        "fast_network_report_max_states": FAST_NETWORK_REPORT_MAX_STATES,
        "fast_network_report_max_total_internal_count": FAST_NETWORK_REPORT_MAX_TOTAL_INTERNAL_COUNT,
    }


def print_run_summary(run_result, trajectory_record) -> None:
    print("\nPDMP Algorithm 2 summary:")
    print(format_stepper_info(run_result.summary.metadata))
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
    build_started_at = perf_counter()
    network, catalysis_result = build_cross_catalysis_network()

    # 关键切换点 1：换网络。
    # PDMPStepper 不能直接吃原始 ReactionNetworkData，因为论文算法 2/3
    # 要求 elementary mass-action 形式。这里在开局预计算阶段显式引入：
    #   C + S -> C:S
    #   C:S -> C + S
    #   C:S + residual -> C + product
    # 以及标准零阶 food inflow。
    elementary = build_elementary_mass_action_network(
        network,
        ElementaryExpansionConfig(
            standard_zero_order_inflow=USE_STANDARD_ZERO_ORDER_INFLOW,
            catalyst_binding_rate=ELEMENTARY_CATALYST_BINDING_RATE,
            catalyst_binding_rate_per_strength=ELEMENTARY_CATALYST_BINDING_RATE_PER_GAMMA,
            catalyst_unbinding_rate=ELEMENTARY_CATALYST_UNBINDING_RATE,
            catalytic_turnover_rate=ELEMENTARY_CATALYTIC_TURNOVER_RATE_K2,
            catalytic_turnover_scale=ELEMENTARY_CATALYTIC_TURNOVER_SCALE,
        ),
    )
    build_elapsed = perf_counter() - build_started_at

    # restriction = build_food_upper_limit_restriction(network)

    # 关键切换点 2：换模拟方法。
    # 下面这些原始 polymer-network 模式只作为对照保留；若启用它们，应把
    # runner.run_one(...) 里的 network 从 elementary 改回 network。
    # stepper = BlendedHybridStepper(
    #     BlendedHybridConfig(
    #         i1=BLENDED_I1,
    #         i2=BLENDED_I2,
    #         dt_cle=BLENDED_DT_CLE,
    #         dt_macro=BLENDED_DT_MACRO,
    #         use_reaction_interval_dt=False,
    #         reaction_interval_update_steps=1,
    #         beta_species_mode="reactants",
    #     )
    # )
    # stepper = SSAStepper()
    # stepper = OptimizedNRMStepper()
    # stepper = NRMBlendedHybridStepper(
    #     BlendedHybridConfig(
    #         i1=BLENDED_I1,
    #         i2=BLENDED_I2,
    #         dt_cle=BLENDED_DT_CLE,
    #         dt_macro=BLENDED_DT_MACRO,
    #         use_reaction_interval_dt=False,
    #         reaction_interval_update_steps=1,
    #         beta_species_mode="reactants",
    #     )
    # )

    # 关键切换点 3：换 Algorithm 2 的分区策略。
    # 当前使用 ScalingPDMPPartitionStrategy，即论文 Algorithm 3：
    #   mu 控制 species 是否进入 continuous copy-number scale；
    #   eta 控制 adaptation/repartition 的宽松区间；
    #   delta 控制 reaction 是否可进入 continuous channel 集合。
    # 每次初始化、越界或离散事件后，PDMPStepper 会按该策略重新区分
    # continuous/discrete species/channels。
    # 如果只想手工固定 continuous channels，可改成 FixedPDMPPartitionStrategy。
    partition_strategy = ScalingPDMPPartitionStrategy(
        ScalingPDMPConfig(
            N0=PDMP_N0,
            continuous_copy_number_scale_threshold_mu=PDMP_CONTINUOUS_COPY_NUMBER_SCALE_THRESHOLD_MU,
            adaptation_scale_threshold_eta=PDMP_ADAPTATION_SCALE_THRESHOLD_ETA,
            reaction_relaxation_delta=PDMP_REACTION_RELAXATION_DELTA,
            use_lp=PDMP_USE_LP,
            enable_fast_subnetworks=PDMP_ENABLE_FAST_SUBNETWORKS,
            fast_subnetwork_threshold=PDMP_FAST_SUBNETWORK_THRESHOLD,
            fast_subnetwork_max_size=PDMP_FAST_SUBNETWORK_MAX_SIZE,
        )
    )
    stepper = PDMPStepper(
        partition_strategy=partition_strategy,
        config=PDMPConfig(
            ode_step=PDMP_ODE_STEP,
            adaptive=True,
            repartition_on_event=PDMP_REPARTITION_ON_EVENT,
            repartition_on_bounds=PDMP_REPARTITION_ON_BOUNDS,
        ),
    )

    print("Cross-catalysis reaction system")
    print(f"alphabet={ALPHABET}, max_len={MAX_LEN}")
    print(f"base n_species={network.n_species}, base n_channels={network.n_channels}")
    print(f"elementary n_species={elementary.n_species}, elementary n_channels={elementary.n_channels}")
    print(f"catalysis_mode={network.catalysis_mode}, saturation_alpha={network.saturation_alpha}")
    print(f"background_rate={BACKGROUND_RATE}, catalytic_gamma={CATALYTIC_GAMMA}")
    print(
        f"elementary catalysis: k1=2*gamma, "
        f"k2={ELEMENTARY_CATALYTIC_TURNOVER_RATE_K2}, "
        f"Ksat={ELEMENTARY_COMPLEX_KSAT}, "
        f"k_minus1={ELEMENTARY_CATALYST_UNBINDING_RATE}"
    )
    print(
        f"initial_food_count={INITIAL_FOOD_COUNT}, "
        f"k_food_zero_order_inflow={K_FOOD_ZERO_ORDER_INFLOW}, "
        f"food_inflow_hill={FOOD_INFLOW_HILL_COEFFICIENT}, "
        f"food_max_count={FOOD_MAX_COUNT}, "
        f"standard_zero_order_inflow={USE_STANDARD_ZERO_ORDER_INFLOW}"
    )
    print(
        f"algorithm3 mu={PDMP_CONTINUOUS_COPY_NUMBER_SCALE_THRESHOLD_MU}, "
        f"eta={PDMP_ADAPTATION_SCALE_THRESHOLD_ETA}, "
        f"delta={PDMP_REACTION_RELAXATION_DELTA}"
    )
    print(f"catalyst species={catalyst_species_names(network)}")
    print(f"catalyzed channels={catalyzed_channel_count(network)}")

    trajectory_recorder = TrajectoryRecorder()
    recorder = trajectory_recorder
    if fast_network_report:
        recorder = FastNetworkReportRecorder(
            trajectory_recorder,
            network=elementary,
            partition_strategy=partition_strategy,
            output_path=FAST_NETWORK_REPORT_PATH,
            interval_events=FAST_NETWORK_REPORT_INTERVAL_EVENTS,
            finite_config=FiniteMarkovConfig(
                max_states=FAST_NETWORK_REPORT_MAX_STATES,
                max_total_internal_count=FAST_NETWORK_REPORT_MAX_TOTAL_INTERNAL_COUNT,
            ),
        )

    result = ExperimentRunner().run_one(
        elementary,
        stepper,
        t_end=T_END,
        seed=SEED,
        recorder=recorder,
        # restriction=restriction,
        max_steps=MAX_STEPS,
        max_runtime_seconds=MAX_TIMES,
        timing_report=True,
        timing_report_dir="timing_reports",
        network_build_elapsed_seconds=build_elapsed,
    )

    trajectory_record = recorder.finalize()
    trajectory_record.run_metadata["example_parameters"] = example_parameters()
    trajectory_record.run_metadata["catalysis_assignment"] = json_ready(catalysis_result)
    trajectory_record.run_metadata["catalyst_species_names"] = catalyst_species_names(network)
    trajectory_record.run_metadata["base_network"] = {
        "n_species": int(network.n_species),
        "n_channels": int(network.n_channels),
    }
    trajectory_record.run_metadata["elementary_network"] = {
        "n_species": int(elementary.n_species),
        "n_channels": int(elementary.n_channels),
        "polymer_species_count": int(elementary.polymer_species_count),
        "n_complex_species": int(elementary.n_species - elementary.polymer_species_count),
    }
    save_trajectory_record(OUTPUT_PATH, trajectory_record)

    print_run_summary(result, trajectory_record)
    print(f"trajectory saved to: {OUTPUT_PATH}")
    if fast_network_report:
        print(f"fast network report saved to: {FAST_NETWORK_REPORT_PATH}")


if __name__ == "__main__":
    main()
