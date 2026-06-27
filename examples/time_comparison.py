from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (
    ExperimentRunner,
    FoodUpperLimitRestriction,
    SSAStepper,
    TrajectoryRecorder,
    assign_paper_minimal_catalysis,
    build_n3_wh_network,
    build_n3_wh_reactions,
    compute_max_raf,
    enumerate_irr_rafs,
    save_trajectory_record,
)

T_END = 4.0
SEED = 123
MAX_STEPS = 100_000_000
MAX_TIMES = 30.0  # Set to None to disable runtime cutoff
K_RIGHT_ADD = 0.1
K_NONFOOD_OUTFLOW = 0.8
INITIAL_FOOD_COUNT = 100.0
FOOD_INFLOW_RATE = 5000.0
FOOD_MAX_COUNT = 100.0
INITIAL_COUNTS = {
    "0": min(INITIAL_FOOD_COUNT, FOOD_MAX_COUNT),
    "1": min(INITIAL_FOOD_COUNT, FOOD_MAX_COUNT),
}
CATALYSIS_MODE = "substrate_saturating"  # "linear" or "substrate_saturating"
SATURATION_ALPHA = 0.01


def build_catalysis_map(network, reactions):
    """Build a static channel -> catalyst set view for RAF analysis."""

    return {
        reaction.channel_id: set(int(sid) for sid in network.get_channel_catalysts(reaction.channel_id))
        for reaction in reactions
    }


def print_static_raf_result(max_raf, irr_rafs) -> None:
    print("maxRAF:")
    for reaction in max_raf:
        print(f"  {reaction.reaction_id} [{reaction.category}]")

    print("\nirrRAFs:")
    for idx, subset in enumerate(irr_rafs):
        labels = ", ".join(reaction.reaction_id for reaction in subset)
        print(f"  irrRAF {idx}: {labels}")


def build_food_upper_limit_restriction(network) -> FoodUpperLimitRestriction:
    return FoodUpperLimitRestriction(
        {
            network.species_idx("0"): FOOD_MAX_COUNT,
            network.species_idx("1"): FOOD_MAX_COUNT,
        }
    )


def print_run_summary(run_result, trajectory_record) -> None:
    last_event_time = float(trajectory_record.times[-2]) if trajectory_record.times.shape[0] >= 2 else 0.0
    stop_reason = run_result.summary.metadata.get("stop_reason", "unknown")
    stopped_by_max_steps = stop_reason == "max_steps"
    stopped_by_max_times = stop_reason == "max_runtime_seconds"
    absorbed_early = (
        stop_reason == "reached_t_end" and last_event_time < float(run_result.summary.final_time)
    )

    print("\nSSA summary:")
    print(
        "  "
        f"t={run_result.summary.final_time:.4f}, "
        f"steps={run_result.summary.n_steps}, "
        f"events={run_result.summary.n_events}, "
        f"seed={run_result.summary.metadata.get('seed')}, "
        f"stop_reason={stop_reason}"
    )
    print(
        "  "
        f"trajectory points={trajectory_record.times.shape[0]}, "
        f"state shape={trajectory_record.states.shape}"
    )
    print(f"  last event time={last_event_time:.4f}")
    if stopped_by_max_steps:
        print("  simulation stopped at max_steps before reaching t_end")
    if stopped_by_max_times:
        print("  simulation stopped at MAX_TIMES before reaching t_end")
    if absorbed_early:
        print(
            "  process became inactive before t_end; increasing t_end beyond the last event time "
            "will not change the final state"
        )


def main() -> None:
    # Network entry point:
    # If you want a different chemistry or a different fixed species space,
    # change the builder here first.
    network = build_n3_wh_network(
        initial_counts=INITIAL_COUNTS,
        k_right_add=K_RIGHT_ADD,
        k_nonfood_outflow=K_NONFOOD_OUTFLOW,
        k_food_inflow=FOOD_INFLOW_RATE,
        catalysis_mode=CATALYSIS_MODE,
        saturation_alpha=SATURATION_ALPHA,
    )

    # Static reaction view entry point:
    # If you want different reaction categories, paper-specific labels,
    # or a different restricted reaction subset, change build_n3_wh_reactions()
    # and classify_wh_reaction_category() in polymer_sim/model/wills_henderson.py.
    reactions = build_n3_wh_reactions(network)

    # Catalysis entry point:
    # If you want a different deterministic paper network or a different
    # catalyst assignment policy, change assign_paper_minimal_catalysis()
    # or add a new helper in polymer_sim/model/wills_henderson.py or
    # polymer_sim/model/catalysis.py.
    assign_paper_minimal_catalysis(network, strength=1.0)

    food = {network.species_idx("0"), network.species_idx("1")}
    catalysis_map = build_catalysis_map(network, reactions)
    max_raf = compute_max_raf(food, reactions, catalysis_map)
    irr_rafs = enumerate_irr_rafs(food, max_raf, catalysis_map)
    print_static_raf_result(max_raf, irr_rafs)

    # Stepper entry point:
    # If you want a different simulation method, replace SSAStepper() here
    # with another stepper implementation.
    stepper = SSAStepper()

    # Strategy entry point:
    # This example uses plain SSA and therefore does not pass partition or
    # blending strategies. For HybridStepper/CLEStepper work, the first place
    # to change is the ExperimentRunner.run_one(...) call below.
    #
    # Restriction entry point:
    # Food input is represented by formal INFLOW channels in the network.
    # The restriction below only caps food from above; it does not replenish
    # food to a fixed count.
    #
    # Runtime cutoff entry point:
    # If MAX_TIMES is not None, runner will stop when wall-clock runtime reaches
    # that limit, even if t_end has not been reached yet.
    t0 = perf_counter()
    build_elapsed = perf_counter() - t0
    
    runner = ExperimentRunner()
    recorder = TrajectoryRecorder()
    restriction = build_food_upper_limit_restriction(network)
    run_result = runner.run_one(
        network,
        stepper,
        t_end=T_END,
        seed=SEED,
        recorder=recorder,
        restriction=restriction,
        max_steps=MAX_STEPS,
        max_runtime_seconds=MAX_TIMES,
        timing_report=True,
        timing_report_dir="timing_reports",
        network_build_elapsed_seconds=build_elapsed,
    )

    # Recording entry point:
    # If you only want lightweight summaries, remove recorder=recorder above.
    # If you want a different on-disk record format, change the save call here
    # or extend polymer_sim/recording/.
    trajectory_record = recorder.finalize()
    output_path = PROJECT_ROOT / "examples" / "time_comparison_trajectory.npz"
    save_trajectory_record(output_path, trajectory_record)

    print_run_summary(run_result, trajectory_record)
    print(f"  trajectory saved to: {output_path}")
    print("  hs2014 model enabled: formal OUTFLOW/INFLOW channels + food upper-limit restriction")


if __name__ == "__main__":
    main()
