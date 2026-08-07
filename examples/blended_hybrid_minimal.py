from __future__ import annotations
from time import perf_counter
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

from examples.MM_catalysis import cross_catalysis_NRM as network_example

from polymer_sim import (
    BlendedHybridConfig,
    BlendedHybridStepper,
    ExperimentRunner,
    TrajectoryRecorder,
    build_food_supply_restriction,
    format_stepper_info,
    normalize_food_supply_mode,
    save_trajectory_record,
    SSAStepper,
)

 
# Minimal blended-hybrid entry.
# - To change the reaction network, replace build_network() below.
# - To change the simulation method, replace build_stepper() below.
# - To change food handling, edit FOOD_SUPPLY_MODE:
#   "explicit_inflow" keeps formal food INFLOW reaction channels;
#   "constant" disables those channels during network construction and lets the
#   runner apply a chemostat-style food projection after each accepted step.
T_END = 200.0
SEED = network_example.SEED
MAX_STEPS = network_example.MAX_STEPS
MAX_TIMES = 60.0
BLENDED_I1 = 50.0
BLENDED_I2 = 80.0
BLENDED_DT_CLE = 0.000389
BLENDED_DT_MACRO = 0.01
OUTPUT_PATH = EXAMPLES_DIR / "blended_hybrid_minimal_trajectory.npz"
FOOD_SUPPLY_MODE = "constant"


def build_network():
    """Build the current polymer network.

    Network-specific rates, catalysis rules, and max length live in
    examples/MM_catalysis/cross_catalysis_NRM.py.  Passing FOOD_SUPPLY_MODE here
    is important: constant food should not also create formal INFLOW channels.
    """

    return network_example.build_cross_catalysis_network(food_supply_mode=FOOD_SUPPLY_MODE)


def build_stepper() -> BlendedHybridStepper:
    """Build the simulation method used by this example.

    Swap this function to SSAStepper, OptimizedNRMStepper, PDMPStepper, etc. if
    you want the same network to run under a different numerical strategy.
    """

    return BlendedHybridStepper(
        BlendedHybridConfig(
            i1=BLENDED_I1,
            i2=BLENDED_I2,
            dt_cle=BLENDED_DT_CLE,
            dt_macro=BLENDED_DT_MACRO,
            use_reaction_interval_dt=False,
            reaction_interval_update_steps=1,
            strict_int_for_CLE=True,
        )
    )


def build_food_restriction(network):
    """Build the optional food boundary condition consumed by ExperimentRunner."""

    return build_food_supply_restriction(
        network,
        mode=FOOD_SUPPLY_MODE,
        food_species=network_example.ALPHABET,
        food_count=network_example.INITIAL_FOOD_COUNT,
    )


def main() -> None:
    build_started = perf_counter()
    network, catalysis_result = build_network()
    restriction = build_food_restriction(network)
    build_elapsed = perf_counter() - build_started
    stepper = build_stepper()
    # stepper = SSAStepper()
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
        timing_report=True,
        timing_report_dir="timing_reports",
        network_build_elapsed_seconds=build_elapsed,
        # timing_report_interval_events=1000,
        # timing_report_sim_interval=0.01,
    )
    record = recorder.finalize()
    record.run_metadata["example_parameters"] = network_example.example_parameters()
    record.run_metadata["food_supply_mode"] = normalize_food_supply_mode(FOOD_SUPPLY_MODE)
    record.run_metadata["catalysis_assignment"] = network_example.json_ready(catalysis_result)
    record.run_metadata["catalyst_species_names"] = network_example.catalyst_species_names(network)
    save_trajectory_record(OUTPUT_PATH, record)

    print("Blended hybrid minimal run")
    print(f"network: n_species={network.n_species}, n_channels={network.n_channels}")
    print(f"catalysis_mode={network.catalysis_mode}, saturation_alpha={network.saturation_alpha}")
    print(f"food_supply_mode={normalize_food_supply_mode(FOOD_SUPPLY_MODE)}")
    print(f"food_restriction={restriction is not None}")
    print(f"cross_catalysis_rules={network_example.CROSS_CATALYSIS_RULES}")
    print(f"final time: {result.summary.final_time:.4f}")
    print(f"n_steps: {result.summary.n_steps}")
    print(f"n_events: {result.summary.n_events}")
    print(f"stop_reason: {result.summary.metadata.get('stop_reason')}")
    print(format_stepper_info(result.summary.metadata))
    print(f"final total abundance: {float(result.summary.final_state.sum()):.4f}")
    print(f"max species count: {float(result.summary.final_state.max()):.4f}")
    print(f"trajectory saved to: {OUTPUT_PATH}")
    


if __name__ == "__main__":
    main()
