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

import catalyst_run

from polymer_sim import (
    BlendedHybridConfig,
    BlendedHybridStepper,
    ExperimentRunner,
    TrajectoryRecorder,
    save_trajectory_record,
)


# Keep the same reaction network, catalysis assignment, rates, outflow, and
# formal INFLOW/food-cap restriction as catalyst_run.py, but use a short
# horizon for this smoke test.
T_END = 2.0
SEED = catalyst_run.SEED
MAX_STEPS = catalyst_run.MAX_STEPS
MAX_TIMES = 300.0
BLENDED_I1 = 10.0
BLENDED_I2 = 30.0
BLENDED_DT_CLE = 0.0001
BLENDED_DT_MACRO = 0.01
OUTPUT_PATH = EXAMPLES_DIR / "blended_hybrid_minimal_trajectory.npz"


def main() -> None:
    network, catalysis_result = catalyst_run.build_random_catalyst_network()
    restriction = catalyst_run.build_food_upper_limit_restriction(network)
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
        timing_report=True,
        timing_report_dir="timing_reports",
        network_build_elapsed_seconds=build_elapsed,
        # timing_report_interval_events=1000,
        # timing_report_sim_interval=0.01,
    )
    record = recorder.finalize()
    record.run_metadata["example_parameters"] = catalyst_run.example_parameters()
    record.run_metadata["catalysis_assignment"] = catalyst_run.json_ready(catalysis_result)
    record.run_metadata["catalyst_species_names"] = catalyst_run.catalyst_species_names(network)
    save_trajectory_record(OUTPUT_PATH, record)

    print("Blended hybrid minimal run")
    print(f"network: n_species={network.n_species}, n_channels={network.n_channels}")
    print(f"catalysis_mode={network.catalysis_mode}, saturation_alpha={network.saturation_alpha}")
    print(f"catalyst_assignment_mode={catalyst_run.CATALYST_ASSIGNMENT_MODE}")
    print(f"final time: {result.summary.final_time:.4f}")
    print(f"n_steps: {result.summary.n_steps}")
    print(f"n_events: {result.summary.n_events}")
    print(f"stop_reason: {result.summary.metadata.get('stop_reason')}")
    print(f"final total abundance: {float(result.summary.final_state.sum()):.4f}")
    print(f"max species count: {float(result.summary.final_state.max()):.4f}")
    print(f"trajectory saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
