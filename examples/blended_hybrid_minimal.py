from __future__ import annotations

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
from polymer_sim.simulation.restriction import build_restriction


# Keep the same reaction network, catalysis assignment, rates, outflow, and
# restriction as catalyst_run.py, but use a short horizon for this smoke test.
T_END = min(catalyst_run.T_END, 0.01)
SEED = catalyst_run.SEED
MAX_STEPS = catalyst_run.MAX_STEPS
MAX_TIMES = min(catalyst_run.MAX_TIMES, 60.0)
BLENDED_I1 = 10.0
BLENDED_I2 = 30.0
BLENDED_DT_CLE = 0.01
BLENDED_DT_MACRO = 0.01
OUTPUT_PATH = EXAMPLES_DIR / "blended_hybrid_minimal_trajectory.npz"


def main() -> None:
    network, catalysis_result = catalyst_run.build_random_catalyst_network()
    restriction = build_restriction(
        network,
        food_species=catalyst_run.ALPHABET,
        food_count=catalyst_run.FOOD_COUNT,
    )
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
