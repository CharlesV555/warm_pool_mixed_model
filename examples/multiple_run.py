from __future__ import annotations

import sys
from pathlib import Path

from polymer_sim.recording.distribution_comparison import compare_species_distributions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

from compute_strategy import ComputeStrategy
from multiple_run_core import MultipleRunConfig, run_config, run_methods


# Main entry configuration:
# - Single-method run: METHODS = "ssa" or ("ssa",)
# - Method comparison: METHODS = ("ssa", "blended")
METHODS = ("ssa", "blended")
N_RUNS = 100
BASE_SEED = 20260524
T_END = 0.2
MAX_STEPS = 10_000_000
MAX_RUNTIME_SECONDS = 60.0

OUTPUT_DIR = EXAMPLES_DIR / "method_run_outputs"
SAVE_TRAJECTORIES = True

# Food handling is defined in multiple_run_core.build_shared_objects(). It
# currently uses catalyst_run.py's formal INFLOW channels plus
# FoodUpperLimitRestriction, so all methods compare the same capped
# finite-reservoir model.

COMPUTE_STRATEGY = ComputeStrategy(
    backend="process",  # "process", "thread", or "serial"
    n_workers=None,  # None means auto-detect logical CPUs, then cap by task count.
    use_gpu=False,
    reserve_logical_cpus=0,
)

# Stepper options.  `stepper_dt` is required for CLE/hybrid; SSA and blended
# can leave it as None.
STEPPER_DT = None
CLE_FAST_CHANNEL_IDS = None
HYBRID_FAST_CHANNEL_IDS: tuple[int, ...] | str = ()

BLENDED_I1 = 110.0
BLENDED_I2 = 150.0
BLENDED_DT_CLE = 0.01
BLENDED_DT_MACRO = 0.0001
BLENDED_USE_REACTION_INTERVAL_DT = True
BLENDED_REACTION_INTERVAL_UPDATE_STEPS = 100


def build_config() -> MultipleRunConfig:
    """Build the single config object used by this script entry point."""

    return MultipleRunConfig(
        methods=METHODS,
        n_runs=N_RUNS,
        base_seed=BASE_SEED,
        t_end=T_END,
        max_steps=MAX_STEPS,
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        output_dir=OUTPUT_DIR,
        save_trajectories=SAVE_TRAJECTORIES,
        compute_strategy=COMPUTE_STRATEGY,
        stepper_dt=STEPPER_DT,
        cle_fast_channel_ids=CLE_FAST_CHANNEL_IDS,
        hybrid_fast_channel_ids=HYBRID_FAST_CHANNEL_IDS,
        blended_i1=BLENDED_I1,
        blended_i2=BLENDED_I2,
        blended_dt_cle=BLENDED_DT_CLE,
        blended_dt_macro=BLENDED_DT_MACRO,
        blended_use_reaction_interval_dt=BLENDED_USE_REACTION_INTERVAL_DT,
        blended_reaction_interval_update_steps=BLENDED_REACTION_INTERVAL_UPDATE_STEPS,
    )


def run() -> dict[str, object]:
    """Run the configured single-method or multi-method batch."""

    return run_config(build_config())


def run_paired_ssa_blended_test(
    *,
    n_pairs: int = N_RUNS,
    base_seed: int = BASE_SEED,
    t_end: float = T_END,
    max_steps: int = MAX_STEPS,
    max_runtime_seconds: float | None = MAX_RUNTIME_SECONDS,
    output_dir: Path | str = EXAMPLES_DIR / "paired_method_outputs",
    compute_strategy: ComputeStrategy | None = None,
) -> dict[str, object]:
    """Compatibility wrapper for the previous paired SSA/blended API."""

    return run_methods(
        methods=("ssa", "blended"),
        n_runs=n_pairs,
        base_seed=base_seed,
        t_end=t_end,
        max_steps=max_steps,
        max_runtime_seconds=max_runtime_seconds,
        output_dir=output_dir,
        save_trajectories=True,
        compute_strategy=compute_strategy or COMPUTE_STRATEGY,
        blended_i1=BLENDED_I1,
        blended_i2=BLENDED_I2,
        blended_dt_cle=BLENDED_DT_CLE,
        blended_dt_macro=BLENDED_DT_MACRO,
        blended_use_reaction_interval_dt=BLENDED_USE_REACTION_INTERVAL_DT,
        blended_reaction_interval_update_steps=BLENDED_REACTION_INTERVAL_UPDATE_STEPS,
    )


def main() -> None:
    run()
    from polymer_sim.recording.distribution_comparison import compare_species_distributions

    result = compare_species_distributions(
        "examples/method_run_outputs/paired_method_metadata.json",
        species=["1", "11","111","1111","11111","0", "00","000","0000","00000"],
        time_range=(0.0, 1.0),
        n_time_points=20,
        groups=["ssa", "blended"],
        output_dir="examples/distribution_comparison",
    )


if __name__ == "__main__":
    main()
