from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

from compute_strategy import ComputeStrategy
from multiple_run import (
    BASE_SEED,
    MAX_RUNTIME_SECONDS,
    MAX_STEPS,
    NETWORK_SOURCE,
    run_methods,
)


def run() -> dict[str, object]:
    """Run a batch comparison between Direct SSA and optimized NRM.

    This file intentionally reuses examples/multiple_run.py's imported
    ``run_methods`` entry point and default network/seed constants.  It is the
    main place to run quick distribution and runtime comparisons after changing
    the NRM implementation.
    """

    return run_methods(
        methods=("ssa", "optimized_nrm"),
        n_runs=4,
        base_seed=BASE_SEED,
        t_end=0.2,
        max_steps=min(int(MAX_STEPS), 1_000_000),
        max_runtime_seconds=120.0 if MAX_RUNTIME_SECONDS is None else min(float(MAX_RUNTIME_SECONDS), 120.0),
        output_dir=EXAMPLES_DIR / "nrm_vs_ssa_outputs",
        network_source=NETWORK_SOURCE,
        save_trajectories=False,
        compute_strategy=ComputeStrategy(
            backend="serial",
            n_workers=1,
            use_gpu=False,
            reserve_logical_cpus=0,
        ),
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
