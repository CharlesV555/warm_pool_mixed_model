from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (
    ElementaryMassActionNetwork,
    ExperimentRunner,
    FixedPDMPPartitionStrategy,
    PDMPConfig,
    PDMPStepper,
    SSAStepper,
    TrajectoryRecorder,
    format_stepper_info,
    save_trajectory_record,
)

# Switch this to "ssa" or "pdmp". A command-line argument overrides it:
#   python examples/fast_dimerization.py ssa
#   python examples/fast_dimerization.py pdmp
SIMULATION_METHOD = "ssa"# "pdmp"

T_END = 400.0
SEED = 123
MAX_STEPS = 100_000_000
MAX_TIMES = 60.0

SPECIES_NAMES = ["S0", "S1", "S2"]
INITIAL_COUNTS = np.array([540.0, 730.0, 0.0], dtype=float)

K_DIMERIZATION = 1.0
K_DIMER_DISSOCIATION = 200.0
K_MONOMER_DECAY = 0.02
K_DIMER_CONVERSION = 0.004

CHANNEL_DIMERIZATION = 0
CHANNEL_DIMER_DISSOCIATION = 1
CHANNEL_MONOMER_DECAY = 2
CHANNEL_DIMER_CONVERSION = 3
PDMP_CONTINUOUS_CHANNELS = (CHANNEL_DIMERIZATION, CHANNEL_DIMER_DISSOCIATION)
# The fast reversible dimerization pair is stiff under the current explicit
# Euler PDMP integrator; 0.01 can overshoot to negative counts near x0=540.
PDMP_ODE_STEP = 0.001


def build_fast_dimerization_network() -> ElementaryMassActionNetwork:
    # Reactions:
    #   2 S0 -> S1
    #   S1 -> 2 S0
    #   S0 -> empty
    #   S1 -> S2
    nu_minus = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    nu_plus = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rate_constants = np.array(
        [
            K_DIMERIZATION,
            K_DIMER_DISSOCIATION,
            K_MONOMER_DECAY,
            K_DIMER_CONVERSION,
        ],
        dtype=float,
    )
    reaction_labels = [
        {"name": "dimerization", "reaction": "2 S0 -> S1"},
        {"name": "dimer_dissociation", "reaction": "S1 -> 2 S0"},
        {"name": "monomer_decay", "reaction": "S0 -> empty"},
        {"name": "dimer_conversion", "reaction": "S1 -> S2"},
    ]
    return ElementaryMassActionNetwork(
        species_names=list(SPECIES_NAMES),
        name_to_idx={name: sid for sid, name in enumerate(SPECIES_NAMES)},
        x0=np.array(INITIAL_COUNTS, dtype=float, copy=True),
        nu_minus=nu_minus,
        nu_plus=nu_plus,
        rate_constants=rate_constants,
        reaction_labels=reaction_labels,
    )


def build_stepper(method: str):
    method_key = method.lower()
    if method_key == "ssa":
        return SSAStepper(use_local_propensity_updates=False), None
    if method_key == "pdmp":
        return (
            PDMPStepper(
                partition_strategy=FixedPDMPPartitionStrategy(
                    continuous_channels=PDMP_CONTINUOUS_CHANNELS,
                ),
                config=PDMPConfig(
                    ode_step=PDMP_ODE_STEP,
                    adaptive=False,
                    repartition_on_event=False,
                    repartition_on_bounds=False,
                ),
            ),
            None,
        )
    raise ValueError("method must be 'ssa' or 'pdmp'")


def output_path_for_method(method: str) -> Path:
    return EXAMPLES_DIR / f"fast_dimerization_{method.lower()}_trajectory.npz"


def example_parameters(method: str) -> dict:
    return {
        "example": "fast_dimerization",
        "method": method.lower(),
        "t_end": T_END,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "max_times": MAX_TIMES,
        "species_names": list(SPECIES_NAMES),
        "initial_counts": INITIAL_COUNTS.tolist(),
        "k_dimerization": K_DIMERIZATION,
        "k_dimer_dissociation": K_DIMER_DISSOCIATION,
        "k_monomer_decay": K_MONOMER_DECAY,
        "k_dimer_conversion": K_DIMER_CONVERSION,
        "pdmp_continuous_channels": list(PDMP_CONTINUOUS_CHANNELS),
        "pdmp_ode_step": PDMP_ODE_STEP,
    }


def print_run_summary(method: str, run_result, trajectory_record) -> None:
    print(f"\n{method.upper()} summary:")
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


def main(method: str | None = None) -> None:
    selected_method = (method or SIMULATION_METHOD).lower()
    build_started_at = perf_counter()
    network = build_fast_dimerization_network()
    build_elapsed = perf_counter() - build_started_at
    stepper, runner_dt = build_stepper(selected_method)

    print("Fast Dimerization network")
    print(f"method={selected_method}")
    print(f"n_species={network.n_species}, n_channels={network.n_channels}")
    print(f"initial_counts={INITIAL_COUNTS.tolist()}")
    print(
        f"rates: dimerization={K_DIMERIZATION}, "
        f"dissociation={K_DIMER_DISSOCIATION}, "
        f"decay={K_MONOMER_DECAY}, "
        f"conversion={K_DIMER_CONVERSION}"
    )
    if selected_method == "pdmp":
        print(f"pdmp_continuous_channels={list(PDMP_CONTINUOUS_CHANNELS)}, ode_step={PDMP_ODE_STEP}")

    recorder = TrajectoryRecorder()
    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=T_END,
        seed=SEED,
        dt=runner_dt,
        recorder=recorder,
        max_steps=MAX_STEPS,
        max_runtime_seconds=MAX_TIMES,
        timing_report=True,
        timing_report_dir="timing_reports",
        timing_report_name=f"fast_dimerization_{selected_method}",
        network_build_elapsed_seconds=build_elapsed,
    )

    trajectory_record = recorder.finalize()
    trajectory_record.run_metadata["example_parameters"] = example_parameters(selected_method)
    trajectory_record.run_metadata["network"] = {
        "n_species": int(network.n_species),
        "n_channels": int(network.n_channels),
        "reaction_labels": [network.describe_channel(channel_id) for channel_id in range(network.n_channels)],
    }
    output_path = output_path_for_method(selected_method)
    save_trajectory_record(output_path, trajectory_record)

    print_run_summary(selected_method, result, trajectory_record)
    print(f"trajectory saved to: {output_path}")


if __name__ == "__main__":
    cli_method = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_method)
