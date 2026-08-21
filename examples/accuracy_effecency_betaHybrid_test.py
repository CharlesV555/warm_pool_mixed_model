from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim import (  # noqa: E402
    BlendedHybridConfig,
    BlendedHybridStepper,
    ChannelBlock,
    ExperimentRunner,
    ReactionNetworkData,
    SSAStepper,
    StepResult,
    TrajectoryRecorder,
    build_reaction_rule_tables,
    clear_all_catalysis,
    generate_fixed_species_space,
    save_trajectory_record,
    set_catalytic_strengths_for_channels,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_NAME = "accuracy_effecency_betaHybrid_test"

# Network sweep.  Each network has monomers A and B.  The pure longest chain
# A...A catalyzes addition of B; B...B catalyzes addition of A.
NETWORK_MAX_LENGTHS = (8, 10, 12)
MONOMERS = ("A", "B")
INITIAL_MONOMER_COUNT = 10.0

# Base reaction coefficients are written explicitly for later scale edits.
K_POLY_LEFT = 0.1
K_POLY_RIGHT = 0.1
K_FRAG_LEFT = 0.11
K_FRAG_RIGHT = 0.11
K_NONFOOD_OUTFLOW = 1.5
CATALYSIS_MODE = "substrate_saturating"
SATURATION_ALPHA = 0.01
CATALYTIC_GAMMA = 10.0

# Stop conditions for every single run.  "Memory exploded" is handled by
# catching MemoryError and preserving the per-run summary written so far.
# SSA has no simulation-clock limit; BASE_T_END is only the blended fallback
# if all SSA runs fail to produce a finite final simulation time.
BASE_T_END = 10_000.0
MAX_STEPS = 100_000_000
MAX_RUNTIME_SECONDS = 3_600.0

# Run counts.  SSA is run first for each network without a t_end limit.  The
# median SSA final simulation time then becomes the t_end for all blended runs.
SSA_RUNS_PER_NETWORK = 100
BLENDED_RUNS_PER_PARAMETER = 30

# Parallelism.  A value of None uses all visible CPUs, capped by task count.
PARALLEL_WORKERS: int | None = None

# BlendedHybridConfig parameter scan.  i1/i2 form a triangular grid:
# i1 is sampled in [20, 100]; i2 is sampled in [40, 120]; keep i2 >= i1 + 20.
I1_LINEAR_SPACE = np.linspace(20.0, 100.0, num=5)
I2_LINEAR_SPACE = np.linspace(40.0, 120.0, num=5)
I2_MIN_GAP = 20.0

# Fixed blended parameters displayed here and copied into metadata.
BLENDED_DT_CLE = 0.01
BLENDED_DT_MACRO = 0.1
BLENDED_BETA_SPECIES_MODE = "reactants"
BLENDED_BETA_COMPUTE_MODE = "beta_compute_by_state_difference"
BLENDED_LOCAL_PROPENSITY_CALCULATION = True

# Full BlendedHybridConfig parameter list for reference:
# - i1, i2: molecule-number thresholds controlling beta.
# - dt_cle: CLE step size used inside mixed/continuous advancement.
# - dt_macro: macro step size for pure SSA/pure CLE branch selection.
# - beta_tol: tolerance used around beta branch checks.
# - round_mode: how low-count observed state is rounded.
# - clip_negative: whether CLE negative trial values are clipped.
# - beta_species_mode: species used to compute channel beta.
# - beta_compute_mode: full beta compute vs state-difference local update.
# - round_low_counts_after_cle: round low-count changed species after CLE.
# - strict_int_for_CLE: compute CLE propensity on rounded observed state.
# - local_propensity_calculation: use cached/local propensity updates.
# - cle_sparsity_sampling, cle_sparsity_sample_interval, cle_sparsity_plot_path:
#   optional diagnostics for CLE amounts/S sparsity.
# - use_reaction_interval_dt, reaction_interval_update_steps,
#   reaction_interval_scale: reaction-interval-derived dt control.
# - adaptive_cle_dt, cle_dt_min, cle_dt_max, cle_dt_shrink_factor,
#   cle_dt_growth_factor, cle_dt_max_retries: adaptive CLE retry controls.


@dataclass(frozen=True, slots=True)
class NetworkCase:
    name: str
    max_len: int


@dataclass(frozen=True, slots=True)
class BlendedParameterCase:
    index: int
    i1: float
    i2: float
    dt_cle: float
    dt_macro: float
    beta_species_mode: str
    beta_compute_mode: str
    local_propensity_calculation: bool

    @property
    def label(self) -> str:
        return (
            f"blended_p{int(self.index):03d}"
            f"_i1_{_float_label(self.i1)}"
            f"_i2_{_float_label(self.i2)}"
        )


class InfiniteHorizonSSAStepper(SSAStepper):
    """SSA stepper variant for calibration runs with no simulation t_end.

    The base SSAStepper advances by dt when no channel is active.  With
    dt=inf, that would turn an inactive finite-time run into final_time=inf.
    Here inactivity returns zero advancement so ExperimentRunner stops with
    no_progress at the last finite simulation time.
    """

    def step(self, state, dt: float, context) -> StepResult:
        if np.isfinite(float(dt)):
            return super().step(state, dt, context)

        network = context.network
        propensities = self._get_propensities(network, state)
        total = float(np.sum(propensities))
        if total <= 0.0:
            state.step_count += 1
            return StepResult(advanced_time=0.0, event_occurred=False, propensity_sum=0.0)

        rng = context.rng
        tau = float(rng.exponential(1.0 / total))
        threshold = float(rng.random() * total)
        cumulative = np.cumsum(propensities)
        chosen = int(np.searchsorted(cumulative, threshold))
        if chosen >= int(network.n_channels):
            chosen = int(network.n_channels - 1)

        changed_species = network.get_channel_changed_species(chosen)
        network.apply_channel_update(state, chosen)
        state.t += tau
        state.step_count += 1
        state.event_count += 1
        self._update_cached_propensities(network, state, changed_species)
        return StepResult(
            advanced_time=tau,
            event_occurred=True,
            channel_id=chosen,
            propensity_sum=total,
            tau=tau,
        )


def run() -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = SCRIPT_DIR / f"{RUN_NAME}_{timestamp}_out"
    output_root.mkdir(parents=True, exist_ok=True)

    seed_rng = np.random.default_rng()
    network_cases = [
        NetworkCase(name=f"ab_len{max_len}_pure_longest_cross_catalysis", max_len=int(max_len))
        for max_len in NETWORK_MAX_LENGTHS
    ]
    blended_cases = build_blended_parameter_cases()

    all_records: list[dict[str, Any]] = []
    for network_case in network_cases:
        network_dir = output_root / network_case.name
        network_dir.mkdir(parents=True, exist_ok=True)
        try:
            network, catalysis_metadata = build_network_case(network_case)
        except Exception as exc:
            record = network_error_record(network_case, exc)
            all_records.append(record)
            _write_json(network_dir / "network_build_error.json", record)
            _write_json(output_root / "run_records.json", all_records)
            print_single_record(record)
            continue
        _write_json(
            network_dir / "network_metadata.json",
            {
                "network_case": asdict(network_case),
                "n_species": int(network.n_species),
                "n_channels": int(network.n_channels),
                "catalysis": catalysis_metadata,
                "base_rates": base_rate_metadata(),
            },
        )

        ssa_dir = network_dir / "ssa"
        ssa_records = run_ssa_batch(
            network_case=network_case,
            output_dir=ssa_dir,
            seed_rng=seed_rng,
        )
        all_records.extend(ssa_records)
        _write_json(output_root / "run_records.json", all_records)
        blended_t_end = median_final_time(ssa_records, fallback=BASE_T_END)
        print(
            f"[{RUN_NAME}] network={network_case.name} "
            f"ssa_median_final_time={blended_t_end:.6g}; blended t_end updated"
        )

        for parameter_case in blended_cases:
            blended_dir = network_dir / parameter_case.label
            blended_records = run_blended_batch(
                network_case=network_case,
                parameter_case=parameter_case,
                t_end=blended_t_end,
                output_dir=blended_dir,
                seed_rng=seed_rng,
            )
            all_records.extend(blended_records)
            _write_json(output_root / "run_records.json", all_records)

    payload = {
        "run_name": RUN_NAME,
        "output_root": str(output_root),
        "seed_source": "np.random.default_rng() without fixed seed",
        "stop_conditions": stop_condition_metadata(),
        "parallel": parallel_metadata(),
        "network_cases": [asdict(case) for case in network_cases],
        "blended_parameter_cases": [asdict(case) | {"label": case.label} for case in blended_cases],
        "n_records": len(all_records),
        "error_summary": error_summary(all_records),
        "records": all_records,
    }
    _write_json(output_root / "run_metadata.json", payload)
    print_error_summary(payload["error_summary"])
    print(f"[{RUN_NAME}] wrote metadata: {output_root / 'run_metadata.json'}")
    return payload


def build_blended_parameter_cases() -> list[BlendedParameterCase]:
    cases: list[BlendedParameterCase] = []
    for i1 in I1_LINEAR_SPACE:
        i1_value = float(i1)
        for i2 in I2_LINEAR_SPACE:
            i2_value = float(i2)
            if i2_value < i1_value + float(I2_MIN_GAP):
                continue
            cases.append(
                BlendedParameterCase(
                    index=len(cases),
                    i1=i1_value,
                    i2=i2_value,
                    dt_cle=float(BLENDED_DT_CLE),
                    dt_macro=float(BLENDED_DT_MACRO),
                    beta_species_mode=str(BLENDED_BETA_SPECIES_MODE),
                    beta_compute_mode=str(BLENDED_BETA_COMPUTE_MODE),
                    local_propensity_calculation=bool(BLENDED_LOCAL_PROPENSITY_CALCULATION),
                )
            )
    if not cases:
        raise ValueError("no blended parameter cases; require i2 >= i1 + I2_MIN_GAP")
    return cases


def build_network_case(case: NetworkCase) -> tuple[ReactionNetworkData, dict[str, Any]]:
    initial_counts = {monomer: float(INITIAL_MONOMER_COUNT) for monomer in MONOMERS}
    space = generate_fixed_species_space(MONOMERS, max_len=int(case.max_len), initial_counts=initial_counts)
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=float(K_POLY_LEFT),
        k_poly_right=float(K_POLY_RIGHT),
        k_frag_left=float(K_FRAG_LEFT),
        k_frag_right=float(K_FRAG_RIGHT),
        k_outflow=float(K_NONFOOD_OUTFLOW),
        outflow_species_ids=[
            sid
            for sid, name in enumerate(space.species_names)
            if name not in MONOMERS
        ],
        catalysis_mode=str(CATALYSIS_MODE),
        saturation_alpha=float(SATURATION_ALPHA),
    )
    catalysis = assign_longest_pure_cross_catalysis(network, case.max_len)
    return network, catalysis


def assign_longest_pure_cross_catalysis(network: ReactionNetworkData, max_len: int) -> dict[str, Any]:
    clear_all_catalysis(network, rebuild=False)
    rules = ((str("A" * int(max_len)), "B"), (str("B" * int(max_len)), "A"))
    primary_channels_by_catalyst: dict[str, list[int]] = {}
    channels_by_catalyst: dict[str, list[int]] = {}
    gamma_by_primary_channel: dict[int, float] = {}

    for catalyst_name, added_monomer_name in rules:
        catalyst_sid = network.species_idx(catalyst_name)
        added_monomer_sid = network.species_idx(added_monomer_name)
        primary_channels = terminal_matched_addition_channels(network, added_monomer_sid, added_monomer_name)
        strengths = np.full(primary_channels.shape[0], float(CATALYTIC_GAMMA), dtype=float)
        set_catalytic_strengths_for_channels(
            network,
            primary_channels,
            int(catalyst_sid),
            strengths,
            mirror_reverse=True,
            rebuild=False,
        )
        for channel_id in primary_channels:
            gamma_by_primary_channel[int(channel_id)] = float(CATALYTIC_GAMMA)
        mirrored = [
            int(reverse_id)
            for channel_id in primary_channels
            for reverse_id in network.get_reverse_channel_ids(int(channel_id))
        ]
        primary_channels_by_catalyst[catalyst_name] = [int(value) for value in primary_channels.tolist()]
        channels_by_catalyst[catalyst_name] = sorted(set(primary_channels_by_catalyst[catalyst_name] + mirrored))

    network.rebuild_dependency_indices()
    return {
        "method": "longest_pure_cross_terminal_matched_addition",
        "rules": {catalyst: added_monomer for catalyst, added_monomer in rules},
        "gamma": float(CATALYTIC_GAMMA),
        "mirror_reverse": True,
        "primary_channels_by_catalyst": primary_channels_by_catalyst,
        "channels_by_catalyst": channels_by_catalyst,
        "gamma_by_primary_channel": gamma_by_primary_channel,
    }


def terminal_matched_addition_channels(
    network: ReactionNetworkData,
    added_monomer_sid: int,
    added_monomer_name: str,
) -> np.ndarray:
    channels: list[int] = []
    for local_id, monomer_sid in enumerate(network.left_add_monomer):
        if int(monomer_sid) != int(added_monomer_sid):
            continue
        polymer_sid = int(network.left_add_species[int(local_id)])
        if network.species_names[polymer_sid].startswith(str(added_monomer_name)):
            channels.append(network.channel_id(ChannelBlock.LEFT_ADD, int(local_id)))

    for local_id, monomer_sid in enumerate(network.right_add_monomer):
        if int(monomer_sid) != int(added_monomer_sid):
            continue
        polymer_sid = int(network.right_add_species[int(local_id)])
        if network.species_names[polymer_sid].endswith(str(added_monomer_name)):
            channels.append(network.channel_id(ChannelBlock.RIGHT_ADD, int(local_id)))

    return np.asarray(channels, dtype=np.int64)


def run_ssa_batch(
    *,
    network_case: NetworkCase,
    output_dir: Path,
    seed_rng: np.random.Generator,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for run_index in range(int(SSA_RUNS_PER_NETWORK)):
        tasks.append(
            {
                "network_case": network_case,
                "method": "ssa",
                "run_index": int(run_index),
                "seed": next_seed(seed_rng),
                "t_end": None,
                "output_dir": output_dir,
                "parameter_case": None,
            }
        )
    records = run_tasks_parallel(tasks, output_dir)
    return records


def run_blended_batch(
    *,
    network_case: NetworkCase,
    parameter_case: BlendedParameterCase,
    t_end: float,
    output_dir: Path,
    seed_rng: np.random.Generator,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for run_index in range(int(BLENDED_RUNS_PER_PARAMETER)):
        tasks.append(
            {
                "network_case": network_case,
                "method": "blended",
                "run_index": int(run_index),
                "seed": next_seed(seed_rng),
                "t_end": float(t_end),
                "output_dir": output_dir,
                "parameter_case": parameter_case,
            }
        )
    records = run_tasks_parallel(tasks, output_dir)
    return records


def run_tasks_parallel(tasks: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    if not tasks:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = resolve_worker_count(len(tasks))
    print(f"[{RUN_NAME}] submitting {len(tasks)} tasks with workers={worker_count} output={output_dir}")
    records: list[dict[str, Any]] = []
    if worker_count <= 1:
        for task in tasks:
            record = run_single_task(task)
            records.append(record)
            _write_json(output_dir / "records.json", sorted_records(records))
            print_single_record(record)
        return sorted_records(records)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {executor.submit(run_single_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                record = future.result()
            except Exception as exc:
                record = task_error_record(task, exc)
            records.append(record)
            _write_json(output_dir / "records.json", sorted_records(records))
            print_single_record(record)
    return sorted_records(records)


def run_single_task(task: dict[str, Any]) -> dict[str, Any]:
    network_case = task["network_case"]
    if not isinstance(network_case, NetworkCase):
        network_case = NetworkCase(**dict(network_case))
    parameter_case = task.get("parameter_case")
    if parameter_case is not None and not isinstance(parameter_case, BlendedParameterCase):
        parameter_case = BlendedParameterCase(**dict(parameter_case))
    network, _ = build_network_case(network_case)
    return run_single(
        network=network,
        network_case=network_case,
        method=str(task["method"]),
        run_index=int(task["run_index"]),
        seed=int(task["seed"]),
        t_end=runner_t_end(task.get("t_end")),
        output_dir=Path(task["output_dir"]),
        parameter_case=parameter_case,
    )


def run_single(
    *,
    network: ReactionNetworkData,
    network_case: NetworkCase,
    method: str,
    run_index: int,
    seed: int,
    t_end: float,
    output_dir: Path,
    parameter_case: BlendedParameterCase | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = TrajectoryRecorder()
    started_at = perf_counter()
    stop_reason = "unknown"
    error = ""
    traceback_path = ""
    try:
        stepper = make_stepper(method, parameter_case)
        result = ExperimentRunner().run_one(
            network,
            stepper,
            t_end=float(t_end),
            seed=int(seed),
            recorder=recorder,
            max_steps=int(MAX_STEPS),
            max_runtime_seconds=float(MAX_RUNTIME_SECONDS),
        )
        summary = result.summary
        stop_reason = str(summary.metadata.get("stop_reason", "unknown"))
    except MemoryError as exc:
        summary = None
        stop_reason = "memory_error"
        error = repr(exc)
    except Exception as exc:
        summary = None
        stop_reason = "exception"
        error = repr(exc)
        trace_path = output_dir / f"{method}_run_{int(run_index):04d}_traceback.txt"
        trace_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        traceback_path = str(trace_path)
    wall_runtime = perf_counter() - started_at

    trajectory_path = output_dir / f"run_{int(run_index):04d}_seed_{int(seed)}.npz"
    if summary is not None:
        final_state = np.asarray(summary.final_state, dtype=float)
        record = {
            "status": "ok",
            "network": network_case.name,
            "max_len": int(network_case.max_len),
            "method": method,
            "run_index": int(run_index),
            "seed": int(seed),
            "requested_t_end": json_float_or_none(t_end),
            "simulation_final_time": float(summary.final_time),
            "wall_runtime_seconds": float(wall_runtime),
            "n_steps": int(summary.n_steps),
            "n_events": int(summary.n_events),
            "ssa_steps": int(summary.metadata.get("ssa_steps", 0)),
            "con_steps": int(summary.metadata.get("con_steps", 0)),
            "stop_reason": stop_reason,
            "trajectory_path": str(trajectory_path),
            "final_total_abundance": float(final_state.sum()),
            "max_species_count": float(final_state.max()) if final_state.size else 0.0,
            "parameter_case": parameter_payload(parameter_case),
            "error": "",
            "traceback_path": "",
        }
        try:
            trajectory_record = recorder.finalize()
            trajectory_record.run_metadata.update(
                {
                    "network": network_case.name,
                    "method": method,
                    "run_index": int(run_index),
                    "seed": int(seed),
                    "requested_t_end": json_float_or_none(t_end),
                    "max_steps": int(MAX_STEPS),
                    "max_runtime_seconds": float(MAX_RUNTIME_SECONDS),
                    "wall_runtime_seconds": float(wall_runtime),
                    "stop_reason": stop_reason,
                    "parameter_case": parameter_payload(parameter_case),
                }
            )
            save_trajectory_record(trajectory_path, trajectory_record)
        except Exception as exc:
            trace_path = output_dir / f"{method}_run_{int(run_index):04d}_save_traceback.txt"
            trace_path.write_text(traceback.format_exc(), encoding="utf-8")
            record["status"] = "error"
            record["stop_reason"] = "trajectory_save_exception"
            record["trajectory_path"] = ""
            record["error"] = repr(exc)
            record["traceback_path"] = str(trace_path)
    else:
        record = {
            "status": "error",
            "network": network_case.name,
            "max_len": int(network_case.max_len),
            "method": method,
            "run_index": int(run_index),
            "seed": int(seed),
            "requested_t_end": json_float_or_none(t_end),
            "simulation_final_time": None,
            "wall_runtime_seconds": float(wall_runtime),
            "n_steps": None,
            "n_events": None,
            "ssa_steps": None,
            "con_steps": None,
            "stop_reason": stop_reason,
            "trajectory_path": "",
            "parameter_case": parameter_payload(parameter_case),
            "error": error,
            "traceback_path": traceback_path,
        }
    _write_json(output_dir / f"{method}_run_{int(run_index):04d}_record.json", record)
    return record


def make_stepper(method: str, parameter_case: BlendedParameterCase | None):
    if method == "ssa":
        return InfiniteHorizonSSAStepper()
    if method == "blended":
        if parameter_case is None:
            raise ValueError("parameter_case is required for blended runs")
        return BlendedHybridStepper(
            BlendedHybridConfig(
                i1=float(parameter_case.i1),
                i2=float(parameter_case.i2),
                dt_cle=float(parameter_case.dt_cle),
                dt_macro=float(parameter_case.dt_macro),
                beta_species_mode=str(parameter_case.beta_species_mode),
                beta_compute_mode=str(parameter_case.beta_compute_mode),
                local_propensity_calculation=bool(parameter_case.local_propensity_calculation),
                use_reaction_interval_dt=False,
            )
        )
    raise ValueError(f"unknown method {method!r}")


def median_final_time(records: list[dict[str, Any]], *, fallback: float) -> float:
    values = [
        float(record["simulation_final_time"])
        for record in records
        if record.get("simulation_final_time") is not None and np.isfinite(float(record["simulation_final_time"]))
    ]
    if not values:
        return float(fallback)
    median = float(np.median(np.asarray(values, dtype=float)))
    return max(median, 0.0)


def runner_t_end(value: Any) -> float:
    # None means "no simulation-clock limit"; max_steps/max_runtime_seconds
    # remain active and determine when the SSA calibration run stops.
    return float("inf") if value is None else float(value)


def json_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return float(result) if np.isfinite(result) else None


def network_error_record(network_case: NetworkCase, exc: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "network": network_case.name,
        "max_len": int(network_case.max_len),
        "method": "network_build",
        "run_index": None,
        "seed": None,
        "requested_t_end": None,
        "simulation_final_time": None,
        "wall_runtime_seconds": None,
        "n_steps": None,
        "n_events": None,
        "ssa_steps": None,
        "con_steps": None,
        "stop_reason": "network_build_exception",
        "trajectory_path": "",
        "parameter_case": None,
        "error": repr(exc),
    }


def task_error_record(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    network_case = task["network_case"]
    if not isinstance(network_case, NetworkCase):
        network_case = NetworkCase(**dict(network_case))
    parameter_case = task.get("parameter_case")
    return {
        "status": "error",
        "network": network_case.name,
        "max_len": int(network_case.max_len),
        "method": str(task.get("method", "unknown")),
        "run_index": int(task.get("run_index", -1)),
        "seed": int(task.get("seed", 0)),
        "requested_t_end": json_float_or_none(task.get("t_end")),
        "simulation_final_time": None,
        "wall_runtime_seconds": None,
        "n_steps": None,
        "n_events": None,
        "ssa_steps": None,
        "con_steps": None,
        "stop_reason": "worker_exception",
        "trajectory_path": "",
        "parameter_case": parameter_payload(parameter_case),
        "error": repr(exc),
        "traceback_path": "",
    }


def sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record.get("network", "")),
            str(record.get("method", "")),
            _parameter_sort_key(record.get("parameter_case")),
            int(record.get("run_index") if record.get("run_index") is not None else -1),
        ),
    )


def _parameter_sort_key(value: Any) -> int:
    if isinstance(value, dict) and value.get("index") is not None:
        return int(value["index"])
    return -1


def parameter_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BlendedParameterCase):
        return asdict(value) | {"label": value.label}
    data = dict(value)
    label = data.get("label")
    if label is None:
        try:
            label = BlendedParameterCase(**data).label
        except Exception:
            label = ""
    data["label"] = str(label)
    return data


def error_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [record for record in records if record.get("status") != "ok"]
    reasons: dict[str, int] = {}
    messages: dict[str, int] = {}
    for record in errors:
        reason = str(record.get("stop_reason", "unknown"))
        message = str(record.get("error", ""))
        reasons[reason] = reasons.get(reason, 0) + 1
        if message:
            messages[message] = messages.get(message, 0) + 1
    return {
        "had_errors": bool(errors),
        "n_errors": len(errors),
        "reasons": reasons,
        "messages": messages,
    }


def print_error_summary(summary: dict[str, Any]) -> None:
    if not bool(summary.get("had_errors", False)):
        print(f"[{RUN_NAME}] error_summary: no errors")
        return
    print(
        f"[{RUN_NAME}] error_summary: n_errors={summary.get('n_errors')} "
        f"reasons={summary.get('reasons')} messages={summary.get('messages')}"
    )


def resolve_worker_count(task_count: int) -> int:
    if int(task_count) <= 0:
        return 0
    if PARALLEL_WORKERS is None:
        requested = os.cpu_count() or 1
    else:
        requested = int(PARALLEL_WORKERS)
    return max(1, min(int(requested), int(task_count)))


def parallel_metadata() -> dict[str, Any]:
    return {
        "backend": "process",
        "parallel_workers": None if PARALLEL_WORKERS is None else int(PARALLEL_WORKERS),
        "resolved_cpu_count": os.cpu_count(),
        "stage_order": "per network: parallel SSA -> median t_end -> parallel blended per parameter case",
        "network_rebuild_per_worker": True,
    }


def next_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(1, np.iinfo(np.uint32).max, dtype=np.uint32))


def stop_condition_metadata() -> dict[str, Any]:
    return {
        "memory_error": "caught MemoryError and stopped the single run",
        "ssa_t_end": None,
        "ssa_t_end_note": "SSA calibration runs have no simulation-clock limit",
        "blended_t_end": "median finite SSA simulation_final_time per network",
        "blended_fallback_t_end": float(BASE_T_END),
        "max_steps": int(MAX_STEPS),
        "max_runtime_seconds": float(MAX_RUNTIME_SECONDS),
    }


def base_rate_metadata() -> dict[str, Any]:
    return {
        "k_poly_left": float(K_POLY_LEFT),
        "k_poly_right": float(K_POLY_RIGHT),
        "k_frag_left": float(K_FRAG_LEFT),
        "k_frag_right": float(K_FRAG_RIGHT),
        "k_nonfood_outflow": float(K_NONFOOD_OUTFLOW),
        "catalysis_mode": str(CATALYSIS_MODE),
        "saturation_alpha": float(SATURATION_ALPHA),
        "catalytic_gamma": float(CATALYTIC_GAMMA),
    }


def print_single_record(record: dict[str, Any]) -> None:
    print(
        f"[{RUN_NAME}] status={record.get('status')} network={record.get('network')} "
        f"method={record.get('method')} run={record.get('run_index')} "
        f"sim_time={record.get('simulation_final_time')} steps={record.get('n_steps')} "
        f"events={record.get('n_events')} ssa_steps={record.get('ssa_steps')} "
        f"con_steps={record.get('con_steps')} stop={record.get('stop_reason')} "
        f"wall={record.get('wall_runtime_seconds')} trajectory={record.get('trajectory_path')}"
    )


def _float_label(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p").replace("+", "")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def apply_environment_overrides() -> None:
    # Optional smoke-test/runtime overrides.  Defaults above remain authoritative
    # when these environment variables are absent.
    global NETWORK_MAX_LENGTHS
    global SSA_RUNS_PER_NETWORK
    global BLENDED_RUNS_PER_PARAMETER
    global BASE_T_END
    global MAX_STEPS
    global MAX_RUNTIME_SECONDS
    global PARALLEL_WORKERS
    global I1_LINEAR_SPACE
    global I2_LINEAR_SPACE

    NETWORK_MAX_LENGTHS = _env_int_tuple("BETA_TEST_NETWORK_MAX_LENGTHS", NETWORK_MAX_LENGTHS)
    SSA_RUNS_PER_NETWORK = _env_int("BETA_TEST_SSA_RUNS", SSA_RUNS_PER_NETWORK)
    BLENDED_RUNS_PER_PARAMETER = _env_int("BETA_TEST_BLENDED_RUNS", BLENDED_RUNS_PER_PARAMETER)
    BASE_T_END = _env_float("BETA_TEST_BASE_T_END", BASE_T_END)
    MAX_STEPS = _env_int("BETA_TEST_MAX_STEPS", MAX_STEPS)
    MAX_RUNTIME_SECONDS = _env_float("BETA_TEST_MAX_RUNTIME_SECONDS", MAX_RUNTIME_SECONDS)
    PARALLEL_WORKERS = _env_optional_int("BETA_TEST_WORKERS", PARALLEL_WORKERS)
    I1_LINEAR_SPACE = np.asarray(_env_float_tuple("BETA_TEST_I1_VALUES", tuple(I1_LINEAR_SPACE)), dtype=float)
    I2_LINEAR_SPACE = np.asarray(_env_float_tuple("BETA_TEST_I2_VALUES", tuple(I2_LINEAR_SPACE)), dtype=float)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default) if value is None or not value.strip() else int(value)


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    if value.strip().lower() in {"none", "null"}:
        return None
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default) if value is None or not value.strip() else float(value)


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(int(item) for item in default)
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(float(item) for item in default)
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    apply_environment_overrides()
    run()


if __name__ == "__main__":
    main()
