from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent

import accuracy_effecency_betaHybrid_test as base  # noqa: E402


RUN_NAME = "accuracy_effecency_betaHybrid_tend_gradient"

# =========================
# Explicit experiment setup
# =========================

# Network cases. These reuse accuracy_effecency_betaHybrid_test.py builders:
# AB polymer networks where A...A catalyzes B addition and B...B catalyzes A addition.
NETWORK_MAX_LENGTHS = (8, 10, 12)

# True: ignore T_END_GRID and use max_runtime_seconds/max_steps to probe how far
# each algorithm advances in simulation time and what final total abundance it reaches.
# False: run the t_end gradient gate test below.
MAX_RUNTIME_PROBE_MODE = True

# Simulation-clock horizons. Runs are gated in ascending order:
# if a network-algorithm group fails to reach a t_end, longer t_end values are skipped.
T_END_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)

# Number of repeated trajectories per network-algorithm-t_end group.
# The same seed list is reused across all t_end values within one network-algorithm group.
RUNS_PER_GROUP = 30

# Per-run stop limits. These are wall-clock/step safety limits, not the t_end grid.
MAX_STEPS = base.MAX_STEPS
MAX_RUNTIME_SECONDS = base.MAX_RUNTIME_SECONDS

# Parallelism. None means all visible CPUs, capped by task count.
PARALLEL_WORKERS: int | None = None

# Algorithm cases. The "method" value must be understood by base.make_stepper:
# - "ssa" uses SSAStepper through base.InfiniteHorizonSSAStepper.
# - "blended" uses BlendedHybridStepper with the parameter_case below.
# Blended cases are always executed before SSA cases.
ALGORITHM_CASES: tuple[dict[str, Any], ...] = (
    {
        "label": "blended_local_beta_local_propensity",
        "method": "blended",
        "parameter_case": {
            "index": 0,
            "i1": 20.0,
            "i2": 40.0,
            "dt_cle": base.BLENDED_DT_CLE,
            "dt_macro": base.BLENDED_DT_MACRO,
            "beta_species_mode": base.BLENDED_BETA_SPECIES_MODE,
            "beta_compute_mode": base.BLENDED_BETA_COMPUTE_MODE,
            "local_propensity_calculation": base.BLENDED_LOCAL_PROPENSITY_CALCULATION,
        },
    },
    {
        "label": "ssa",
        "method": "ssa",
        "parameter_case": None,
    },
)


def run() -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = SCRIPT_DIR / f"{RUN_NAME}_{timestamp}_out"
    output_root.mkdir(parents=True, exist_ok=True)

    seed_rng = np.random.default_rng()
    network_cases = [
        base.NetworkCase(name=f"ab_len{max_len}_pure_longest_cross_catalysis", max_len=int(max_len))
        for max_len in NETWORK_MAX_LENGTHS
    ]
    t_end_values = sorted(float(value) for value in T_END_GRID)
    algorithm_cases = sorted(
        [normalize_algorithm_case(item) for item in ALGORITHM_CASES],
        key=lambda item: 0 if item["method"] == "blended" else 1,
    )

    if MAX_RUNTIME_PROBE_MODE:
        return run_max_runtime_probe(
            output_root=output_root,
            seed_rng=seed_rng,
            network_cases=network_cases,
            algorithm_cases=algorithm_cases,
        )

    all_records: list[dict[str, Any]] = []
    gate_reports: list[dict[str, Any]] = []

    for network_case in network_cases:
        network_dir = output_root / network_case.name
        network_dir.mkdir(parents=True, exist_ok=True)
        try:
            network, catalysis_metadata = base.build_network_case(network_case)
        except Exception as exc:
            record = base.network_error_record(network_case, exc)
            all_records.append(record)
            write_json(network_dir / "network_build_error.json", record)
            write_json(output_root / "run_records.json", base.sorted_records(all_records))
            base.print_single_record(record)
            continue

        write_json(
            network_dir / "network_metadata.json",
            {
                "network_case": asdict(network_case),
                "n_species": int(network.n_species),
                "n_channels": int(network.n_channels),
                "catalysis": catalysis_metadata,
                "base_rates": base.base_rate_metadata(),
            },
        )

        for algorithm_case in algorithm_cases:
            group_label = str(algorithm_case["label"])
            group_dir = network_dir / group_label
            group_dir.mkdir(parents=True, exist_ok=True)
            seeds = [base.next_seed(seed_rng) for _ in range(int(RUNS_PER_GROUP))]
            write_json(
                group_dir / "seed_plan.json",
                {
                    "network": network_case.name,
                    "algorithm_label": group_label,
                    "method": algorithm_case["method"],
                    "runs_per_group": int(RUNS_PER_GROUP),
                    "seeds": seeds,
                    "same_seeds_across_t_end": True,
                },
            )

            group_failed = False
            for t_end in t_end_values:
                if group_failed:
                    skipped = {
                        "network": network_case.name,
                        "algorithm_label": group_label,
                        "method": algorithm_case["method"],
                        "t_end": float(t_end),
                        "status": "skipped",
                        "reason": "previous_t_end_not_reached",
                        "n_runs": int(RUNS_PER_GROUP),
                    }
                    gate_reports.append(skipped)
                    print_gate_report(skipped)
                    continue

                t_dir = group_dir / f"t_end_{float_label(t_end)}"
                tasks = build_tasks(
                    network_case=network_case,
                    algorithm_case=algorithm_case,
                    t_end=float(t_end),
                    output_dir=t_dir,
                    seeds=seeds,
                )
                records = run_tasks_parallel(tasks, t_dir)
                annotate_records(records, algorithm_case=algorithm_case, requested_t_end=float(t_end))
                all_records.extend(records)
                write_json(output_root / "run_records.json", sorted_records(all_records))

                report = t_end_gate_report(
                    records,
                    network=network_case.name,
                    algorithm_label=group_label,
                    method=str(algorithm_case["method"]),
                    t_end=float(t_end),
                )
                gate_reports.append(report)
                write_json(t_dir / "t_end_gate_report.json", report)
                write_json(output_root / "t_end_gate_reports.json", gate_reports)
                print_gate_report(report)
                if not bool(report["all_reached_t_end"]):
                    group_failed = True

    payload = {
        "run_name": RUN_NAME,
        "output_root": str(output_root),
        "seed_source": "np.random.default_rng() without fixed seed",
        "settings": settings_metadata(),
        "network_cases": [asdict(case) for case in network_cases],
        "algorithm_cases": json_ready_algorithm_cases(algorithm_cases),
        "t_end_grid": [float(value) for value in t_end_values],
        "n_records": len(all_records),
        "gate_summary": gate_summary(gate_reports),
        "error_summary": base.error_summary(all_records),
        "batch_report": base.batch_report(all_records),
        "gate_reports": gate_reports,
        "records": all_records,
    }
    write_json(output_root / "batch_report.json", payload["batch_report"])
    write_json(output_root / "t_end_gate_summary.json", payload["gate_summary"])
    write_json(output_root / "run_metadata.json", payload)
    base.print_batch_report(payload["batch_report"])
    base.print_error_summary(payload["error_summary"])
    print_gate_summary(payload["gate_summary"])
    print(f"[{RUN_NAME}] wrote metadata: {output_root / 'run_metadata.json'}")
    return payload


def run_max_runtime_probe(
    *,
    output_root: Path,
    seed_rng: np.random.Generator,
    network_cases: list[base.NetworkCase],
    algorithm_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    all_records: list[dict[str, Any]] = []
    probe_reports: list[dict[str, Any]] = []

    for network_case in network_cases:
        network_dir = output_root / network_case.name
        network_dir.mkdir(parents=True, exist_ok=True)
        try:
            network, catalysis_metadata = base.build_network_case(network_case)
        except Exception as exc:
            record = base.network_error_record(network_case, exc)
            all_records.append(record)
            write_json(network_dir / "network_build_error.json", record)
            write_json(output_root / "run_records.json", sorted_records(all_records))
            base.print_single_record(record)
            continue

        write_json(
            network_dir / "network_metadata.json",
            {
                "network_case": asdict(network_case),
                "n_species": int(network.n_species),
                "n_channels": int(network.n_channels),
                "catalysis": catalysis_metadata,
                "base_rates": base.base_rate_metadata(),
            },
        )

        for algorithm_case in algorithm_cases:
            group_label = str(algorithm_case["label"])
            group_dir = network_dir / group_label / "max_runtime_probe"
            seeds = [base.next_seed(seed_rng) for _ in range(int(RUNS_PER_GROUP))]
            write_json(
                group_dir / "seed_plan.json",
                {
                    "network": network_case.name,
                    "algorithm_label": group_label,
                    "method": algorithm_case["method"],
                    "runs_per_group": int(RUNS_PER_GROUP),
                    "seeds": seeds,
                    "mode": "max_runtime_probe",
                    "requested_t_end": None,
                    "runner_t_end": "inf",
                    "max_runtime_seconds": float(MAX_RUNTIME_SECONDS),
                },
            )
            tasks = build_tasks(
                network_case=network_case,
                algorithm_case=algorithm_case,
                t_end=float("inf"),
                output_dir=group_dir,
                seeds=seeds,
            )
            records = run_tasks_parallel(tasks, group_dir)
            annotate_records(records, algorithm_case=algorithm_case, requested_t_end=None)
            for record in records:
                record["max_runtime_probe"] = True
            all_records.extend(records)
            write_json(output_root / "run_records.json", sorted_records(all_records))

            report = max_runtime_probe_report(
                records,
                network=network_case.name,
                algorithm_label=group_label,
                method=str(algorithm_case["method"]),
            )
            probe_reports.append(report)
            write_json(group_dir / "max_runtime_probe_report.json", report)
            write_json(output_root / "max_runtime_probe_reports.json", probe_reports)
            print_probe_report(report)

    payload = {
        "run_name": RUN_NAME,
        "output_root": str(output_root),
        "mode": "max_runtime_probe",
        "seed_source": "np.random.default_rng() without fixed seed",
        "settings": settings_metadata(),
        "network_cases": [asdict(case) for case in network_cases],
        "algorithm_cases": json_ready_algorithm_cases(algorithm_cases),
        "n_records": len(all_records),
        "probe_summary": max_runtime_probe_summary(probe_reports),
        "error_summary": base.error_summary(all_records),
        "batch_report": base.batch_report(all_records),
        "probe_reports": probe_reports,
        "records": all_records,
    }
    write_json(output_root / "batch_report.json", payload["batch_report"])
    write_json(output_root / "max_runtime_probe_summary.json", payload["probe_summary"])
    write_json(output_root / "run_metadata.json", payload)
    base.print_batch_report(payload["batch_report"])
    base.print_error_summary(payload["error_summary"])
    print_probe_summary(payload["probe_summary"])
    print(f"[{RUN_NAME}] wrote metadata: {output_root / 'run_metadata.json'}")
    return payload


def build_tasks(
    *,
    network_case: base.NetworkCase,
    algorithm_case: dict[str, Any],
    t_end: float,
    output_dir: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    tasks = []
    parameter_case = algorithm_case.get("parameter_case")
    for run_index, seed in enumerate(seeds):
        tasks.append(
            {
                "network_case": network_case,
                "method": str(algorithm_case["method"]),
                "algorithm_label": str(algorithm_case["label"]),
                "run_index": int(run_index),
                "seed": int(seed),
                "t_end": float(t_end),
                "output_dir": output_dir,
                "parameter_case": parameter_case,
            }
        )
    return tasks


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
            write_json(output_dir / "records.json", sorted_records(records))
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
            write_json(output_dir / "records.json", sorted_records(records))
            print_single_record(record)
    return sorted_records(records)


def run_single_task(task: dict[str, Any]) -> dict[str, Any]:
    network_case = task["network_case"]
    if not isinstance(network_case, base.NetworkCase):
        network_case = base.NetworkCase(**dict(network_case))
    parameter_case = task.get("parameter_case")
    if parameter_case is not None and not isinstance(parameter_case, base.BlendedParameterCase):
        parameter_case = base.BlendedParameterCase(**dict(parameter_case))
    network, _ = base.build_network_case(network_case)
    return run_single(
        network=network,
        network_case=network_case,
        algorithm_label=str(task["algorithm_label"]),
        method=str(task["method"]),
        run_index=int(task["run_index"]),
        seed=int(task["seed"]),
        t_end=float(task["t_end"]),
        output_dir=Path(task["output_dir"]),
        parameter_case=parameter_case,
    )


def run_single(
    *,
    network: base.ReactionNetworkData,
    network_case: base.NetworkCase,
    algorithm_label: str,
    method: str,
    run_index: int,
    seed: int,
    t_end: float,
    output_dir: Path,
    parameter_case: base.BlendedParameterCase | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = base.TrajectoryRecorder()
    started_at = perf_counter()
    stop_reason = "unknown"
    error = ""
    traceback_path = ""
    try:
        stepper = base.make_stepper(method, parameter_case)
        result = base.ExperimentRunner().run_one(
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
        trace_path = output_dir / f"{algorithm_label}_run_{int(run_index):04d}_traceback.txt"
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        traceback_path = str(trace_path)
    wall_runtime = perf_counter() - started_at

    trajectory_path = output_dir / f"run_{int(run_index):04d}_seed_{int(seed)}.npz"
    if summary is not None:
        final_state = np.asarray(summary.final_state, dtype=float)
        record = {
            "status": "ok",
            "network": network_case.name,
            "max_len": int(network_case.max_len),
            "algorithm_label": str(algorithm_label),
            "method": method,
            "run_index": int(run_index),
            "seed": int(seed),
            "requested_t_end": base.json_float_or_none(t_end),
            "simulation_final_time": float(summary.final_time),
            "wall_runtime_seconds": float(wall_runtime),
            "n_steps": int(summary.n_steps),
            "n_events": int(summary.n_events),
            "ssa_steps": int(summary.metadata.get("ssa_steps", 0)),
            "con_steps": int(summary.metadata.get("con_steps", 0)),
            "stop_reason": stop_reason,
            "reached_requested_t_end": reached_requested_t_end(summary.final_time, t_end, stop_reason),
            "trajectory_path": str(trajectory_path),
            "final_total_abundance": float(final_state.sum()),
            "max_species_count": float(final_state.max()) if final_state.size else 0.0,
            "parameter_case": base.parameter_payload(parameter_case),
            "error": "",
            "traceback_path": "",
        }
        try:
            trajectory_record = recorder.finalize()
            trajectory_record.run_metadata.update(
                {
                    "network": network_case.name,
                    "algorithm_label": str(algorithm_label),
                    "method": method,
                    "run_index": int(run_index),
                    "seed": int(seed),
                    "requested_t_end": base.json_float_or_none(t_end),
                    "max_steps": int(MAX_STEPS),
                    "max_runtime_seconds": float(MAX_RUNTIME_SECONDS),
                    "wall_runtime_seconds": float(wall_runtime),
                    "stop_reason": stop_reason,
                    "reached_requested_t_end": bool(record["reached_requested_t_end"]),
                    "parameter_case": base.parameter_payload(parameter_case),
                }
            )
            base.save_trajectory_record(trajectory_path, trajectory_record)
        except Exception as exc:
            trace_path = output_dir / f"{algorithm_label}_run_{int(run_index):04d}_save_traceback.txt"
            trace_path.write_text(traceback.format_exc(), encoding="utf-8")
            record["status"] = "error"
            record["stop_reason"] = "trajectory_save_exception"
            record["reached_requested_t_end"] = False
            record["trajectory_path"] = ""
            record["error"] = repr(exc)
            record["traceback_path"] = str(trace_path)
    else:
        record = {
            "status": "error",
            "network": network_case.name,
            "max_len": int(network_case.max_len),
            "algorithm_label": str(algorithm_label),
            "method": method,
            "run_index": int(run_index),
            "seed": int(seed),
            "requested_t_end": base.json_float_or_none(t_end),
            "simulation_final_time": None,
            "wall_runtime_seconds": float(wall_runtime),
            "n_steps": None,
            "n_events": None,
            "ssa_steps": None,
            "con_steps": None,
            "stop_reason": stop_reason,
            "reached_requested_t_end": False,
            "trajectory_path": "",
            "parameter_case": base.parameter_payload(parameter_case),
            "error": error,
            "traceback_path": traceback_path,
        }
        partial_path = output_dir / f"run_{int(run_index):04d}_seed_{int(seed)}_partial.npz"
        if base.try_save_partial_trajectory(
            recorder,
            partial_path,
            metadata={
                "network": network_case.name,
                "algorithm_label": str(algorithm_label),
                "method": method,
                "run_index": int(run_index),
                "seed": int(seed),
                "requested_t_end": base.json_float_or_none(t_end),
                "max_steps": int(MAX_STEPS),
                "max_runtime_seconds": float(MAX_RUNTIME_SECONDS),
                "wall_runtime_seconds": float(wall_runtime),
                "stop_reason": stop_reason,
                "reached_requested_t_end": False,
                "partial_after_error": True,
                "error": error,
                "parameter_case": base.parameter_payload(parameter_case),
            },
        ):
            record["trajectory_path"] = str(partial_path)
            record["partial_trajectory_saved"] = True
        else:
            record["partial_trajectory_saved"] = False
    write_json(output_dir / f"{algorithm_label}_run_{int(run_index):04d}_record.json", record)
    return record


def reached_requested_t_end(final_time: float, requested_t_end: float, stop_reason: str) -> bool:
    return (
        str(stop_reason) == "reached_t_end"
        and np.isfinite(float(final_time))
        and float(final_time) >= float(requested_t_end) - 1e-12
    )


def t_end_gate_report(
    records: list[dict[str, Any]],
    *,
    network: str,
    algorithm_label: str,
    method: str,
    t_end: float,
) -> dict[str, Any]:
    failures = [record for record in records if not bool(record.get("reached_requested_t_end", False))]
    return {
        "network": str(network),
        "algorithm_label": str(algorithm_label),
        "method": str(method),
        "t_end": float(t_end),
        "status": "ok" if not failures else "failed",
        "all_reached_t_end": not failures,
        "n_runs": len(records),
        "n_reached_t_end": len(records) - len(failures),
        "n_failed_to_reach_t_end": len(failures),
        "failure_stop_reasons": base.count_values(record.get("stop_reason") for record in failures),
        "failure_errors": base.count_values(
            record.get("error") for record in failures if str(record.get("error", "")).strip()
        ),
        "failed_run_indices": [int(record.get("run_index", -1)) for record in failures],
    }


def gate_summary(gate_reports: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [item for item in gate_reports if item.get("status") != "skipped"]
    failed = [item for item in evaluated if not bool(item.get("all_reached_t_end", False))]
    skipped = [item for item in gate_reports if item.get("status") == "skipped"]
    return {
        "n_groups_total_including_skipped": len(gate_reports),
        "n_groups_evaluated": len(evaluated),
        "n_groups_all_reached_t_end": len(evaluated) - len(failed),
        "n_groups_failed_to_reach_t_end": len(failed),
        "n_groups_skipped_after_failure": len(skipped),
        "failed_groups": failed,
        "skipped_groups": skipped,
        "failure_stop_reasons": base.count_values(
            reason
            for item in failed
            for reason, count in dict(item.get("failure_stop_reasons", {})).items()
            for _ in range(int(count))
        ),
    }


def max_runtime_probe_report(
    records: list[dict[str, Any]],
    *,
    network: str,
    algorithm_label: str,
    method: str,
) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    error_records = [record for record in records if record.get("status") != "ok"]
    return {
        "network": str(network),
        "algorithm_label": str(algorithm_label),
        "method": str(method),
        "mode": "max_runtime_probe",
        "n_runs": len(records),
        "n_ok": len(ok_records),
        "n_error": len(error_records),
        "max_runtime_seconds": float(MAX_RUNTIME_SECONDS),
        "max_steps": int(MAX_STEPS),
        "stop_reasons": base.count_values(record.get("stop_reason") for record in records),
        "simulation_final_time_quantiles": base.finite_quantiles(
            record.get("simulation_final_time") for record in ok_records
        ),
        "final_total_abundance_quantiles": base.finite_quantiles(
            record.get("final_total_abundance") for record in ok_records
        ),
        "max_species_count_quantiles": base.finite_quantiles(
            record.get("max_species_count") for record in ok_records
        ),
        "wall_runtime_seconds_quantiles": base.finite_quantiles(
            record.get("wall_runtime_seconds") for record in ok_records
        ),
        "n_steps_quantiles": base.finite_quantiles(record.get("n_steps") for record in ok_records),
        "n_events_quantiles": base.finite_quantiles(record.get("n_events") for record in ok_records),
        "error_messages": base.count_values(
            record.get("error") for record in error_records if str(record.get("error", "")).strip()
        ),
    }


def max_runtime_probe_summary(probe_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_groups": len(probe_reports),
        "n_runs_total": int(sum(int(item.get("n_runs", 0)) for item in probe_reports)),
        "n_ok_total": int(sum(int(item.get("n_ok", 0)) for item in probe_reports)),
        "n_error_total": int(sum(int(item.get("n_error", 0)) for item in probe_reports)),
        "stop_reasons": base.count_values(
            reason
            for item in probe_reports
            for reason, count in dict(item.get("stop_reasons", {})).items()
            for _ in range(int(count))
        ),
        "groups": probe_reports,
    }


def annotate_records(records: list[dict[str, Any]], *, algorithm_case: dict[str, Any], requested_t_end: float | None) -> None:
    for record in records:
        record["algorithm_label"] = str(algorithm_case["label"])
        record["requested_t_end"] = None if requested_t_end is None else float(requested_t_end)


def normalize_algorithm_case(item: dict[str, Any]) -> dict[str, Any]:
    data = dict(item)
    method = str(data["method"])
    parameter_case = data.get("parameter_case")
    if parameter_case is not None and not isinstance(parameter_case, base.BlendedParameterCase):
        parameter_case = base.BlendedParameterCase(**dict(parameter_case))
    data["method"] = method
    data["parameter_case"] = parameter_case
    data["label"] = str(data.get("label") or method)
    return data


def json_ready_algorithm_cases(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        result.append(
            {
                "label": str(item["label"]),
                "method": str(item["method"]),
                "parameter_case": base.parameter_payload(item.get("parameter_case")),
            }
        )
    return result


def task_error_record(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    network_case = task["network_case"]
    if not isinstance(network_case, base.NetworkCase):
        network_case = base.NetworkCase(**dict(network_case))
    parameter_case = task.get("parameter_case")
    return {
        "status": "error",
        "network": network_case.name,
        "max_len": int(network_case.max_len),
        "algorithm_label": str(task.get("algorithm_label", task.get("method", "unknown"))),
        "method": str(task.get("method", "unknown")),
        "run_index": int(task.get("run_index", -1)),
        "seed": int(task.get("seed", 0)),
        "requested_t_end": base.json_float_or_none(task.get("t_end")),
        "simulation_final_time": None,
        "wall_runtime_seconds": None,
        "n_steps": None,
        "n_events": None,
        "ssa_steps": None,
        "con_steps": None,
        "stop_reason": "worker_exception",
        "reached_requested_t_end": False,
        "trajectory_path": "",
        "parameter_case": base.parameter_payload(parameter_case),
        "error": repr(exc),
        "traceback_path": "",
    }


def sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record.get("network", "")),
            str(record.get("algorithm_label", "")),
            float(record.get("requested_t_end") or -1.0),
            int(record.get("run_index") if record.get("run_index") is not None else -1),
        ),
    )


def print_single_record(record: dict[str, Any]) -> None:
    print(
        f"[{RUN_NAME}] status={record.get('status')} network={record.get('network')} "
        f"algorithm={record.get('algorithm_label')} method={record.get('method')} "
        f"t_end={record.get('requested_t_end')} run={record.get('run_index')} "
        f"seed={record.get('seed')} sim_time={record.get('simulation_final_time')} "
        f"reached={record.get('reached_requested_t_end')} stop={record.get('stop_reason')} "
        f"wall={record.get('wall_runtime_seconds')} trajectory={record.get('trajectory_path')}"
    )


def print_gate_report(report: dict[str, Any]) -> None:
    if report.get("status") == "skipped":
        print(
            f"[{RUN_NAME}] gate skipped network={report.get('network')} "
            f"algorithm={report.get('algorithm_label')} t_end={report.get('t_end')} "
            f"reason={report.get('reason')}"
        )
        return
    print(
        f"[{RUN_NAME}] gate network={report.get('network')} "
        f"algorithm={report.get('algorithm_label')} t_end={report.get('t_end')} "
        f"reached={report.get('n_reached_t_end')}/{report.get('n_runs')} "
        f"failed={report.get('n_failed_to_reach_t_end')} "
        f"reasons={report.get('failure_stop_reasons')}"
    )


def print_gate_summary(summary: dict[str, Any]) -> None:
    print(
        f"[{RUN_NAME}] gate_summary evaluated={summary.get('n_groups_evaluated')} "
        f"all_reached={summary.get('n_groups_all_reached_t_end')} "
        f"failed={summary.get('n_groups_failed_to_reach_t_end')} "
        f"skipped={summary.get('n_groups_skipped_after_failure')} "
        f"failure_reasons={summary.get('failure_stop_reasons')}"
    )


def print_probe_report(report: dict[str, Any]) -> None:
    print(
        f"[{RUN_NAME}] probe network={report.get('network')} "
        f"algorithm={report.get('algorithm_label')} "
        f"ok={report.get('n_ok')}/{report.get('n_runs')} "
        f"stop={report.get('stop_reasons')} "
        f"sim_time={report.get('simulation_final_time_quantiles')} "
        f"total_mol={report.get('final_total_abundance_quantiles')}"
    )


def print_probe_summary(summary: dict[str, Any]) -> None:
    print(
        f"[{RUN_NAME}] probe_summary groups={summary.get('n_groups')} "
        f"ok={summary.get('n_ok_total')}/{summary.get('n_runs_total')} "
        f"errors={summary.get('n_error_total')} "
        f"stop_reasons={summary.get('stop_reasons')}"
    )


def resolve_worker_count(task_count: int) -> int:
    if int(task_count) <= 0:
        return 0
    requested = os.cpu_count() or 1 if PARALLEL_WORKERS is None else int(PARALLEL_WORKERS)
    return max(1, min(int(requested), int(task_count)))


def settings_metadata() -> dict[str, Any]:
    return {
        "max_runtime_probe_mode": bool(MAX_RUNTIME_PROBE_MODE),
        "network_max_lengths": [int(value) for value in NETWORK_MAX_LENGTHS],
        "t_end_grid": [float(value) for value in T_END_GRID],
        "runs_per_group": int(RUNS_PER_GROUP),
        "max_steps": int(MAX_STEPS),
        "max_runtime_seconds": float(MAX_RUNTIME_SECONDS),
        "parallel_workers": None if PARALLEL_WORKERS is None else int(PARALLEL_WORKERS),
        "same_seeds_across_t_end": True,
        "blended_runs_first": True,
        "skip_longer_t_end_after_first_failed_group": True,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def float_label(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p").replace("+", "")


def apply_environment_overrides() -> None:
    global MAX_RUNTIME_PROBE_MODE
    global NETWORK_MAX_LENGTHS
    global T_END_GRID
    global RUNS_PER_GROUP
    global MAX_STEPS
    global MAX_RUNTIME_SECONDS
    global PARALLEL_WORKERS

    base.apply_environment_overrides()
    MAX_RUNTIME_PROBE_MODE = env_bool("TEND_GRADIENT_MAX_RUNTIME_PROBE_MODE", MAX_RUNTIME_PROBE_MODE)
    NETWORK_MAX_LENGTHS = env_int_tuple("TEND_GRADIENT_NETWORK_MAX_LENGTHS", NETWORK_MAX_LENGTHS)
    T_END_GRID = env_float_tuple("TEND_GRADIENT_T_END_GRID", T_END_GRID)
    RUNS_PER_GROUP = env_int("TEND_GRADIENT_RUNS_PER_GROUP", RUNS_PER_GROUP)
    MAX_STEPS = env_int("TEND_GRADIENT_MAX_STEPS", base.MAX_STEPS)
    MAX_RUNTIME_SECONDS = env_float("TEND_GRADIENT_MAX_RUNTIME_SECONDS", base.MAX_RUNTIME_SECONDS)
    PARALLEL_WORKERS = env_optional_int("TEND_GRADIENT_PARALLEL_WORKERS", PARALLEL_WORKERS)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default) if value is None or not value.strip() else int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value")


def env_optional_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    return None if parsed <= 0 else parsed


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default) if value is None or not value.strip() else float(value)


def env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(int(item) for item in default)
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(float(item) for item in default)
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    apply_environment_overrides()
    run()


if __name__ == "__main__":
    main()
