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


RUN_NAME = "accuracy_effecency_betaHybrid__test_beta"

# In this reverse-calibration test, blended runs have no simulation-clock
# t_end. They stop by max_steps/max_runtime_seconds/no_progress. For each
# network, the median finite final time is computed for each blended parameter
# case. The maximum of those medians is then used as the network-level SSA
# t_end. SSA has its own wall-time limit.
SSA_MAX_RUNTIME_SECONDS = 3_600.0


def run() -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = SCRIPT_DIR / f"{RUN_NAME}_{timestamp}_out"
    output_root.mkdir(parents=True, exist_ok=True)

    seed_rng = np.random.default_rng()
    network_cases = [
        base.NetworkCase(name=f"ab_len{max_len}_pure_longest_cross_catalysis", max_len=int(max_len))
        for max_len in base.NETWORK_MAX_LENGTHS
    ]
    blended_cases = base.build_blended_parameter_cases()

    all_records: list[dict[str, Any]] = []
    for network_case in network_cases:
        network_dir = output_root / network_case.name
        network_dir.mkdir(parents=True, exist_ok=True)
        try:
            network, catalysis_metadata = base.build_network_case(network_case)
        except Exception as exc:
            record = base.network_error_record(network_case, exc)
            all_records.append(record)
            write_json(network_dir / "network_build_error.json", record)
            write_json(output_root / "run_records.json", all_records)
            print_single_record(record)
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

        network_blended_medians: list[dict[str, Any]] = []
        for parameter_case in blended_cases:
            case_dir = network_dir / parameter_case.label

            blended_dir = case_dir / "blended"
            blended_records = run_blended_calibration_batch(
                network_case=network_case,
                parameter_case=parameter_case,
                output_dir=blended_dir,
                seed_rng=seed_rng,
            )
            annotate_records(
                blended_records,
                physical_network=network_case.name,
                calibration_label=parameter_case.label,
                calibration_role="blended_calibration",
            )
            all_records.extend(blended_records)
            write_json(output_root / "run_records.json", all_records)

            blended_median_t_end = base.median_final_time(blended_records, fallback=base.BASE_T_END)
            network_blended_medians.append(
                {
                    "parameter_label": parameter_case.label,
                    "median_final_time": float(blended_median_t_end),
                    "n_records": len(blended_records),
                }
            )
            print(
                f"[{RUN_NAME}] network={network_case.name} parameter={parameter_case.label} "
                f"blended_median_final_time={blended_median_t_end:.6g}"
            )

        ssa_t_end = max(
            (float(item["median_final_time"]) for item in network_blended_medians),
            default=float(base.BASE_T_END),
        )
        write_json(
            network_dir / "blended_t_end_calibration.json",
            {
                "network": network_case.name,
                "ssa_t_end_rule": "max median blended simulation_final_time across parameter cases",
                "ssa_t_end": float(ssa_t_end),
                "blended_parameter_medians": network_blended_medians,
            },
        )
        print(
            f"[{RUN_NAME}] network={network_case.name} "
            f"max_blended_median_final_time={ssa_t_end:.6g}; SSA t_end updated"
        )

        ssa_dir = network_dir / "ssa"
        ssa_records = run_ssa_to_t_end_batch(
            network_case=network_case,
            t_end=ssa_t_end,
            output_dir=ssa_dir,
            seed_rng=seed_rng,
        )
        annotate_records(
            ssa_records,
            physical_network=network_case.name,
            calibration_label="max_blended_median_t_end",
            calibration_role="ssa_to_network_max_blended_median_t_end",
        )
        all_records.extend(ssa_records)
        write_json(output_root / "run_records.json", all_records)

    payload = {
        "run_name": RUN_NAME,
        "output_root": str(output_root),
        "seed_source": "np.random.default_rng() without fixed seed",
        "stop_conditions": stop_condition_metadata(),
        "parallel": parallel_metadata(),
        "network_cases": [asdict(case) for case in network_cases],
        "blended_parameter_cases": [asdict(case) | {"label": case.label} for case in blended_cases],
        "n_records": len(all_records),
        "error_summary": base.error_summary(all_records),
        "records": all_records,
    }
    write_json(output_root / "run_metadata.json", payload)
    print_error_summary(payload["error_summary"])
    print(f"[{RUN_NAME}] wrote metadata: {output_root / 'run_metadata.json'}")
    return payload


def run_blended_calibration_batch(
    *,
    network_case: base.NetworkCase,
    parameter_case: base.BlendedParameterCase,
    output_dir: Path,
    seed_rng: np.random.Generator,
) -> list[dict[str, Any]]:
    tasks = []
    for run_index in range(int(base.BLENDED_RUNS_PER_PARAMETER)):
        tasks.append(
            {
                "network_case": network_case,
                "analysis_network": network_case.name,
                "physical_network": network_case.name,
                "method": "blended",
                "run_index": int(run_index),
                "seed": base.next_seed(seed_rng),
                "t_end": None,
                "output_dir": output_dir,
                "parameter_case": parameter_case,
                "max_steps": int(base.MAX_STEPS),
                "max_runtime_seconds": float(base.MAX_RUNTIME_SECONDS),
            }
        )
    return run_tasks_parallel(tasks, output_dir)


def run_ssa_to_t_end_batch(
    *,
    network_case: base.NetworkCase,
    t_end: float,
    output_dir: Path,
    seed_rng: np.random.Generator,
) -> list[dict[str, Any]]:
    tasks = []
    for run_index in range(int(base.SSA_RUNS_PER_NETWORK)):
        tasks.append(
            {
                "network_case": network_case,
                "analysis_network": network_case.name,
                "physical_network": network_case.name,
                "method": "ssa",
                "run_index": int(run_index),
                "seed": base.next_seed(seed_rng),
                "t_end": float(t_end),
                "output_dir": output_dir,
                "parameter_case": None,
                "max_steps": int(base.MAX_STEPS),
                "max_runtime_seconds": float(SSA_MAX_RUNTIME_SECONDS),
            }
        )
    return run_tasks_parallel(tasks, output_dir)


def run_tasks_parallel(tasks: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    if not tasks:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = base.resolve_worker_count(len(tasks))
    print(f"[{RUN_NAME}] submitting {len(tasks)} tasks with workers={worker_count} output={output_dir}")
    records: list[dict[str, Any]] = []
    if worker_count <= 1:
        for task in tasks:
            record = run_single_task(task)
            records.append(record)
            write_json(output_dir / "records.json", base.sorted_records(records))
            print_single_record(record)
        return base.sorted_records(records)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {executor.submit(run_single_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                record = future.result()
            except Exception as exc:
                record = task_error_record(task, exc)
            records.append(record)
            write_json(output_dir / "records.json", base.sorted_records(records))
            print_single_record(record)
    return base.sorted_records(records)


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
        analysis_network=str(task.get("analysis_network", network_case.name)),
        method=str(task["method"]),
        run_index=int(task["run_index"]),
        seed=int(task["seed"]),
        t_end=base.runner_t_end(task.get("t_end")),
        output_dir=Path(task["output_dir"]),
        parameter_case=parameter_case,
        max_steps=int(task["max_steps"]),
        max_runtime_seconds=float(task["max_runtime_seconds"]),
    )


def run_single(
    *,
    network: base.ReactionNetworkData,
    network_case: base.NetworkCase,
    analysis_network: str,
    method: str,
    run_index: int,
    seed: int,
    t_end: float,
    output_dir: Path,
    parameter_case: base.BlendedParameterCase | None,
    max_steps: int,
    max_runtime_seconds: float,
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
            max_steps=int(max_steps),
            max_runtime_seconds=float(max_runtime_seconds),
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
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        traceback_path = str(trace_path)
    wall_runtime = perf_counter() - started_at

    trajectory_path = output_dir / f"run_{int(run_index):04d}_seed_{int(seed)}.npz"
    if summary is not None:
        final_state = np.asarray(summary.final_state, dtype=float)
        record = {
            "status": "ok",
            "network": str(analysis_network),
            "physical_network": network_case.name,
            "max_len": int(network_case.max_len),
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
                    "network": str(analysis_network),
                    "physical_network": network_case.name,
                    "method": method,
                    "run_index": int(run_index),
                    "seed": int(seed),
                    "requested_t_end": base.json_float_or_none(t_end),
                    "max_steps": int(max_steps),
                    "max_runtime_seconds": float(max_runtime_seconds),
                    "wall_runtime_seconds": float(wall_runtime),
                    "stop_reason": stop_reason,
                    "parameter_case": base.parameter_payload(parameter_case),
                }
            )
            base.save_trajectory_record(trajectory_path, trajectory_record)
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
            "network": str(analysis_network),
            "physical_network": network_case.name,
            "max_len": int(network_case.max_len),
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
            "trajectory_path": "",
            "parameter_case": base.parameter_payload(parameter_case),
            "error": error,
            "traceback_path": traceback_path,
        }
    write_json(output_dir / f"{method}_run_{int(run_index):04d}_record.json", record)
    return record


def annotate_records(
    records: list[dict[str, Any]],
    *,
    physical_network: str,
    calibration_label: str,
    calibration_role: str,
) -> None:
    for record in records:
        record["physical_network"] = str(physical_network)
        record["network"] = str(physical_network)
        record["calibration_label"] = str(calibration_label)
        record["calibration_role"] = str(calibration_role)


def task_error_record(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    network_case = task["network_case"]
    if not isinstance(network_case, base.NetworkCase):
        network_case = base.NetworkCase(**dict(network_case))
    parameter_case = task.get("parameter_case")
    return {
        "status": "error",
        "network": str(task.get("analysis_network", network_case.name)),
        "physical_network": network_case.name,
        "max_len": int(network_case.max_len),
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
        "trajectory_path": "",
        "parameter_case": base.parameter_payload(parameter_case),
        "error": repr(exc),
        "traceback_path": "",
    }


def parallel_metadata() -> dict[str, Any]:
    return {
        "backend": "process",
        "parallel_workers": None if base.PARALLEL_WORKERS is None else int(base.PARALLEL_WORKERS),
        "resolved_cpu_count": os.cpu_count(),
        "stage_order": (
            "per network: parallel blended calibration for each parameter case "
            "-> max of per-parameter median blended t_end -> parallel SSA"
        ),
        "network_rebuild_per_worker": True,
    }


def stop_condition_metadata() -> dict[str, Any]:
    return {
        "memory_error": "caught MemoryError and stopped the single run",
        "blended_t_end": None,
        "blended_t_end_note": "blended calibration runs have no simulation-clock limit",
        "ssa_t_end": "max median finite blended simulation_final_time per network across parameter cases",
        "ssa_fallback_t_end": float(base.BASE_T_END),
        "max_steps": int(base.MAX_STEPS),
        "blended_max_runtime_seconds": float(base.MAX_RUNTIME_SECONDS),
        "ssa_max_runtime_seconds": float(SSA_MAX_RUNTIME_SECONDS),
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


def print_error_summary(summary: dict[str, Any]) -> None:
    if not bool(summary.get("had_errors", False)):
        print(f"[{RUN_NAME}] error_summary: no errors")
        return
    print(
        f"[{RUN_NAME}] error_summary: n_errors={summary.get('n_errors')} "
        f"reasons={summary.get('reasons')} messages={summary.get('messages')}"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def apply_environment_overrides() -> None:
    global SSA_MAX_RUNTIME_SECONDS
    base.apply_environment_overrides()
    SSA_MAX_RUNTIME_SECONDS = env_float("BETA_TEST_SSA_MAX_RUNTIME_SECONDS", SSA_MAX_RUNTIME_SECONDS)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default) if value is None or not value.strip() else float(value)


def main() -> None:
    apply_environment_overrides()
    run()


if __name__ == "__main__":
    main()
