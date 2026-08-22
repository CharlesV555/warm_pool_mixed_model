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
from polymer_sim import SummaryRecorder, WindowedTrajectoryRecorder, load_trajectory_record  # noqa: E402


RUN_NAME = "accuracy_effecency_betaHybrid__test_beta"

# In this reverse-calibration test, blended runs have no simulation-clock
# t_end. They stop by max_steps/max_runtime_seconds/no_progress. For each
# network, the median finite final time is computed for each blended parameter
# case. The minimum of those medians is used as a shared comparison t_end so
# every blended parameter case has enough simulated time for formal sampling.
BLENDED_MAX_STEPS = base.MAX_STEPS
BLENDED_MAX_WALL_TIME_SECONDS = base.MAX_RUNTIME_SECONDS

SSA_T_END_RULE = "min_median_blended_final_time_per_network"
SSA_MAX_STEPS = base.MAX_STEPS
SSA_MAX_WALL_TIME_SECONDS = 3_600.0
SSA_FALLBACK_T_END = base.BASE_T_END

N_SAMPLE_TIME_POINTS = 100
TARGET_EVENTS_PER_WINDOW = 100.0
LOCAL_WINDOW_ESTIMATION_RUNS = 3
MIN_SAMPLE_WINDOW_WIDTH = 1e-12


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

        calibration_records: list[dict[str, Any]] = []
        network_blended_medians: list[dict[str, Any]] = []
        global_window_widths: dict[str, float] = {}
        for parameter_case in blended_cases:
            case_dir = network_dir / parameter_case.label

            pre_dir = case_dir / "precompute_blended"
            pre_records = run_blended_calibration_batch(
                network_case=network_case,
                parameter_case=parameter_case,
                output_dir=pre_dir,
                seed_rng=seed_rng,
            )
            annotate_records(
                pre_records,
                physical_network=network_case.name,
                calibration_label=parameter_case.label,
                calibration_role="blended_precompute",
            )
            calibration_records.extend(pre_records)
            write_json(network_dir / "calibration_records.json", calibration_records)

            blended_median_t_end = median_final_time_or_none(pre_records)
            if blended_median_t_end is None:
                blended_median_t_end = float(SSA_FALLBACK_T_END)
            width = median_event_window_width(pre_records)
            global_window_widths[parameter_case.label] = float(width)
            network_blended_medians.append(
                {
                    "parameter_label": parameter_case.label,
                    "median_final_time": float(blended_median_t_end),
                    "global_window_width": float(width),
                    "n_records": len(pre_records),
                }
            )
            print(
                f"[{RUN_NAME}] network={network_case.name} parameter={parameter_case.label} "
                f"precompute_blended_median_final_time={blended_median_t_end:.6g} "
                f"global_window_width={width:.6g}"
            )

        finite_blended_medians = [
            float(item["median_final_time"])
            for item in network_blended_medians
            if np.isfinite(float(item["median_final_time"]))
        ]
        common_t_end = min(finite_blended_medians) if finite_blended_medians else float(SSA_FALLBACK_T_END)
        sample_times = np.linspace(0.0, float(common_t_end), int(N_SAMPLE_TIME_POINTS))
        ssa_t_end = float(common_t_end)
        ssa_pre_dir = network_dir / "precompute_ssa"
        ssa_pre_records = run_ssa_to_t_end_batch(
            network_case=network_case,
            t_end=ssa_t_end,
            output_dir=ssa_pre_dir,
            seed_rng=seed_rng,
            recording_mode="summary",
        )
        annotate_records(
            ssa_pre_records,
            physical_network=network_case.name,
            calibration_label="common_blended_min_t_end",
            calibration_role="ssa_precompute",
        )
        calibration_records.extend(ssa_pre_records)
        ssa_global_width = median_event_window_width(ssa_pre_records)

        global_windows_by_label: dict[str, np.ndarray] = {
            item["parameter_label"]: windows_from_widths(
                sample_times,
                np.full(sample_times.shape, float(global_window_widths[str(item["parameter_label"])]), dtype=float),
                float(common_t_end),
            )
            for item in network_blended_medians
        }
        global_windows_by_label["ssa"] = windows_from_widths(
            sample_times,
            np.full(sample_times.shape, float(ssa_global_width), dtype=float),
            float(common_t_end),
        )

        local_widths_by_label: dict[str, np.ndarray] = {}
        for parameter_case in blended_cases:
            local_dir = network_dir / parameter_case.label / "local_window_estimate_blended"
            local_records = run_blended_window_batch(
                network_case=network_case,
                parameter_case=parameter_case,
                t_end=float(common_t_end),
                output_dir=local_dir,
                seed_rng=seed_rng,
                windows=global_windows_by_label[parameter_case.label],
                n_runs=int(LOCAL_WINDOW_ESTIMATION_RUNS),
                recording_mode="windowed",
            )
            annotate_records(
                local_records,
                physical_network=network_case.name,
                calibration_label=parameter_case.label,
                calibration_role="blended_local_window_estimate",
            )
            calibration_records.extend(local_records)
            local_widths_by_label[parameter_case.label] = estimate_local_widths(
                global_windows_by_label[parameter_case.label],
                local_records,
                fallback_width=float(global_window_widths[parameter_case.label]),
            )

        ssa_local_dir = network_dir / "local_window_estimate_ssa"
        ssa_local_records = run_ssa_to_t_end_batch(
            network_case=network_case,
            t_end=float(common_t_end),
            output_dir=ssa_local_dir,
            seed_rng=seed_rng,
            windows=global_windows_by_label["ssa"],
            n_runs=int(LOCAL_WINDOW_ESTIMATION_RUNS),
            recording_mode="windowed",
        )
        annotate_records(
            ssa_local_records,
            physical_network=network_case.name,
            calibration_label="common_blended_min_t_end",
            calibration_role="ssa_local_window_estimate",
        )
        calibration_records.extend(ssa_local_records)
        local_widths_by_label["ssa"] = estimate_local_widths(
            global_windows_by_label["ssa"],
            ssa_local_records,
            fallback_width=float(ssa_global_width),
        )
        write_json(network_dir / "calibration_records.json", calibration_records)

        final_windows_by_label = {
            label: windows_from_widths(sample_times, widths, float(common_t_end))
            for label, widths in local_widths_by_label.items()
        }
        write_json(
            network_dir / "sampling_plan.json",
            {
                "network": network_case.name,
                "common_t_end_rule": "minimum median finite blended simulation_final_time across parameter cases",
                "common_t_end": float(common_t_end),
                "n_sample_time_points": int(N_SAMPLE_TIME_POINTS),
                "target_events_per_window": float(TARGET_EVENTS_PER_WINDOW),
                "sample_times": sample_times.tolist(),
                "blended_parameter_medians": network_blended_medians,
                "ssa_global_window_width": float(ssa_global_width),
                "local_window_estimation_runs": int(LOCAL_WINDOW_ESTIMATION_RUNS),
                "final_window_widths": {
                    label: np.asarray(windows[:, 1] - windows[:, 0], dtype=float).tolist()
                    for label, windows in final_windows_by_label.items()
                },
            },
        )
        print(
            f"[{RUN_NAME}] network={network_case.name} common_t_end={common_t_end:.6g} "
            f"sample_points={int(N_SAMPLE_TIME_POINTS)}"
        )

        for parameter_case in blended_cases:
            blended_dir = network_dir / parameter_case.label / "blended"
            blended_records = run_blended_window_batch(
                network_case=network_case,
                parameter_case=parameter_case,
                t_end=float(common_t_end),
                output_dir=blended_dir,
                seed_rng=seed_rng,
                windows=final_windows_by_label[parameter_case.label],
                n_runs=int(base.BLENDED_RUNS_PER_PARAMETER),
                recording_mode="windowed",
            )
            annotate_records(
                blended_records,
                physical_network=network_case.name,
                calibration_label=parameter_case.label,
                calibration_role="blended_formal_windowed",
            )
            all_records.extend(blended_records)
            write_json(output_root / "run_records.json", all_records)

        write_json(
            network_dir / "blended_t_end_calibration.json",
            {
                "network": network_case.name,
                "ssa_t_end_rule": str(SSA_T_END_RULE),
                "ssa_t_end": float(ssa_t_end),
                "common_t_end": float(common_t_end),
                "blended_parameter_medians": network_blended_medians,
            },
        )
        print(f"[{RUN_NAME}] network={network_case.name} SSA t_end={ssa_t_end:.6g}")

        ssa_dir = network_dir / "ssa"
        ssa_records = run_ssa_to_t_end_batch(
            network_case=network_case,
            t_end=ssa_t_end,
            output_dir=ssa_dir,
            seed_rng=seed_rng,
            windows=final_windows_by_label["ssa"],
            n_runs=int(base.SSA_RUNS_PER_NETWORK),
            recording_mode="windowed",
        )
        annotate_records(
            ssa_records,
            physical_network=network_case.name,
            calibration_label="common_blended_min_t_end",
            calibration_role="ssa_formal_windowed",
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
        "batch_report": base.batch_report(all_records),
        "records": all_records,
    }
    write_json(output_root / "batch_report.json", payload["batch_report"])
    write_json(output_root / "run_metadata.json", payload)
    print_batch_report(payload["batch_report"])
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
                "max_steps": int(BLENDED_MAX_STEPS),
                "max_runtime_seconds": float(BLENDED_MAX_WALL_TIME_SECONDS),
                "recording_mode": "summary",
                "windows": None,
            }
        )
    return run_tasks_parallel(tasks, output_dir)


def run_blended_window_batch(
    *,
    network_case: base.NetworkCase,
    parameter_case: base.BlendedParameterCase,
    t_end: float,
    output_dir: Path,
    seed_rng: np.random.Generator,
    windows: np.ndarray,
    n_runs: int,
    recording_mode: str,
) -> list[dict[str, Any]]:
    tasks = []
    for run_index in range(int(n_runs)):
        tasks.append(
            {
                "network_case": network_case,
                "analysis_network": network_case.name,
                "physical_network": network_case.name,
                "method": "blended",
                "run_index": int(run_index),
                "seed": base.next_seed(seed_rng),
                "t_end": float(t_end),
                "output_dir": output_dir,
                "parameter_case": parameter_case,
                "max_steps": int(BLENDED_MAX_STEPS),
                "max_runtime_seconds": float(BLENDED_MAX_WALL_TIME_SECONDS),
                "recording_mode": str(recording_mode),
                "windows": np.asarray(windows, dtype=float),
            }
        )
    return run_tasks_parallel(tasks, output_dir)


def run_ssa_to_t_end_batch(
    *,
    network_case: base.NetworkCase,
    t_end: float,
    output_dir: Path,
    seed_rng: np.random.Generator,
    windows: np.ndarray | None = None,
    n_runs: int | None = None,
    recording_mode: str = "windowed",
) -> list[dict[str, Any]]:
    tasks = []
    run_count = int(base.SSA_RUNS_PER_NETWORK if n_runs is None else n_runs)
    for run_index in range(run_count):
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
                "max_steps": int(SSA_MAX_STEPS),
                "max_runtime_seconds": float(SSA_MAX_WALL_TIME_SECONDS),
                "recording_mode": str(recording_mode),
                "windows": None if windows is None else np.asarray(windows, dtype=float),
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
        recording_mode=str(task.get("recording_mode", "windowed")),
        windows=task.get("windows"),
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
    recording_mode: str,
    windows: Any,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = make_recorder(recording_mode, windows)
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
            "trajectory_path": "" if str(recording_mode) == "summary" else str(trajectory_path),
            "final_total_abundance": float(final_state.sum()),
            "max_species_count": float(final_state.max()) if final_state.size else 0.0,
            "parameter_case": base.parameter_payload(parameter_case),
            "recording_mode": str(recording_mode),
            "error": "",
            "traceback_path": "",
        }
        try:
            if str(recording_mode) != "summary":
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
                        "recording_mode": str(recording_mode),
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
            "recording_mode": str(recording_mode),
            "error": error,
            "traceback_path": traceback_path,
        }
        partial_path = output_dir / f"run_{int(run_index):04d}_seed_{int(seed)}_partial.npz"
        if str(recording_mode) != "summary" and base.try_save_partial_trajectory(
            recorder,
            partial_path,
            metadata={
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
                "partial_after_error": True,
                "error": error,
                "parameter_case": base.parameter_payload(parameter_case),
            },
        ):
            record["trajectory_path"] = str(partial_path)
            record["partial_trajectory_saved"] = True
        else:
            record["partial_trajectory_saved"] = False
    write_json(output_dir / f"{method}_run_{int(run_index):04d}_record.json", record)
    return record


def make_recorder(recording_mode: str, windows: Any):
    mode = str(recording_mode)
    if mode == "summary":
        return SummaryRecorder()
    if mode == "windowed":
        if windows is None:
            raise ValueError("windowed recording requires windows")
        return WindowedTrajectoryRecorder(windows=np.asarray(windows, dtype=float))
    if mode == "full_trajectory":
        return base.TrajectoryRecorder()
    raise ValueError(f"unknown recording_mode {recording_mode!r}")


def median_final_time_or_none(records: list[dict[str, Any]]) -> float | None:
    values = [
        float(record["simulation_final_time"])
        for record in records
        if record.get("status") == "ok"
        and record.get("simulation_final_time") is not None
        and np.isfinite(float(record["simulation_final_time"]))
    ]
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def median_event_window_width(records: list[dict[str, Any]]) -> float:
    widths = []
    for record in records:
        final_time = record.get("simulation_final_time")
        n_events = record.get("n_events")
        if final_time is None or n_events is None:
            continue
        final = float(final_time)
        events = int(n_events)
        if np.isfinite(final) and final > 0.0 and events > 0:
            widths.append(float(TARGET_EVENTS_PER_WINDOW) * final / float(events))
    if not widths:
        return float(MIN_SAMPLE_WINDOW_WIDTH)
    return max(float(np.median(np.asarray(widths, dtype=float))), float(MIN_SAMPLE_WINDOW_WIDTH))


def windows_from_widths(sample_times: np.ndarray, widths: np.ndarray, common_t_end: float) -> np.ndarray:
    centers = np.asarray(sample_times, dtype=float)
    width_values = np.maximum(np.asarray(widths, dtype=float), float(MIN_SAMPLE_WINDOW_WIDTH))
    starts = np.maximum(centers - 0.5 * width_values, 0.0)
    ends = np.minimum(centers + 0.5 * width_values, float(common_t_end))
    ends = np.maximum(ends, starts)
    return np.column_stack([starts, ends])


def estimate_local_widths(
    windows: np.ndarray,
    records: list[dict[str, Any]],
    *,
    fallback_width: float,
) -> np.ndarray:
    base_windows = np.asarray(windows, dtype=float)
    counts = np.zeros(base_windows.shape[0], dtype=float)
    durations = np.maximum(base_windows[:, 1] - base_windows[:, 0], float(MIN_SAMPLE_WINDOW_WIDTH))
    usable_runs = 0
    for record in records:
        path = str(record.get("trajectory_path", "")).strip()
        if not path:
            continue
        try:
            trajectory = load_trajectory_record(path)
        except Exception:
            continue
        times = np.asarray(trajectory.times, dtype=float)
        if times.size == 0:
            continue
        usable_runs += 1
        for index, (start, end) in enumerate(base_windows):
            counts[index] += float(np.count_nonzero((times >= float(start)) & (times <= float(end))))
    fallback = max(float(fallback_width), float(MIN_SAMPLE_WINDOW_WIDTH))
    if usable_runs <= 0:
        return np.full(base_windows.shape[0], fallback, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        event_rate = counts / (durations * float(usable_runs))
        widths = float(TARGET_EVENTS_PER_WINDOW) / event_rate
    widths[~np.isfinite(widths) | (widths <= 0.0)] = fallback
    max_width = max(fallback * 10.0, float(MIN_SAMPLE_WINDOW_WIDTH))
    return np.clip(widths, float(MIN_SAMPLE_WINDOW_WIDTH), max_width)


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
        "recording_mode": str(task.get("recording_mode", "")),
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
        "blended_max_steps": int(BLENDED_MAX_STEPS),
        "blended_max_wall_time_seconds": float(BLENDED_MAX_WALL_TIME_SECONDS),
        "ssa_t_end_rule": str(SSA_T_END_RULE),
        "ssa_t_end": "computed per network from blended calibration",
        "ssa_fallback_t_end": float(SSA_FALLBACK_T_END),
        "ssa_max_steps": int(SSA_MAX_STEPS),
        "ssa_max_wall_time_seconds": float(SSA_MAX_WALL_TIME_SECONDS),
        "n_sample_time_points": int(N_SAMPLE_TIME_POINTS),
        "target_events_per_window": float(TARGET_EVENTS_PER_WINDOW),
        "local_window_estimation_runs": int(LOCAL_WINDOW_ESTIMATION_RUNS),
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


def print_batch_report(report: dict[str, Any]) -> None:
    print(f"[{RUN_NAME}] batch_report n_records={report.get('n_records')}")
    for method, item in dict(report.get("by_method", {})).items():
        print(
            f"[{RUN_NAME}] batch_report method={method} "
            f"n_total={item.get('n_total')} n_ok={item.get('n_ok')} "
            f"n_error={item.get('n_error')} n_reached_t_end={item.get('n_reached_t_end')} "
            f"stop_reasons={item.get('stop_reasons')} "
            f"simulation_final_time_quantiles={item.get('simulation_final_time_quantiles')}"
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def apply_environment_overrides() -> None:
    global BLENDED_MAX_STEPS
    global BLENDED_MAX_WALL_TIME_SECONDS
    global SSA_MAX_STEPS
    global SSA_MAX_WALL_TIME_SECONDS
    global SSA_FALLBACK_T_END
    global N_SAMPLE_TIME_POINTS
    global TARGET_EVENTS_PER_WINDOW
    global LOCAL_WINDOW_ESTIMATION_RUNS
    global MIN_SAMPLE_WINDOW_WIDTH
    base.apply_environment_overrides()
    BLENDED_MAX_STEPS = env_int("BETA_TEST_BLENDED_MAX_STEPS", base.MAX_STEPS)
    BLENDED_MAX_WALL_TIME_SECONDS = env_float(
        "BETA_TEST_BLENDED_MAX_WALL_TIME_SECONDS",
        base.MAX_RUNTIME_SECONDS,
    )
    SSA_MAX_STEPS = env_int("BETA_TEST_SSA_MAX_STEPS", base.MAX_STEPS)
    SSA_MAX_WALL_TIME_SECONDS = env_float(
        "BETA_TEST_SSA_MAX_WALL_TIME_SECONDS",
        env_float("BETA_TEST_SSA_MAX_RUNTIME_SECONDS", SSA_MAX_WALL_TIME_SECONDS),
    )
    SSA_FALLBACK_T_END = env_float("BETA_TEST_SSA_FALLBACK_T_END", base.BASE_T_END)
    N_SAMPLE_TIME_POINTS = env_int("BETA_TEST_N_SAMPLE_TIME_POINTS", N_SAMPLE_TIME_POINTS)
    TARGET_EVENTS_PER_WINDOW = env_float("BETA_TEST_TARGET_EVENTS_PER_WINDOW", TARGET_EVENTS_PER_WINDOW)
    LOCAL_WINDOW_ESTIMATION_RUNS = env_int(
        "BETA_TEST_LOCAL_WINDOW_ESTIMATION_RUNS",
        LOCAL_WINDOW_ESTIMATION_RUNS,
    )
    MIN_SAMPLE_WINDOW_WIDTH = env_float("BETA_TEST_MIN_SAMPLE_WINDOW_WIDTH", MIN_SAMPLE_WINDOW_WIDTH)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default) if value is None or not value.strip() else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default) if value is None or not value.strip() else float(value)


def main() -> None:
    apply_environment_overrides()
    run()


if __name__ == "__main__":
    main()
