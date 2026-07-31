from __future__ import annotations

"""Parallel batch entry for comparing several wall-clock budgets.

Each task is one (wall_seconds, network, method) simulation.  Tasks are run in
separate worker processes, individual cProfile files are written under
profiles/wall_<seconds>/, and aggregate reports are written only after every
task has finished.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_NETWORKS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SETTINGS,
    METHOD_ORDER,
    NETWORK_SPECS,
    RunSettings,
    normalize_method,
    run_method_profiled,
    write_profile_report,
    write_simulation_summary_tables,
)


NETWORKS = DEFAULT_NETWORKS
METHODS = METHOD_ORDER
OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "wall_time_sweep"
SETTINGS = DEFAULT_SETTINGS
WALL_SECONDS_LIST = (1.0, 3.0, 10.0)


def main() -> None:
    args = _parser().parse_args()
    wall_seconds_list = tuple(float(value) for value in args.wall_seconds_list)
    if not wall_seconds_list:
        raise ValueError("--wall-seconds-list must contain at least one value")
    for value in wall_seconds_list:
        if value <= 0.0:
            raise ValueError("--wall-seconds-list values must be > 0")

    settings = RunSettings(
        seed=int(args.seed),
        t_end=_parse_optional_float(args.t_end),
        max_steps=int(args.max_steps),
        max_runtime_seconds=float(wall_seconds_list[0]),
        blended_i1=float(args.blended_i1),
        blended_i2=float(args.blended_i2),
        blended_dt_cle=float(args.blended_dt_cle),
        blended_dt_macro=float(args.blended_dt_macro),
        pdmp_ode_step=float(args.pdmp_ode_step),
    )
    output_dir = Path(args.output_dir)
    methods = tuple(normalize_method(method) for method in args.methods)
    networks = tuple(args.networks)
    tasks = [
        {
            "wall_seconds": wall_seconds,
            "network": network,
            "method": method,
        }
        for wall_seconds in wall_seconds_list
        for network in networks
        for method in methods
    ]
    workers = int(args.workers) if args.workers is not None else min(len(tasks), os.cpu_count() or 1)
    workers = max(1, min(workers, len(tasks)))

    print(
        "[wall-time sweep] "
        f"tasks={len(tasks)} workers={workers} "
        f"networks={list(networks)} methods={list(methods)} wall_seconds={list(wall_seconds_list)}"
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if workers == 1:
        for task in tasks:
            try:
                record = _run_wall_time_task(
                    task["method"],
                    task["network"],
                    float(task["wall_seconds"]),
                    settings,
                    str(output_dir),
                    int(args.profile_limit),
                )
            except Exception as exc:
                failure = {
                    "wall_seconds": float(task["wall_seconds"]),
                    "network": str(task["network"]),
                    "method": str(task["method"]),
                    "error": repr(exc),
                }
                failures.append(failure)
                _print_failure(failure)
                continue
            records.append(record)
            _print_done(record)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(
                    _run_wall_time_task,
                    task["method"],
                    task["network"],
                    float(task["wall_seconds"]),
                    settings,
                    str(output_dir),
                    int(args.profile_limit),
                ): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    record = future.result()
                except Exception as exc:
                    failure = {
                        "wall_seconds": float(task["wall_seconds"]),
                        "network": str(task["network"]),
                        "method": str(task["method"]),
                        "error": repr(exc),
                    }
                    failures.append(failure)
                    _print_failure(failure)
                    continue
                records.append(record)
                _print_done(record)

    records.sort(key=lambda item: (float(item["max_runtime_seconds"]), str(item["network"]), str(item["method"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    if failures:
        failure_path = output_dir / "report" / "wall_time_sweep_failures.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[wall-time sweep] wrote failures: {failure_path}")

    write_profile_report(records, output_dir, profile_limit=int(args.profile_limit))
    write_simulation_summary_tables(records, output_dir)
    if failures:
        raise RuntimeError(f"{len(failures)} wall-time sweep tasks failed; see report/wall_time_sweep_failures.json")


def _print_done(record: dict[str, Any]) -> None:
    print(
        "[done] "
        f"wall={record['max_runtime_seconds']:.6g}s "
        f"network={record['network']} method={record['method']} "
        f"virtual_time={record['simulation_final_time']:.6g} "
        f"steps={record['n_steps']} events={record['n_events']} "
        f"stop={record['stop_reason']}"
    )


def _print_failure(failure: dict[str, Any]) -> None:
    print(
        "[failed] "
        f"wall={failure['wall_seconds']} network={failure['network']} "
        f"method={failure['method']} error={failure['error']}"
    )


def _run_wall_time_task(
    method: str,
    network: str,
    wall_seconds: float,
    base_settings: RunSettings,
    output_dir: str,
    profile_limit: int,
) -> dict[str, Any]:
    settings = replace(base_settings, max_runtime_seconds=float(wall_seconds))
    profile_dir = Path(output_dir) / "profiles" / f"wall_{_wall_tag(wall_seconds)}"
    record, _profile_text = run_method_profiled(
        method,
        network,
        settings,
        output_dir=Path(output_dir),
        profile_dir=profile_dir,
        profile_limit=int(profile_limit),
        write_json=False,
    )
    record["wall_budget_tag"] = _wall_tag(wall_seconds)
    return record


def _wall_tag(value: float) -> str:
    return f"{float(value):.10g}".replace(".", "p").replace("-", "m")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run every selected network/method at every wall-clock budget in parallel."
    )
    parser.add_argument("--wall-seconds-list", "--wall-times", nargs="+", type=float, default=list(WALL_SECONDS_LIST))
    parser.add_argument("--networks", nargs="+", default=list(NETWORKS), choices=sorted(NETWORK_SPECS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHOD_ORDER))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SETTINGS.seed)
    parser.add_argument("--t-end", default="none", help="Use 'none' to stop by wall-clock/max-steps.")
    parser.add_argument("--max-steps", type=int, default=SETTINGS.max_steps)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--blended-i1", type=float, default=SETTINGS.blended_i1)
    parser.add_argument("--blended-i2", type=float, default=SETTINGS.blended_i2)
    parser.add_argument("--blended-dt-cle", type=float, default=SETTINGS.blended_dt_cle)
    parser.add_argument("--blended-dt-macro", type=float, default=SETTINGS.blended_dt_macro)
    parser.add_argument("--pdmp-ode-step", type=float, default=SETTINGS.pdmp_ode_step)
    parser.add_argument("--profile-limit", type=int, default=40)
    return parser


def _parse_optional_float(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "inf", "infinity"}:
        return None
    return float(text)


if __name__ == "__main__":
    main()
