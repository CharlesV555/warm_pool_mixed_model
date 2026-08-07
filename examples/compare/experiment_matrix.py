from __future__ import annotations

"""Experiment matrix runner for compare benchmarks.

This layer keeps two separate tables:
- test_config.csv/xlsx: input design matrix. Rows are method configurations,
  columns are network instances, and each cell is a JSON run specification.
- test_result.csv/xlsx: output result matrix with the same index/columns. Each
  completed cell is a JSON result containing runtime, simulated time, events,
  trajectory path, and lightweight memory metrics.

CSV is the canonical editable format.  XLSX files are also written for viewing
in spreadsheet tools without requiring openpyxl/xlsxwriter.
"""

import argparse
import cProfile
import io
import json
import os
import pstats
import shutil
import tracemalloc
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from common import (
    DEFAULT_NETWORKS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SETTINGS,
    METHOD_ORDER,
    NETWORK_SPECS,
    RunSettings,
    build_compare_food_restriction,
    build_network,
    json_ready,
    make_stepper,
    normalize_method,
    normalized_food_supply_mode,
    prepare_network_for_method,
    spec_uses_explicit_food_inflow,
)
from polymer_sim import ExperimentRunner, TrajectoryRecorder, save_trajectory_record


MATRIX_OUTPUT_SUBDIR = "experiment_matrix"
DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_DIR
CONFIG_CSV_NAME = "test_config.csv"
CONFIG_XLSX_NAME = "test_config.xlsx"
RESULT_CSV_NAME = "test_result.csv"
RESULT_XLSX_NAME = "test_result.xlsx"
RESULT_LONG_CSV_NAME = "test_result_long.csv"
RESULT_LONG_XLSX_NAME = "test_result_long.xlsx"
DEFAULT_PROFILE_LIMIT = 40


def create_default_config_dataframe(
    *,
    networks: Sequence[str] = DEFAULT_NETWORKS,
    wall_seconds: float = 1.0,
    max_steps: int = 100_000_000,
    seed: int = 123,
    t_end: float | None = None,
) -> pd.DataFrame:
    """Create a matrix of default local/global update experiments."""

    method_configs = [
        (
            "ssa_local",
            "gillespie_ssa",
            {"ssa_use_local_propensity_updates": True},
        ),
        (
            "ssa_global",
            "gillespie_ssa",
            {"ssa_use_local_propensity_updates": False},
        ),
        (
            "nrm_dependency",
            "optimized_nrm",
            {"nrm_use_dependency_graph": True, "nrm_fallback_full_recompute": True},
        ),
        (
            "nrm_global",
            "optimized_nrm",
            {"nrm_use_dependency_graph": False, "nrm_fallback_full_recompute": True},
        ),
        (
            "pdmp_gillespie_local",
            "gillespie_pdmp_lp",
            {"pdmp_use_local_propensity_updates": True},
        ),
        (
            "pdmp_gillespie_global",
            "gillespie_pdmp_lp",
            {"pdmp_use_local_propensity_updates": False},
        ),
        (
            "pdmp_nrm_local",
            "nrm_pdmp_lp",
            {"pdmp_use_local_propensity_updates": True},
        ),
        (
            "pdmp_nrm_global",
            "nrm_pdmp_lp",
            {"pdmp_use_local_propensity_updates": False},
        ),
        (
            "gillespie_cle_hybrid",
            "gillespie_cle_hybrid",
            {},
        ),
        (
            "nrm_cle_hybrid",
            "nrm_cle_hybrid",
            {},
        ),
    ]
    network_names = [str(network) for network in networks]
    rows: dict[str, dict[str, str]] = {}
    for config_id, method, overrides in method_configs:
        settings = {
            "seed": int(seed),
            "t_end": t_end,
            "max_steps": int(max_steps),
            "max_runtime_seconds": float(wall_seconds),
            **overrides,
        }
        rows[config_id] = {
            network: _json_cell(
                {
                    "enabled": True,
                    "function": "run_matrix_cell",
                    "method": method,
                    "settings": settings,
                }
            )
            for network in network_names
        }
    df = pd.DataFrame.from_dict(rows, orient="index", columns=network_names)
    df.index.name = "method_config_id"
    return df


def write_config_files(config: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Write test_config.csv and test_config.xlsx."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / CONFIG_CSV_NAME
    xlsx_path = out_dir / CONFIG_XLSX_NAME
    config.to_csv(csv_path, encoding="utf-8")
    write_simple_xlsx(config, xlsx_path, sheet_name="test_config")
    return {"csv": csv_path, "xlsx": xlsx_path}


def load_config_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a config matrix from CSV or the simple XLSX emitted by this module."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(config_path, index_col=0, dtype=str).fillna("")
    elif suffix == ".xlsx":
        df = read_simple_xlsx(config_path)
    else:
        raise ValueError("config path must end with .csv or .xlsx")
    df.index = [str(index) for index in df.index]
    df.index.name = df.index.name or "method_config_id"
    df.columns = [str(column) for column in df.columns]
    return df.fillna("")


def run_experiment_matrix(
    config: pd.DataFrame,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    timestamp: str | None = None,
    workers: int = 1,
    copy_config_from: str | Path | None = None,
    profile: bool = True,
    profile_limit: int = DEFAULT_PROFILE_LIMIT,
) -> dict[str, Path | pd.DataFrame]:
    """Run enabled matrix cells and write result matrices after all tasks finish."""

    run_dir = matrix_run_dir(output_root, timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_files(config, run_dir)
    if copy_config_from is not None:
        source = Path(copy_config_from)
        if source.exists() and source.resolve() not in {Path(run_dir / CONFIG_CSV_NAME).resolve(), Path(run_dir / CONFIG_XLSX_NAME).resolve()}:
            shutil.copy2(source, run_dir / f"input_{source.name}")

    tasks, result_df, long_rows = _tasks_from_config(
        config,
        run_dir,
        profile=bool(profile),
        profile_limit=int(profile_limit),
    )
    worker_count = max(1, min(int(workers), len(tasks) if tasks else 1))
    print(f"[matrix] run_dir={run_dir}")
    print(
        f"[matrix] enabled_tasks={len(tasks)} workers={worker_count} "
        f"profile={bool(profile)} profile_limit={int(profile_limit)}"
    )

    if worker_count == 1:
        for task in tasks:
            result = _run_task_capture_errors(task)
            _store_result(result_df, long_rows, result)
            _print_task_result(result)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {executor.submit(_run_task_capture_errors, task): task for task in tasks}
            for future in as_completed(future_to_task):
                result = future.result()
                _store_result(result_df, long_rows, result)
                _print_task_result(result)

    paths = write_result_files(result_df, long_rows, run_dir)
    paths["run_dir"] = run_dir
    return {"result": result_df, "long_result": pd.DataFrame(long_rows), **paths}


def write_result_files(
    result: pd.DataFrame,
    long_rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write test_result.csv/xlsx and a long result companion table."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_csv = out_dir / RESULT_CSV_NAME
    result_xlsx = out_dir / RESULT_XLSX_NAME
    long_csv = out_dir / RESULT_LONG_CSV_NAME
    long_xlsx = out_dir / RESULT_LONG_XLSX_NAME
    result.to_csv(result_csv, encoding="utf-8")
    write_simple_xlsx(result, result_xlsx, sheet_name="test_result")
    long_df = pd.DataFrame(list(long_rows))
    long_df.to_csv(long_csv, index=False, encoding="utf-8")
    write_simple_xlsx(long_df, long_xlsx, sheet_name="test_result_long", include_index=False)
    print(f"[matrix] wrote result matrix: {result_csv}")
    print(f"[matrix] wrote result matrix xlsx: {result_xlsx}")
    print(f"[matrix] wrote long results: {long_csv}")
    return {
        "result_csv": result_csv,
        "result_xlsx": result_xlsx,
        "result_long_csv": long_csv,
        "result_long_xlsx": long_xlsx,
    }


def run_matrix_cell(
    *,
    config_id: str,
    network_name: str,
    method: str,
    settings: RunSettings,
    output_dir: str | Path,
    profile: bool = True,
    profile_limit: int = DEFAULT_PROFILE_LIMIT,
) -> dict[str, Any]:
    """Unified run wrapper used for every algorithm/config/network cell."""

    method_key = normalize_method(method)
    if not profile:
        return _run_matrix_cell_body(
            config_id=config_id,
            network_name=network_name,
            method=method_key,
            settings=settings,
            output_dir=output_dir,
        )

    profile_dir = Path(output_dir) / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = f"{safe_name(config_id)}__{safe_name(network_name)}__{safe_name(method_key)}"
    profile_path = profile_dir / f"{safe_stem}.prof"
    report_path = profile_dir / f"{safe_stem}_top{int(profile_limit)}.txt"
    profiler = cProfile.Profile()
    try:
        record = profiler.runcall(
            _run_matrix_cell_body,
            config_id=config_id,
            network_name=network_name,
            method=method_key,
            settings=settings,
            output_dir=output_dir,
        )
    finally:
        profiler.dump_stats(profile_path)
        _write_profile_report(profile_path, report_path, int(profile_limit))

    profile_entries = _read_profile_entries(profile_path, int(profile_limit))
    record["profile_prof_path"] = str(profile_path)
    record["profile_report_path"] = str(report_path)
    record["profile_top"] = profile_entries
    return record


def _run_matrix_cell_body(
    *,
    config_id: str,
    network_name: str,
    method: str,
    settings: RunSettings,
    output_dir: str | Path,
) -> dict[str, Any]:
    method_key = normalize_method(method)
    run_dir = Path(output_dir)
    trajectory_dir = run_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = f"{safe_name(config_id)}__{safe_name(network_name)}__{safe_name(method_key)}"
    trajectory_path = trajectory_dir / f"{safe_stem}.npz"

    tracemalloc.start()
    rss_before = process_rss_mb()
    wall_started = perf_counter()
    build_started = perf_counter()
    network, catalysis_result, spec = build_network(network_name, food_supply_mode=settings.food_supply_mode)
    network, catalysis_result = prepare_network_for_method(method_key, network, catalysis_result)
    build_wall_seconds = perf_counter() - build_started
    stepper, dt = make_stepper(method_key, settings, network)
    restriction = build_compare_food_restriction(network, spec)
    recorder = TrajectoryRecorder()

    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=runner_t_end(settings.t_end),
        seed=int(settings.seed),
        dt=dt,
        max_steps=int(settings.max_steps),
        max_runtime_seconds=float(settings.max_runtime_seconds),
        restriction=restriction,
        recorder=recorder,
        network_build_elapsed_seconds=build_wall_seconds,
    )
    wall_runtime_seconds = perf_counter() - wall_started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process_rss_mb()

    trajectory = recorder.finalize()
    trajectory.run_metadata.update(
        {
            "config_id": str(config_id),
            "network": str(spec.name),
            "method": str(method_key),
            "trajectory_path": str(trajectory_path),
            "run_settings": asdict(settings),
            "catalysis_assignment": json_ready(catalysis_result),
        }
    )
    save_trajectory_record(trajectory_path, trajectory)
    summary = result.summary
    final_state = np.asarray(summary.final_state, dtype=float)
    return {
        "status": "ok",
        "config_id": str(config_id),
        "network": str(spec.name),
        "method": str(method_key),
        "seed": int(settings.seed),
        "simulation_final_time": float(summary.final_time),
        "n_steps": int(summary.n_steps),
        "n_events": int(summary.n_events),
        "stop_reason": str(summary.metadata.get("stop_reason")),
        "wall_runtime_seconds": float(wall_runtime_seconds),
        "network_build_wall_seconds": float(build_wall_seconds),
        "python_memory_current_mb": bytes_to_mb(current_bytes),
        "python_memory_peak_mb": bytes_to_mb(peak_bytes),
        "process_rss_before_mb": rss_before,
        "process_rss_after_mb": rss_after,
        "process_rss_delta_mb": None if rss_before is None or rss_after is None else float(rss_after - rss_before),
        "trajectory_path": str(trajectory_path),
        "n_trajectory_points": int(trajectory.times.shape[0]),
        "n_species": int(network.n_species),
        "n_channels": int(network.n_channels),
        "food_supply_mode": normalized_food_supply_mode(spec),
        "uses_explicit_food_inflow": bool(spec_uses_explicit_food_inflow(spec)),
        "uses_food_restriction": restriction is not None,
        "final_total_abundance": float(final_state.sum()),
        "max_species_count": float(final_state.max()) if final_state.size else 0.0,
        "error": "",
    }


def main() -> None:
    args = parser().parse_args()
    output_root = Path(args.output_root)
    timestamp = args.timestamp or make_timestamp()
    if args.config:
        config = load_config_dataframe(args.config)
        config_source = args.config
    else:
        config = create_default_config_dataframe(
            networks=tuple(args.networks),
            wall_seconds=float(args.wall_seconds),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
            t_end=parse_optional_float(args.t_end),
        )
        config_source = None

    run_dir = matrix_run_dir(output_root, timestamp)
    config_paths = write_config_files(config, run_dir)
    print(f"[matrix] wrote config csv: {config_paths['csv']}")
    print(f"[matrix] wrote config xlsx: {config_paths['xlsx']}")
    if args.no_run:
        return

    run_experiment_matrix(
        config,
        output_root=output_root,
        timestamp=timestamp,
        workers=int(args.workers),
        copy_config_from=config_source,
        profile=bool(args.profile),
        profile_limit=int(args.profile_limit),
    )


def _tasks_from_config(
    config: pd.DataFrame,
    run_dir: Path,
    *,
    profile: bool,
    profile_limit: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    result_df = pd.DataFrame("", index=config.index.copy(), columns=config.columns.copy())
    result_df.index.name = config.index.name or "method_config_id"
    long_rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for config_id in config.index:
        for network_name in config.columns:
            raw_cell = config.at[config_id, network_name]
            try:
                cell = parse_config_cell(raw_cell)
            except Exception as exc:
                disabled = disabled_result(config_id, network_name, error=f"invalid config cell: {exc!r}")
                _store_result(result_df, long_rows, disabled)
                continue
            if not cell.get("enabled", False):
                disabled = disabled_result(config_id, network_name)
                _store_result(result_df, long_rows, disabled)
                continue
            method = str(cell.get("method", "")).strip()
            if not method:
                disabled = disabled_result(config_id, network_name, error="enabled cell has no method")
                _store_result(result_df, long_rows, disabled)
                continue
            settings = settings_from_overrides(cell.get("settings", {}))
            tasks.append(
                {
                    "config_id": str(config_id),
                    "network_name": str(cell.get("network", network_name) or network_name),
                    "method": method,
                    "settings": settings,
                    "output_dir": str(run_dir),
                    "profile": bool(profile),
                    "profile_limit": int(profile_limit),
                }
            )
    return tasks, result_df, long_rows


def _run_task_capture_errors(task: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_matrix_cell(**task)
    except Exception as exc:
        return {
            "status": "error",
            "config_id": str(task.get("config_id")),
            "network": str(task.get("network_name")),
            "method": str(task.get("method")),
            "simulation_final_time": None,
            "n_steps": None,
            "n_events": None,
            "wall_runtime_seconds": None,
            "trajectory_path": "",
            "python_memory_peak_mb": None,
            "profile_prof_path": "",
            "profile_report_path": "",
            "error": repr(exc),
        }


def _store_result(result_df: pd.DataFrame, long_rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    config_id = str(result.get("config_id"))
    network = str(result.get("network"))
    if config_id in result_df.index and network in result_df.columns:
        result_df.at[config_id, network] = _json_cell(result)
    long_rows.append(dict(result))


def _print_task_result(result: dict[str, Any]) -> None:
    status = result.get("status")
    print(
        "[matrix] "
        f"status={status} config={result.get('config_id')} network={result.get('network')} "
        f"method={result.get('method')} sim_time={result.get('simulation_final_time')} "
        f"events={result.get('n_events')} wall={result.get('wall_runtime_seconds')} "
        f"trajectory={result.get('trajectory_path')} profile={result.get('profile_report_path')} "
        f"error={result.get('error')}"
    )


def parse_config_cell(value: Any) -> dict[str, Any]:
    if value is None:
        return {"enabled": False}
    if isinstance(value, float) and np.isnan(value):
        return {"enabled": False}
    if isinstance(value, dict):
        cell = dict(value)
    else:
        text = str(value).strip()
        if not text:
            return {"enabled": False}
        if text.lower() in {"0", "false", "no", "disabled"}:
            return {"enabled": False}
        if text.lower() in {"1", "true", "yes", "enabled"}:
            return {"enabled": True}
        cell = json.loads(text)
    cell["enabled"] = bool(cell.get("enabled", True))
    return cell


def settings_from_overrides(overrides: Any) -> RunSettings:
    if overrides is None:
        overrides = {}
    if isinstance(overrides, str):
        overrides = json.loads(overrides) if overrides.strip() else {}
    if not isinstance(overrides, dict):
        raise TypeError("settings must be a JSON object")
    valid_fields = {field.name for field in fields(RunSettings)}
    unknown = sorted(str(key) for key in overrides if str(key) not in valid_fields)
    if unknown:
        raise ValueError(f"unknown RunSettings fields: {unknown}")
    base = DEFAULT_SETTINGS
    cleaned = {str(key): value for key, value in overrides.items()}
    return replace(base, **cleaned)


def disabled_result(config_id: str, network: str, *, error: str = "") -> dict[str, Any]:
    return {
        "status": "disabled" if not error else "error",
        "config_id": str(config_id),
        "network": str(network),
        "method": "",
        "simulation_final_time": None,
        "n_steps": None,
        "n_events": None,
        "wall_runtime_seconds": None,
        "trajectory_path": "",
        "python_memory_peak_mb": None,
        "profile_prof_path": "",
        "profile_report_path": "",
        "error": str(error),
    }


def _write_profile_report(profile_path: Path, report_path: Path, profile_limit: int) -> None:
    stream = io.StringIO()
    stats = pstats.Stats(str(profile_path), stream=stream)
    stats.strip_dirs().sort_stats("cumtime").print_stats(int(profile_limit))
    report_path.write_text(stream.getvalue(), encoding="utf-8")


def _read_profile_entries(profile_path: Path, profile_limit: int) -> list[dict[str, object]]:
    stats = pstats.Stats(str(profile_path), stream=io.StringIO())
    stats.strip_dirs().sort_stats("cumtime")
    entries: list[dict[str, object]] = []
    functions = list(stats.fcn_list or [])
    for rank, func in enumerate(functions[: int(profile_limit)], start=1):
        primitive_calls, total_calls, tottime, cumtime, _callers = stats.stats[func]
        filename, line_no, function_name = func
        primitive = int(primitive_calls)
        total = int(total_calls)
        entries.append(
            {
                "rank": int(rank),
                "primitive_calls": primitive,
                "total_calls": total,
                "tottime": float(tottime),
                "percall_tottime": None if total <= 0 else float(tottime / total),
                "cumtime": float(cumtime),
                "percall_cumtime": None if primitive <= 0 else float(cumtime / primitive),
                "function": f"{filename}:{int(line_no)}({function_name})",
            }
        )
    return entries


def write_simple_xlsx(
    df: pd.DataFrame,
    path: str | Path,
    *,
    sheet_name: str = "Sheet1",
    include_index: bool = True,
) -> None:
    """Write a minimal XLSX workbook with string cells only."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = dataframe_rows(df, include_index=include_index)
    sheet_xml = make_sheet_xml(rows)
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
        "<fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>"
        "<borders count=\"1\"><border/></borders>"
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        "<cellXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/></cellXfs>"
        "</styleSheet>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def read_simple_xlsx(path: str | Path) -> pd.DataFrame:
    """Read the first sheet of a simple XLSX workbook into a DataFrame."""

    with zipfile.ZipFile(Path(path), "r") as archive:
        shared_strings = read_shared_strings(archive)
        sheet_xml = archive.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(sheet_xml)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    grid: dict[tuple[int, int], str] = {}
    max_row = 0
    max_col = 0
    for cell in root.findall(".//x:c", ns):
        ref = cell.attrib.get("r", "")
        row_idx, col_idx = split_cell_ref(ref)
        max_row = max(max_row, row_idx)
        max_col = max(max_col, col_idx)
        grid[(row_idx, col_idx)] = cell_text(cell, shared_strings, ns)
    values = [
        [grid.get((row, col), "") for col in range(1, max_col + 1)]
        for row in range(1, max_row + 1)
    ]
    if not values:
        return pd.DataFrame()
    header = values[0]
    body = values[1:]
    if not body:
        return pd.DataFrame(columns=header[1:]).rename_axis(header[0] or "method_config_id")
    index = [row[0] for row in body]
    data = [row[1:] for row in body]
    df = pd.DataFrame(data, index=index, columns=header[1:])
    df.index.name = header[0] or "method_config_id"
    return df


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: list[str] = []
    for item in root.findall("x:si", ns):
        texts = [node.text or "" for node in item.findall(".//x:t", ns)]
        values.append("".join(texts))
    return values


def dataframe_rows(df: pd.DataFrame, *, include_index: bool) -> list[list[str]]:
    if include_index:
        header = [df.index.name or "index", *[str(column) for column in df.columns]]
        rows = [header]
        for index, row in df.iterrows():
            rows.append([str(index), *[format_cell_value(value) for value in row.tolist()]])
        return rows
    rows = [[str(column) for column in df.columns]]
    for _index, row in df.iterrows():
        rows.append([format_cell_value(value) for value in row.tolist()])
    return rows


def make_sheet_xml(rows: Sequence[Sequence[str]]) -> str:
    xml_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            ref = f"{column_letter(col_idx)}{row_idx}"
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def cell_text(cell: ET.Element, shared_strings: Sequence[str], ns: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text_node = cell.find("x:is/x:t", ns)
        return "" if text_node is None or text_node.text is None else str(text_node.text)
    value_node = cell.find("x:v", ns)
    value = "" if value_node is None or value_node.text is None else str(value_node.text)
    if cell_type == "s" and value:
        idx = int(value)
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
    return value


def split_cell_ref(ref: str) -> tuple[int, int]:
    letters = "".join(char for char in ref if char.isalpha())
    digits = "".join(char for char in ref if char.isdigit())
    col = 0
    for char in letters.upper():
        col = col * 26 + (ord(char) - ord("A") + 1)
    return int(digits or 1), int(col or 1)


def column_letter(index: int) -> str:
    letters = []
    value = int(index)
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters or ["A"]))


def format_cell_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _json_cell(value: dict[str, Any]) -> str:
    return json.dumps(json_ready(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def runner_t_end(value: float | None) -> float:
    return float("inf") if value is None else float(value)


def matrix_run_dir(output_root: str | Path = DEFAULT_OUTPUT_ROOT, timestamp: str | None = None) -> Path:
    """Return the canonical matrix output directory: output_root/timestamp/experiment_matrix."""

    return Path(output_root) / (timestamp or make_timestamp()) / MATRIX_OUTPUT_SUBDIR


def parse_optional_float(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "inf", "infinity"}:
        return None
    return float(text)


def bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def process_rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create and run compare experiment matrices.")
    p.add_argument("--config", default=None, help="Existing test_config.csv/xlsx. Omit to create a default matrix.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--timestamp", default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-run", action="store_true", help="Only write test_config.csv/xlsx.")
    p.add_argument("--networks", nargs="+", default=list(DEFAULT_NETWORKS), choices=sorted(NETWORK_SPECS))
    p.add_argument("--wall-seconds", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=DEFAULT_SETTINGS.seed)
    p.add_argument("--t-end", default="none")
    p.add_argument("--max-steps", type=int, default=DEFAULT_SETTINGS.max_steps)
    p.add_argument("--profile", dest="profile", action="store_true")
    p.add_argument("--no-profile", dest="profile", action="store_false")
    p.set_defaults(profile=True)
    p.add_argument("--profile-limit", type=int, default=DEFAULT_PROFILE_LIMIT)
    return p


if __name__ == "__main__":
    main()
