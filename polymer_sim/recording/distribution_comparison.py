"""Distribution comparison utilities for batched trajectory files.

Expected batch data structures
------------------------------
1. Paired metadata JSON from ``examples/multiple_run.py``:

   {
       "shared": {...},
       "runs": [
           {
               "mode": "ssa" | "blended" | ...,
               "stepper_method": "ssa" | "blended" | ...,
               "seed": int,
               "trajectory_path": "path/to/run.npz",
               ...
           },
           ...
       ]
   }

2. A folder containing trajectory ``.npz`` files written by
   ``save_trajectory_record(...)`` in ``polymer_sim.recording.trajectory``.
   Each file contains ``times``, ``states``, ``species_names``, and
   ``run_metadata_json``.

If future batch metadata adds new fields, change the metadata readers in
``_metadata_rows(...)``, ``_trajectory_path_from_row(...)``, and
``_group_from_row(...)``.  The extraction and plotting code only depends on
resolved trajectory paths, group labels, species ids, and time points.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from time import strftime
import csv
import json
import math
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polymer_sim.recording.base import PathLike


@dataclass(slots=True)
class BatchTrajectoryItem:
    """One trajectory row resolved from metadata JSON or a trajectory folder.

    ``group`` is usually the simulation method, such as ``ssa`` or
    ``blended``.  To use a new grouping field in future metadata, pass
    ``group_key=...`` to ``resolve_batch_trajectory_items(...)`` or update
    ``_group_from_row(...)``.
    """

    index: int
    trajectory_path: Path
    group: str
    seed: int | None = None
    label: str = ""


@dataclass(slots=True)
class DistributionExtraction:
    """Small temporary sample files extracted from full trajectories.

    Each temporary ``.npz`` file contains only:
    ``time_points``, ``species_ids``, ``species_names``, and ``values`` with
    shape ``(n_time_points, n_species)`` for one trajectory.  Plotting and
    statistical comparison should read these files instead of reloading full
    trajectory arrays.
    """

    temp_dir: Path
    manifest_path: Path
    sample_files: list[Path]
    groups: dict[str, list[Path]]
    species_ids: list[int]
    species_names: list[str]
    time_points: np.ndarray


@dataclass(slots=True)
class DistributionComparisonResult:
    """Paths and summary metrics produced by distribution comparison."""

    extraction: DistributionExtraction
    within_group_plots: list[Path]
    comparison_plots: list[Path]
    statistics_csv: Path | None
    statistics_json: Path | None
    non_significant_ratio: float | None


def compare_species_distributions(
    batch_source: PathLike | Sequence[PathLike],
    species: int | str | Sequence[int | str],
    *,
    time_points: Iterable[float] | None = None,
    time_range: tuple[float, float] | None = None,
    n_time_points: int = 20,
    group_key: str = "mode",
    groups: Sequence[str] | None = None,
    pattern: str = "*.npz",
    recursive: bool = False,
    output_dir: PathLike | None = None,
    temp_dir: PathLike | None = None,
    alpha: float = 0.05,
    outside: str = "nan",
    dpi: int = 150,
    print_memory: bool = True,
) -> DistributionComparisonResult:
    """Extract, plot, and compare species distributions across batch groups.

    Parameters
    ----------
    batch_source:
        Metadata JSON, trajectory folder, single trajectory ``.npz``, or a
        list of trajectory paths.
    species:
        Species ids or species names to compare.
    time_points / time_range:
        Use explicit ``time_points`` or generate ``n_time_points`` evenly
        spaced points over ``time_range``.
    group_key:
        Metadata key used for grouping.  Current batch files usually use
        ``mode`` and ``stepper_method``.  If future metadata introduces a
        different grouping field, pass it here.
    outside:
        ``"nan"`` marks requested times outside a trajectory range as missing;
        ``"hold"`` clips to the first or final recorded state.

    Returns
    -------
    DistributionComparisonResult
        Includes paths for temporary sample files, plots, and rank-sum
        statistics.
    """

    out_dir = _default_output_dir(batch_source, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(temp_dir) if temp_dir is not None else out_dir / "temp_species_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    items = resolve_batch_trajectory_items(
        batch_source,
        group_key=group_key,
        groups=groups,
        pattern=pattern,
        recursive=recursive,
    )
    if not items:
        raise ValueError("no trajectory files were resolved from batch_source")

    points = _normalize_time_points(time_points=time_points, time_range=time_range, n_time_points=n_time_points)
    species_ids, species_names = _resolve_species_selection(species, items[0].trajectory_path)

    extraction = extract_species_time_samples(
        items,
        species_ids=species_ids,
        species_names=species_names,
        time_points=points,
        temp_dir=sample_dir,
        outside=outside,
        print_memory=print_memory,
    )
    within_plots = plot_within_group_distributions(
        extraction,
        output_dir=out_dir / "within_group",
        dpi=dpi,
        print_memory=print_memory,
    )
    statistics_csv, statistics_json, ratio = compare_extracted_group_distributions(
        extraction,
        output_dir=out_dir,
        alpha=alpha,
        print_memory=print_memory,
    )
    comparison_plots = plot_group_comparison_box_scatter(
        extraction,
        output_dir=out_dir / "group_comparison",
        dpi=dpi,
        print_memory=print_memory,
    )
    return DistributionComparisonResult(
        extraction=extraction,
        within_group_plots=within_plots,
        comparison_plots=comparison_plots,
        statistics_csv=statistics_csv,
        statistics_json=statistics_json,
        non_significant_ratio=ratio,
    )


def resolve_batch_trajectory_items(
    batch_source: PathLike | Sequence[PathLike],
    *,
    group_key: str = "mode",
    groups: Sequence[str] | None = None,
    pattern: str = "*.npz",
    recursive: bool = False,
) -> list[BatchTrajectoryItem]:
    """Resolve batch metadata into trajectory items.

    Current supported inputs are paired metadata JSON, a folder of trajectory
    ``.npz`` files, a single ``.npz`` file, or a list of ``.npz`` files.  If a
    future batch format stores trajectories under another top-level key, update
    ``_metadata_rows(...)``.
    """

    wanted = None if groups is None else {str(group).lower() for group in groups}
    source_paths = _source_to_paths(batch_source, pattern=pattern, recursive=recursive)
    items: list[BatchTrajectoryItem] = []

    for source in source_paths:
        if source.suffix.lower() == ".json":
            rows = _metadata_rows(json.loads(source.read_text(encoding="utf-8")))
            for row_index, row in enumerate(rows):
                trajectory_path = _trajectory_path_from_row(row, source)
                if trajectory_path is None:
                    continue
                group = _group_from_row(row, group_key)
                if group is None and trajectory_path.exists():
                    group = _group_from_trajectory_metadata(trajectory_path, group_key)
                group = "unknown" if group is None else str(group).lower()
                if wanted is not None and group not in wanted:
                    continue
                items.append(
                    BatchTrajectoryItem(
                        index=len(items),
                        trajectory_path=trajectory_path,
                        group=group,
                        seed=_seed_from_row(row),
                        label=_label_from_row(row, row_index),
                    )
                )
            continue

        if source.suffix.lower() != ".npz":
            continue
        group = _group_from_trajectory_metadata(source, group_key) or "unknown"
        group = str(group).lower()
        if wanted is not None and group not in wanted:
            continue
        items.append(
            BatchTrajectoryItem(
                index=len(items),
                trajectory_path=source,
                group=group,
                seed=None,
                label=source.stem,
            )
        )

    print(f"[distribution] resolved {len(items)} trajectory item(s)")
    for item in items:
        print(f"  - group={item.group} seed={item.seed} path={item.trajectory_path}")
    return items


def extract_species_time_samples(
    items: Sequence[BatchTrajectoryItem],
    *,
    species_ids: Sequence[int],
    species_names: Sequence[str],
    time_points: Sequence[float],
    temp_dir: PathLike,
    outside: str = "nan",
    print_memory: bool = True,
) -> DistributionExtraction:
    """Extract selected species values into one small temp file per run.

    This is the memory-control step.  Full trajectory arrays are loaded one file
    at a time, sampled at requested time points, written to a compact ``.npz``,
    then released before the next trajectory is processed.
    """

    if outside not in {"nan", "hold"}:
        raise ValueError("outside must be 'nan' or 'hold'")
    output_dir = Path(temp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(list(time_points), dtype=float)
    species_ids_list = [int(sid) for sid in species_ids]
    species_names_list = [str(name) for name in species_names]

    sample_files: list[Path] = []
    grouped: dict[str, list[Path]] = {}
    manifest_rows = []

    for item in items:
        with np.load(item.trajectory_path, allow_pickle=True) as data:
            times = np.asarray(data["times"], dtype=float)
            states = np.asarray(data["states"], dtype=float)
            values = _sample_species_values(
                times,
                states,
                points,
                species_ids_list,
                outside=outside,
            )
            loaded_mb = _nbytes_mb(times, states)

        sample_path = output_dir / _sample_filename(item)
        np.savez_compressed(
            sample_path,
            group=np.asarray(item.group, dtype=object),
            seed=np.asarray(-1 if item.seed is None else int(item.seed), dtype=np.int64),
            source_path=np.asarray(str(item.trajectory_path), dtype=object),
            label=np.asarray(item.label, dtype=object),
            time_points=points,
            species_ids=np.asarray(species_ids_list, dtype=np.int64),
            species_names=np.asarray(species_names_list, dtype=object),
            values=np.asarray(values, dtype=float),
        )
        sample_files.append(sample_path)
        grouped.setdefault(item.group, []).append(sample_path)
        manifest_rows.append(
            {
                "index": int(item.index),
                "group": item.group,
                "seed": item.seed,
                "source_path": str(item.trajectory_path),
                "sample_path": str(sample_path),
            }
        )
        _print_memory(
            f"[memory] stored temp sample {sample_path.name}",
            print_memory=print_memory,
            loaded_mb=loaded_mb,
            arrays=[values],
        )

    manifest_path = output_dir / "samples_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": strftime("%Y-%m-%d %H:%M:%S"),
                "time_points": points.tolist(),
                "species_ids": species_ids_list,
                "species_names": species_names_list,
                "samples": manifest_rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[distribution] wrote sample manifest: {manifest_path}")
    return DistributionExtraction(
        temp_dir=output_dir,
        manifest_path=manifest_path,
        sample_files=sample_files,
        groups=grouped,
        species_ids=species_ids_list,
        species_names=species_names_list,
        time_points=points,
    )


def plot_within_group_distributions(
    extraction: DistributionExtraction,
    *,
    output_dir: PathLike,
    dpi: int = 150,
    print_memory: bool = True,
) -> list[Path]:
    """Plot same-group samples as distributions of the same random variable.

    For each ``group`` and ``species``, all run-level temp files in that group
    are treated as independent samples at each requested time point.  The output
    is one box-plus-scatter figure per group/species.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    for group, files in extraction.groups.items():
        for species_pos, species_name in enumerate(extraction.species_names):
            matrix = _load_group_matrix(files, species_pos)
            _print_memory(
                f"[memory] plotting within-group group={group} species={species_name}",
                print_memory=print_memory,
                arrays=[matrix],
            )
            fig, ax = plt.subplots(figsize=_distribution_figsize(extraction.time_points))
            _box_scatter_by_time(ax, matrix, extraction.time_points)
            ax.set_title(f"{group}: {species_name} distribution over time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            path = out_dir / f"{_safe_label(group)}_{_safe_label(species_name)}_distribution.png"
            fig.savefig(path, dpi=int(dpi))
            plt.close(fig)
            print(f"[distribution] wrote within-group plot: {path}")
            plot_paths.append(path)
    return plot_paths


def compare_extracted_group_distributions(
    extraction: DistributionExtraction,
    *,
    output_dir: PathLike,
    alpha: float = 0.05,
    print_memory: bool = True,
) -> tuple[Path | None, Path | None, float | None]:
    """Run Wilcoxon rank-sum style comparisons between groups.

    The comparison is independent-sample rank-sum by default.  If SciPy is
    installed, ``scipy.stats.ranksums`` is used.  Otherwise a normal
    approximation with average ranks is used.  If future experiments are paired
    by seed and need signed-rank tests, add that branch here.
    """

    groups = sorted(extraction.groups)
    if len(groups) < 2:
        print("[distribution] skip group comparison: fewer than two groups")
        return None, None, None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    valid_tests = 0
    nonsignificant = 0

    for species_pos, species_name in enumerate(extraction.species_names):
        matrices = {
            group: _load_group_matrix(extraction.groups[group], species_pos)
            for group in groups
        }
        _print_memory(
            f"[memory] rank-sum comparison species={species_name}",
            print_memory=print_memory,
            arrays=list(matrices.values()),
        )
        for time_index, time_value in enumerate(extraction.time_points):
            for group_a, group_b in combinations(groups, 2):
                values_a = _finite_column(matrices[group_a], time_index)
                values_b = _finite_column(matrices[group_b], time_index)
                statistic, p_value, test_name = _rank_sum_test(values_a, values_b)
                is_valid = bool(np.isfinite(p_value))
                if is_valid:
                    valid_tests += 1
                    if p_value >= float(alpha):
                        nonsignificant += 1
                rows.append(
                    {
                        "species_id": int(extraction.species_ids[species_pos]),
                        "species_name": species_name,
                        "time": float(time_value),
                        "group_a": group_a,
                        "group_b": group_b,
                        "n_a": int(values_a.size),
                        "n_b": int(values_b.size),
                        "test": test_name,
                        "statistic": statistic,
                        "p_value": p_value,
                        "alpha": float(alpha),
                        "significant": bool(is_valid and p_value < float(alpha)),
                    }
                )

    ratio = None if valid_tests == 0 else float(nonsignificant / valid_tests)
    csv_path = out_dir / "distribution_rank_sum_statistics.csv"
    json_path = out_dir / "distribution_rank_sum_summary.json"
    _write_statistics_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "alpha": float(alpha),
                "valid_tests": int(valid_tests),
                "nonsignificant_tests": int(nonsignificant),
                "non_significant_ratio": ratio,
                "rows": rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[distribution] wrote statistics csv: {csv_path}")
    print(f"[distribution] wrote statistics json: {json_path}")
    print(f"[distribution] non-significant ratio: {ratio}")
    return csv_path, json_path, ratio


def plot_group_comparison_box_scatter(
    extraction: DistributionExtraction,
    *,
    output_dir: PathLike,
    dpi: int = 150,
    print_memory: bool = True,
) -> list[Path]:
    """Plot box plus scatter comparisons for the same species across groups.

    Each species gets one figure.  For every time point, each group contributes
    one box and jittered scatter points.  The source arrays are the small temp
    sample files, not the original full trajectories.
    """

    groups = sorted(extraction.groups)
    if len(groups) < 2:
        print("[distribution] skip comparison plots: fewer than two groups")
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    for species_pos, species_name in enumerate(extraction.species_names):
        matrices = {
            group: _load_group_matrix(extraction.groups[group], species_pos)
            for group in groups
        }
        _print_memory(
            f"[memory] plotting group comparison species={species_name}",
            print_memory=print_memory,
            arrays=list(matrices.values()),
        )
        fig, ax = plt.subplots(figsize=_comparison_figsize(extraction.time_points, groups))
        _comparison_box_scatter(ax, matrices, extraction.time_points, groups)
        ax.set_title(f"Group comparison: {species_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        path = out_dir / f"{_safe_label(species_name)}_group_comparison.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        print(f"[distribution] wrote comparison plot: {path}")
        plot_paths.append(path)
    return plot_paths


def _source_to_paths(
    batch_source: PathLike | Sequence[PathLike],
    *,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    if isinstance(batch_source, (str, Path)):
        path = Path(batch_source)
        if path.is_dir():
            iterator = path.rglob(pattern) if recursive else path.glob(pattern)
            return sorted(item for item in iterator if item.is_file())
        return [path]
    return [Path(item) for item in batch_source]


def _metadata_rows(raw: object) -> list[dict]:
    if isinstance(raw, dict):
        for key in ("runs", "trajectories", "records", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
        return [dict(raw)]
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    raise ValueError("metadata JSON must contain a dict or a list")


def _trajectory_path_from_row(row: dict, source_json: Path) -> Path | None:
    raw = (
        row.get("trajectory_path")
        or row.get("path")
        or row.get("file")
        or row.get("filename")
        or row.get("output_path")
    )
    if raw is None and isinstance(row.get("metadata"), dict):
        raw = row["metadata"].get("trajectory_path")
    if raw is None:
        return None
    return _resolve_trajectory_path(raw, source_json)


def _resolve_trajectory_path(raw_path: object, source_json: Path) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        return (source_json.parent / path).resolve()
    if path.exists():
        return path
    candidates = [
        source_json.parent / path.name,
        source_json.parent / "trajectories" / path.name,
        source_json.parent.parent / "trajectories" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def _group_from_row(row: dict, group_key: str) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = row.get(group_key)
    if value is None:
        value = row.get("stepper_method") or row.get("mode")
    if value is None:
        value = metadata.get(group_key) or metadata.get("stepper_method") or metadata.get("mode")
    return None if value is None else str(value)


def _seed_from_row(row: dict) -> int | None:
    value = row.get("seed", row.get("pair_seed"))
    return None if value is None else int(value)


def _label_from_row(row: dict, index: int) -> str:
    if "pair_order" in row:
        return f"pair={int(row['pair_order'])}"
    if "run_index" in row:
        return f"run={int(row['run_index'])}"
    return f"index={index}"


def _trajectory_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        if "run_metadata_json" not in data:
            return {}
        return json.loads(str(data["run_metadata_json"]))


def _group_from_trajectory_metadata(path: Path, group_key: str) -> str | None:
    metadata = _trajectory_metadata(path)
    value = metadata.get(group_key) or metadata.get("stepper_method") or metadata.get("mode")
    return None if value is None else str(value)


def _trajectory_species_names(path: Path) -> list[str]:
    with np.load(path, allow_pickle=True) as data:
        return [str(name) for name in data["species_names"].tolist()]


def _resolve_species_selection(
    species: int | str | Sequence[int | str],
    trajectory_path: Path,
) -> tuple[list[int], list[str]]:
    names = _trajectory_species_names(trajectory_path)
    requested = [species] if isinstance(species, (int, str)) else list(species)
    ids: list[int] = []
    labels: list[str] = []
    for item in requested:
        if isinstance(item, str) and not item.isdigit():
            if item not in names:
                raise KeyError(f"species name not found in trajectory: {item}")
            sid = int(names.index(item))
        else:
            sid = int(item)
        if sid < 0 or sid >= len(names):
            raise IndexError(f"species id out of range: {sid}")
        ids.append(sid)
        labels.append(names[sid])
    return ids, labels


def _normalize_time_points(
    *,
    time_points: Iterable[float] | None,
    time_range: tuple[float, float] | None,
    n_time_points: int,
) -> np.ndarray:
    if time_points is not None:
        points = np.asarray(list(time_points), dtype=float)
    else:
        if time_range is None:
            raise ValueError("provide either time_points or time_range")
        start, end = float(time_range[0]), float(time_range[1])
        if int(n_time_points) <= 0:
            raise ValueError("n_time_points must be > 0")
        points = np.linspace(start, end, int(n_time_points), dtype=float)
    if points.ndim != 1 or points.size == 0:
        raise ValueError("time points must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(points)):
        raise ValueError("time points must be finite")
    return points


def _sample_species_values(
    times: np.ndarray,
    states: np.ndarray,
    time_points: np.ndarray,
    species_ids: Sequence[int],
    *,
    outside: str,
) -> np.ndarray:
    if times.ndim != 1 or states.ndim != 2 or states.shape[0] != times.shape[0]:
        raise ValueError("invalid trajectory arrays")
    if times.size == 0:
        return np.full((time_points.size, len(species_ids)), np.nan, dtype=float)

    indices = np.searchsorted(times, time_points, side="right") - 1
    outside_mask = (time_points < times[0]) | (time_points > times[-1])
    indices = np.clip(indices, 0, times.size - 1)
    values = states[indices[:, np.newaxis], np.asarray(species_ids, dtype=np.int64)[np.newaxis, :]]
    values = np.asarray(values, dtype=float)
    if outside == "nan":
        values[outside_mask, :] = np.nan
    return values


def _sample_filename(item: BatchTrajectoryItem) -> str:
    seed_text = "none" if item.seed is None else str(item.seed)
    return f"sample_{item.index:04d}_{_safe_label(item.group)}_seed_{seed_text}_{_safe_label(item.trajectory_path.stem)}.npz"


def _load_group_matrix(files: Sequence[Path], species_pos: int) -> np.ndarray:
    rows = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            values = np.asarray(data["values"], dtype=float)
        rows.append(values[:, int(species_pos)])
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.vstack(rows)


def _finite_column(matrix: np.ndarray, time_index: int) -> np.ndarray:
    if matrix.size == 0:
        return np.empty(0, dtype=float)
    values = np.asarray(matrix[:, int(time_index)], dtype=float)
    return values[np.isfinite(values)]


def _rank_sum_test(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float, str]:
    x = np.asarray(values_a, dtype=float)
    y = np.asarray(values_b, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan"), "rank_sum"
    try:
        from scipy.stats import ranksums

        result = ranksums(x, y)
        return float(result.statistic), float(result.pvalue), "scipy.stats.ranksums"
    except Exception:
        statistic, p_value = _rank_sum_normal_approximation(x, y)
        return statistic, p_value, "normal_approx_rank_sum"


def _rank_sum_normal_approximation(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values_a, dtype=float)
    y = np.asarray(values_b, dtype=float)
    combined = np.concatenate([x, y])
    ranks = _average_ranks(combined)
    n1 = x.size
    n2 = y.size
    rank_sum_1 = float(np.sum(ranks[:n1]))
    expected = n1 * (n1 + n2 + 1) / 2.0
    variance = n1 * n2 * (n1 + n2 + 1) / 12.0
    if variance <= 0.0:
        return float("nan"), float("nan")
    z = (rank_sum_1 - expected) / math.sqrt(variance)
    p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z)))
    return float(z), float(p_value)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _write_statistics_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _box_scatter_by_time(ax, matrix: np.ndarray, time_points: np.ndarray) -> None:
    positions = np.arange(len(time_points), dtype=float)
    data = [_finite_column(matrix, idx) for idx in range(len(time_points))]
    _safe_boxplot(ax, data, positions, width=0.45, color="tab:blue")
    rng = np.random.default_rng(12345)
    for position, values in zip(positions, data):
        if values.size == 0:
            continue
        jitter = rng.normal(0.0, 0.035, size=values.size)
        ax.scatter(np.full(values.size, position) + jitter, values, s=15, alpha=0.65, color="black")
    ax.set_xticks(positions)
    ax.set_xticklabels([_format_time(value) for value in time_points], rotation=45, ha="right")


def _comparison_box_scatter(
    ax,
    matrices: dict[str, np.ndarray],
    time_points: np.ndarray,
    groups: Sequence[str],
) -> None:
    n_groups = len(groups)
    base_positions = np.arange(len(time_points), dtype=float)
    offsets = np.linspace(-0.32, 0.32, n_groups) if n_groups > 1 else np.asarray([0.0])
    width = min(0.22, 0.7 / max(n_groups, 1))
    rng = np.random.default_rng(12345)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_groups, 1)))
    for group_index, group in enumerate(groups):
        matrix = matrices[group]
        positions = base_positions + offsets[group_index]
        data = [_finite_column(matrix, idx) for idx in range(len(time_points))]
        _safe_boxplot(ax, data, positions, width=width, color=colors[group_index], label=group)
        for position, values in zip(positions, data):
            if values.size == 0:
                continue
            jitter = rng.normal(0.0, width * 0.15, size=values.size)
            ax.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=12,
                alpha=0.55,
                color=colors[group_index],
            )
    ax.set_xticks(base_positions)
    ax.set_xticklabels([_format_time(value) for value in time_points], rotation=45, ha="right")


def _safe_boxplot(ax, data: Sequence[np.ndarray], positions: np.ndarray, *, width: float, color, label: str | None = None) -> None:
    valid_data = []
    valid_positions = []
    for values, position in zip(data, positions):
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size:
            valid_data.append(clean)
            valid_positions.append(float(position))
    if not valid_data:
        return
    result = ax.boxplot(
        valid_data,
        positions=valid_positions,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    for patch in result["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for key in ("whiskers", "caps", "medians"):
        for artist in result[key]:
            artist.set_color(color)
    if label is not None:
        ax.plot([], [], color=color, label=label)


def _distribution_figsize(time_points: np.ndarray) -> tuple[float, float]:
    return (max(8.0, min(18.0, 0.75 * len(time_points))), 5.0)


def _comparison_figsize(time_points: np.ndarray, groups: Sequence[str]) -> tuple[float, float]:
    return (max(9.0, min(22.0, 0.85 * len(time_points) * max(1, len(groups) / 2))), 5.5)


def _format_time(value: float) -> str:
    return f"{float(value):.4g}"


def _default_output_dir(batch_source: PathLike | Sequence[PathLike], output_dir: PathLike | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if isinstance(batch_source, (str, Path)):
        path = Path(batch_source)
        if path.is_dir():
            return path / "distribution_comparison"
        return path.parent / f"{path.stem}_distribution_comparison"
    return Path("distribution_comparison")


def _safe_label(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "item"


def _nbytes_mb(*arrays: np.ndarray) -> float:
    return float(sum(np.asarray(array).nbytes for array in arrays) / (1024.0 * 1024.0))


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        return None


def _print_memory(
    message: str,
    *,
    print_memory: bool,
    loaded_mb: float = 0.0,
    arrays: Sequence[np.ndarray] = (),
) -> None:
    if not print_memory:
        return
    array_mb = _nbytes_mb(*arrays) if arrays else 0.0
    rss = _rss_mb()
    rss_text = "rss=unavailable" if rss is None else f"rss={rss:.2f} MB"
    print(f"{message}: loaded_arrays={loaded_mb:.2f} MB temp_arrays={array_mb:.2f} MB {rss_text}")

