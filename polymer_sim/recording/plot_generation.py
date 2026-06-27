"""Batch plot generation and filtering for saved trajectory files.

Current processing content:
- input: a folder containing `.npz` trajectory files saved by
  `save_trajectory_record(...)`, or a metadata JSON containing trajectory paths;
- output: one PNG per trajectory, or one vertically stacked PNG per selected group;
- current plot type: all-species trajectory time series only;
- filter 1: split trajectories by whether non-food species at the final time
  cross a threshold;
- filter 2: split trajectories by `stepper_method`/`mode` from trajectory JSON
  metadata.

Run directly:

    python polymer_sim/recording/plot_generation.py examples/paired_method_outputs/trajectories

Grouped plots:

    python polymer_sim/recording/plot_generation.py examples/paired_method_outputs/trajectories --grouped --threshold 100 --methods ssa blended

By default images are written to `<input_folder>/plots`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polymer_sim.recording.plot_single_run import plot_time_series
from polymer_sim.recording.trajectory import load_trajectory_record


@dataclass(slots=True)
class GeneratedPlot:
    trajectory_path: Path
    output_path: Path
    plot_type: str


@dataclass(slots=True)
class GeneratedGroupPlot:
    group_label: str
    trajectory_paths: list[Path]
    output_path: Path
    plot_type: str


def filter_trajectories_by_final_threshold(
    folder: str | Path,
    threshold: float,
    *,
    nonfood_start_species_id: int = 3,
    pattern: str = "*.npz",
    recursive: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Split trajectory files by final-time non-food abundance.

    A trajectory is placed in the first returned list when any species with
    `species_id >= nonfood_start_species_id` has final count larger than
    `threshold`.  The second list contains the remaining files.  With the
    default `nonfood_start_species_id=3`, species ids 0, 1, and 2 are treated as
    food and ignored by this filter.
    """

    paths = _trajectory_paths_from_folder(folder, pattern=pattern, recursive=recursive)
    return _split_paths_by_final_threshold(
        paths,
        threshold=float(threshold),
        nonfood_start_species_id=int(nonfood_start_species_id),
    )


def filter_trajectories_by_stepper_method(
    trajectory_json: str | Path,
    methods: Sequence[str] | None = None,
) -> dict[str, list[Path]]:
    """Read a trajectory metadata JSON and group trajectory paths by method.

    Supported JSON layouts:
    - paired metadata from `examples/multiple_run.py`, with top-level `runs`;
    - a plain list of run dictionaries;
    - a dictionary with `trajectories`, `records`, or `items`.

    The method label is read from `stepper_method` first, then `mode`, then
    `metadata.stepper_method`.  If the JSON row has no method but its trajectory
    file exists, the function falls back to the trajectory `.npz` run metadata.
    """

    json_path = Path(trajectory_json)
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    rows = _metadata_rows(raw)
    wanted = None if methods is None else {str(method).lower() for method in methods}
    grouped: dict[str, list[Path]] = {}
    if wanted is not None:
        grouped = {str(method).lower(): [] for method in methods}

    for row in rows:
        path = _trajectory_path_from_metadata_row(row, json_path)
        if path is None:
            continue
        method = _method_from_metadata_row(row)
        if method is None and path.exists():
            method = _method_from_record(load_trajectory_record(path))
        if method is None:
            method = "unknown"
        key = str(method).lower()
        if wanted is not None and key not in wanted:
            continue
        grouped.setdefault(key, []).append(path)
    return grouped


def batch_plot_species_trajectories(
    folder: str | Path,
    *,
    output_dir: str | Path | None = None,
    plot_type: str = "all_species",
    pattern: str = "*.npz",
    recursive: bool = False,
    dpi: int = 150,
    overwrite: bool = True,
) -> list[GeneratedPlot]:
    """Batch-generate all-species trajectory plots for a folder.

    Parameters
    ----------
    folder:
        Directory containing trajectory `.npz` files.
    output_dir:
        Directory for PNG outputs.  Defaults to `<folder>/plots`.
    plot_type:
        Plot type to generate.  Currently only "all_species" is supported.
    pattern:
        Filename pattern used to find trajectory files.
    recursive:
        If True, search subdirectories recursively.
    dpi:
        Output PNG resolution.
    overwrite:
        If False, skip plots whose output files already exist.
    """

    input_dir = Path(folder)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if plot_type != "all_species":
        raise ValueError("plot_type must be 'all_species'; no other batch plot types are implemented yet")

    out_dir = Path(output_dir) if output_dir is not None else input_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory_paths = _trajectory_paths_from_folder(input_dir, pattern=pattern, recursive=recursive)
    generated: list[GeneratedPlot] = []

    for trajectory_path in trajectory_paths:
        if not trajectory_path.is_file():
            continue
        if out_dir in trajectory_path.parents:
            continue

        output_path = _output_path_for(trajectory_path, input_dir, out_dir)
        if output_path.exists() and not overwrite:
            generated.append(
                GeneratedPlot(
                    trajectory_path=trajectory_path,
                    output_path=output_path,
                    plot_type="all_species_time_series",
                )
            )
            continue

        record = load_trajectory_record(trajectory_path)
        fig, _ax = plot_time_series(
            record,
            title=f"All Species Trajectory: {trajectory_path.stem}",
        )
        fig.set_size_inches(12, 6)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=int(dpi))
        plt.close(fig)

        generated.append(
            GeneratedPlot(
                trajectory_path=trajectory_path,
                output_path=output_path,
                plot_type="all_species_time_series",
            )
        )

    return generated


def batch_plot_grouped_species_trajectories(
    folder: str | Path | None = None,
    *,
    trajectory_json: str | Path | None = None,
    trajectory_paths: Sequence[str | Path] | None = None,
    output_dir: str | Path | None = None,
    method_labels: Sequence[str] | None = None,
    threshold: float | None = None,
    threshold_labels: Sequence[str | Sequence[str]] | None = None,
    nonfood_start_species_id: int = 3,
    pattern: str = "*.npz",
    recursive: bool = False,
    dpi: int = 150,
    overwrite: bool = True,
    species_indices: list[int] | np.ndarray | None = None,
    time_range: tuple[float | None, float | None] | None = None,
    verbose: bool = False,
) -> list[GeneratedGroupPlot]:
    """Create vertically stacked all-species plots for selected trajectory groups.

    Grouping order:
    1. if `method_labels` is provided, split records by method first;
    2. if `threshold` is provided, split each method group into threshold-fit
       and threshold-unfit groups.

    Each output figure contains one subplot per trajectory in that final group.
    All subplots share the same real simulation-time axis; record times are not
    rescaled or stretched, so shorter runs simply end earlier.
    """

    if verbose:
        _print_grouped_plot_parameters(
            folder=folder,
            trajectory_json=trajectory_json,
            output_dir=output_dir,
            method_labels=method_labels,
            threshold=threshold,
            threshold_labels=threshold_labels,
            nonfood_start_species_id=nonfood_start_species_id,
            pattern=pattern,
            recursive=recursive,
            dpi=dpi,
            overwrite=overwrite,
        )

    paths = _resolve_input_trajectory_paths(
        folder=folder,
        trajectory_json=trajectory_json,
        trajectory_paths=trajectory_paths,
        pattern=pattern,
        recursive=recursive,
    )
    if verbose:
        _print_input_structure_report(
            folder=folder,
            trajectory_json=trajectory_json,
            paths=paths,
            pattern=pattern,
            recursive=recursive,
        )
    if not paths:
        return []

    out_dir = _default_group_output_dir(folder, trajectory_json, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_groups = _method_groups(paths, trajectory_json=trajectory_json, method_labels=method_labels)
    if verbose:
        _print_path_groups("method filter result", base_groups)
    fit_label, unfit_label = _normalize_threshold_labels(threshold_labels)

    final_groups: list[tuple[str, list[Path]]] = []
    for method_label, group_paths in base_groups:
        if threshold is None:
            final_groups.append((method_label, group_paths))
            continue
        fit_paths, unfit_paths = _split_paths_by_final_threshold(
            group_paths,
            threshold=float(threshold),
            nonfood_start_species_id=int(nonfood_start_species_id),
        )
        if verbose:
            _print_path_groups(
                f"threshold filter result for {method_label}",
                [(fit_label, fit_paths), (unfit_label, unfit_paths)],
            )
        final_groups.append((f"{method_label}_{fit_label}", fit_paths))
        final_groups.append((f"{method_label}_{unfit_label}", unfit_paths))
    if verbose:
        _print_path_groups("final plotting groups", final_groups)

    generated: list[GeneratedGroupPlot] = []
    for group_label, group_paths in final_groups:
        if not group_paths:
            if verbose:
                print(f"[plot] skip empty group: {group_label}")
            continue
        output_path = out_dir / f"{_safe_file_label(group_label)}_all_species_group.png"
        if output_path.exists() and not overwrite:
            if verbose:
                print(f"[plot] skip existing group: {group_label} -> {output_path}")
            generated.append(
                GeneratedGroupPlot(
                    group_label=group_label,
                    trajectory_paths=list(group_paths),
                    output_path=output_path,
                    plot_type="grouped_all_species_time_series",
                )
            )
            continue

        if verbose:
            print(f"[plot] start group: {group_label}")
            print(f"[plot] output: {output_path}")
            print(f"[plot] records: {len(group_paths)}")
        records = [(path, load_trajectory_record(path)) for path in group_paths]
        fig = _plot_records_vertically(
            records,
            title=f"All Species Trajectories: {group_label}",
            species_indices=species_indices,
            time_range=time_range,
            verbose=verbose,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=int(dpi))
        plt.close(fig)
        if verbose:
            print(f"[plot] completed group: {group_label} -> {output_path}")
        generated.append(
            GeneratedGroupPlot(
                group_label=group_label,
                trajectory_paths=list(group_paths),
                output_path=output_path,
                plot_type="grouped_all_species_time_series",
            )
        )

    return generated


def _output_path_for(trajectory_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = trajectory_path.relative_to(input_dir)
    if relative.parent == Path("."):
        return output_dir / f"{trajectory_path.stem}_all_species.png"
    return output_dir / relative.parent / f"{trajectory_path.stem}_all_species.png"


def _trajectory_paths_from_folder(
    folder: str | Path,
    *,
    pattern: str = "*.npz",
    recursive: bool = False,
) -> list[Path]:
    input_dir = Path(folder)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    return sorted(path for path in (input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)) if path.is_file())


def _resolve_input_trajectory_paths(
    *,
    folder: str | Path | None,
    trajectory_json: str | Path | None,
    trajectory_paths: Sequence[str | Path] | None,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    if trajectory_paths is not None:
        return [Path(path) for path in trajectory_paths]
    if trajectory_json is not None:
        grouped = filter_trajectories_by_stepper_method(trajectory_json)
        paths: list[Path] = []
        for method_paths in grouped.values():
            paths.extend(method_paths)
        return paths
    if folder is None:
        raise ValueError("folder, trajectory_json, or trajectory_paths must be provided")
    if not Path(folder).is_dir():
        return []
    return _trajectory_paths_from_folder(folder, pattern=pattern, recursive=recursive)


def _split_paths_by_final_threshold(
    paths: Sequence[Path],
    *,
    threshold: float,
    nonfood_start_species_id: int,
) -> tuple[list[Path], list[Path]]:
    if nonfood_start_species_id < 0:
        raise ValueError("nonfood_start_species_id must be >= 0")
    fit: list[Path] = []
    unfit: list[Path] = []
    for path in paths:
        record = load_trajectory_record(path)
        target = fit if _final_nonfood_crosses_threshold(record, threshold, nonfood_start_species_id) else unfit
        target.append(Path(path))
    return fit, unfit


def _final_nonfood_crosses_threshold(record, threshold: float, nonfood_start_species_id: int) -> bool:
    if record.states.shape[0] == 0 or record.states.shape[1] <= nonfood_start_species_id:
        return False
    final_state = np.asarray(record.states[-1, nonfood_start_species_id:], dtype=float)
    return bool(np.any(final_state > float(threshold)))


def _metadata_rows(raw: object) -> list[dict]:
    if isinstance(raw, dict):
        for key in ("runs", "trajectories", "records", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
        return [dict(raw)]
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    raise ValueError("trajectory JSON must contain a dict or a list")


def _trajectory_path_from_metadata_row(row: dict, json_path: Path) -> Path | None:
    raw_path = (
        row.get("trajectory_path")
        or row.get("path")
        or row.get("file")
        or row.get("filename")
        or row.get("output_path")
    )
    if raw_path is None and isinstance(row.get("metadata"), dict):
        raw_path = row["metadata"].get("trajectory_path")
    if raw_path is None:
        return None
    return _resolve_metadata_path(raw_path, json_path)


def _resolve_metadata_path(raw_path: object, json_path: Path) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        return (json_path.parent / path).resolve()
    if path.exists():
        return path
    candidates = [
        json_path.parent / path.name,
        json_path.parent / "trajectories" / path.name,
        json_path.parent.parent / "trajectories" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def _method_from_metadata_row(row: dict) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = row.get("stepper_method") or row.get("mode") or metadata.get("stepper_method") or metadata.get("mode")
    return None if value is None else str(value).lower()


def _method_from_record(record) -> str | None:
    value = record.run_metadata.get("stepper_method") or record.run_metadata.get("mode")
    return None if value is None else str(value).lower()


def _method_groups(
    paths: Sequence[Path],
    *,
    trajectory_json: str | Path | None,
    method_labels: Sequence[str] | None,
) -> list[tuple[str, list[Path]]]:
    if method_labels is None:
        return [("all", list(paths))]
    labels = [str(method).lower() for method in method_labels]
    groups = {label: [] for label in labels}

    if trajectory_json is not None:
        by_method = filter_trajectories_by_stepper_method(trajectory_json, methods=labels)
        for label in labels:
            groups[label].extend(by_method.get(label, []))
    else:
        for path in paths:
            record = load_trajectory_record(path)
            method = _method_from_record(record)
            if method in groups:
                groups[method].append(Path(path))
    return [(label, groups[label]) for label in labels]


def _normalize_threshold_labels(labels: Sequence[str | Sequence[str]] | None) -> tuple[str, str]:
    if labels is None:
        return "fit_threshold", "unfit_threshold"
    values: Sequence[str | Sequence[str]] = labels
    if len(values) == 1 and isinstance(values[0], Sequence) and not isinstance(values[0], str):
        values = values[0]  # accept [["fit", "unfit"]] as a convenience
    if len(values) != 2:
        raise ValueError("threshold_labels must contain exactly two labels")
    return str(values[0]), str(values[1])


def _print_grouped_plot_parameters(
    *,
    folder: str | Path | None,
    trajectory_json: str | Path | None,
    output_dir: str | Path | None,
    method_labels: Sequence[str] | None,
    threshold: float | None,
    threshold_labels: Sequence[str | Sequence[str]] | None,
    nonfood_start_species_id: int,
    pattern: str,
    recursive: bool,
    dpi: int,
    overwrite: bool,
) -> None:
    print("[parameters]")
    print(f"  target folder: {folder}")
    print(f"  trajectory_json: {trajectory_json}")
    print(f"  output_dir: {output_dir}")
    print(f"  methods: {None if method_labels is None else list(method_labels)}")
    print(f"  threshold: {threshold}")
    print(f"  threshold_labels: {threshold_labels}")
    print(f"  nonfood_start_species_id: {nonfood_start_species_id}")
    print(f"  pattern: {pattern}")
    print(f"  recursive: {recursive}")
    print(f"  dpi: {dpi}")
    print(f"  overwrite: {overwrite}")


def _print_input_structure_report(
    *,
    folder: str | Path | None,
    trajectory_json: str | Path | None,
    paths: Sequence[Path],
    pattern: str,
    recursive: bool,
) -> None:
    print("[input structure]")
    if folder is not None:
        folder_path = Path(folder)
        print(f"  target path: {folder_path}")
        print(f"  resolved path: {folder_path.resolve()}")
        print(f"  exists: {folder_path.exists()}")
        print(f"  is_dir: {folder_path.is_dir()}")
    if trajectory_json is not None:
        json_path = Path(trajectory_json)
        print(f"  metadata json: {json_path}")
        print(f"  metadata json exists: {json_path.exists()}")
    print(f"  search pattern: {pattern}")
    print(f"  recursive: {recursive}")
    print(f"  matched trajectory files: {len(paths)}")
    structure_ok = bool(paths) and all(Path(path).suffix == ".npz" for path in paths)
    print(f"  structure normal: {'YES' if structure_ok else 'NO'}")
    if not structure_ok:
        print("  expected: a folder or metadata JSON that resolves to one or more .npz trajectory files")
    _print_path_list("matched trajectory file list", paths)


def _print_path_groups(title: str, groups: Sequence[tuple[str, Sequence[Path]]]) -> None:
    print(f"[{title}]")
    for label, paths in groups:
        print(f"  {label}: {len(paths)} file(s)")
        for path in paths:
            print(f"    - {path}")


def _print_path_list(title: str, paths: Sequence[Path]) -> None:
    print(f"[{title}]")
    if not paths:
        print("  <empty>")
        return
    for path in paths:
        print(f"  - {path}")


def _plot_records_vertically(
    records: Sequence[tuple[Path, object]],
    *,
    title: str,
    species_indices: list[int] | np.ndarray | None,
    time_range: tuple[float | None, float | None] | None,
    verbose: bool = False,
):
    n_rows = len(records)
    height = max(3.0, min(3.0 * n_rows, 36.0))
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, height), sharex=True, squeeze=False)
    axes_flat = axes[:, 0]

    x_min: float | None = None
    x_max: float | None = None
    for ax, (path, record) in zip(axes_flat, records):
        times, states = _record_time_window(record, time_range)
        if times.size:
            x_min = float(times[0]) if x_min is None else min(x_min, float(times[0]))
            x_max = float(times[-1]) if x_max is None else max(x_max, float(times[-1]))
        if verbose:
            print(f"[plot] draw record: {path} ({times.size} time points, {states.shape[1]} species)")
        _plot_record_time_series_on_axis(ax, record, times, states, species_indices=species_indices)
        method = _method_from_record(record) or "unknown"
        ax.set_title(f"{path.name} | method={method}", fontsize=9)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    if x_min is not None and x_max is not None and x_max > x_min:
        for ax in axes_flat:
            ax.set_xlim(x_min, x_max)
    axes_flat[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _plot_record_time_series_on_axis(
    ax,
    record,
    times: np.ndarray,
    states: np.ndarray,
    *,
    species_indices: list[int] | np.ndarray | None,
) -> None:
    if species_indices is None:
        indices = np.arange(states.shape[1], dtype=np.int64)
    else:
        indices = np.asarray(species_indices, dtype=np.int64)
    for sid in indices:
        ax.plot(times, states[:, int(sid)], linewidth=0.8, alpha=0.75)


def _record_time_window(record, time_range: tuple[float | None, float | None] | None) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(record.times, dtype=float)
    states = np.asarray(record.states, dtype=float)
    if time_range is None:
        return times, states
    start, end = time_range
    mask = np.ones(times.shape, dtype=bool)
    if start is not None:
        mask &= times >= float(start)
    if end is not None:
        mask &= times <= float(end)
    return times[mask], states[mask, :]


def _default_group_output_dir(
    folder: str | Path | None,
    trajectory_json: str | Path | None,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if folder is not None:
        return Path(folder) / "plots"
    if trajectory_json is not None:
        return Path(trajectory_json).parent / "plots"
    return Path("plots")


def _safe_file_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label)).strip("_") or "group"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-generate all-species trajectory plots for trajectory npz files.",
    )
    parser.add_argument("folder", nargs="?", default=None, help="Folder containing trajectory .npz files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output folder for generated PNG files. Defaults to <folder>/plots.",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="File pattern to match. Defaults to *.npz.",
    )
    parser.add_argument(
        "--plot-type",
        default="all_species",
        choices=["all_species"],
        help="Plot type to generate. Currently only all_species is implemented.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for trajectory files recursively.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI. Defaults to 150.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip existing output files.",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Create grouped vertical plots instead of one PNG per trajectory.",
    )
    parser.add_argument(
        "--trajectory-json",
        default=None,
        help="Metadata JSON containing trajectory paths and stepper_method/mode fields.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Stepper methods to include in grouped plots, e.g. --methods ssa blended.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Final-time non-food threshold used to split trajectories in grouped plots.",
    )
    parser.add_argument(
        "--threshold-labels",
        nargs=2,
        default=None,
        metavar=("FIT_LABEL", "UNFIT_LABEL"),
        help="Labels for threshold-fit and threshold-unfit groups.",
    )
    parser.add_argument(
        "--nonfood-start-species-id",
        type=int,
        default=3,
        help="First non-food species id. Defaults to 3, meaning species ids 0, 1, 2 are food.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.grouped:
        generated_groups = batch_plot_grouped_species_trajectories(
            args.folder,
            trajectory_json=args.trajectory_json,
            output_dir=args.output_dir,
            method_labels=args.methods,
            threshold=args.threshold,
            threshold_labels=args.threshold_labels,
            nonfood_start_species_id=int(args.nonfood_start_species_id),
            pattern=args.pattern,
            recursive=bool(args.recursive),
            dpi=int(args.dpi),
            overwrite=not bool(args.no_overwrite),
            verbose=True,
        )
        print(f"generated {len(generated_groups)} grouped plot(s)")
        for item in generated_groups:
            print(f"{item.group_label}: {len(item.trajectory_paths)} trajectory file(s) -> {item.output_path}")
        return 0

    if args.folder is None:
        raise SystemExit("folder is required unless --grouped uses --trajectory-json")
    generated = batch_plot_species_trajectories(
        args.folder,
        output_dir=args.output_dir,
        plot_type=args.plot_type,
        pattern=args.pattern,
        recursive=bool(args.recursive),
        dpi=int(args.dpi),
        overwrite=not bool(args.no_overwrite),
    )
    print(f"generated {len(generated)} plot(s)")
    for item in generated:
        print(f"{item.trajectory_path} -> {item.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
