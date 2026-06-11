"""Batch plot generation for saved trajectory files.

Current processing content:
- input: a folder containing `.npz` trajectory files saved by
  `save_trajectory_record(...)`;
- output: one PNG per trajectory;
- current plot type: all-species trajectory time series only.

Run directly:

    python polymer_sim/recording/plot_generation.py examples/paired_method_outputs/trajectories

By default images are written to `<input_folder>/plots`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from polymer_sim.recording.plot_single_run import plot_time_series
from polymer_sim.recording.trajectory import load_trajectory_record


@dataclass(slots=True)
class GeneratedPlot:
    trajectory_path: Path
    output_path: Path
    plot_type: str


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

    trajectory_paths = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
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


def _output_path_for(trajectory_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = trajectory_path.relative_to(input_dir)
    if relative.parent == Path("."):
        return output_dir / f"{trajectory_path.stem}_all_species.png"
    return output_dir / relative.parent / f"{trajectory_path.stem}_all_species.png"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-generate all-species trajectory plots for trajectory npz files.",
    )
    parser.add_argument("folder", help="Folder containing trajectory .npz files.")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
