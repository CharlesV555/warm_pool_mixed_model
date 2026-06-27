from __future__ import annotations

"""Lightweight analysis entry for existing batch trajectory data.

This script does not run new simulations.  It reads an existing batch metadata
JSON written by examples/multiple_run.py or examples/multiple_run_core.py, then
compares selected species distributions across methods.

Edit the configuration block below, then run:

    python examples/test.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_sim.recording.distribution_comparison import compare_species_distributions


# Existing batch data.  This should point to metadata that contains a top-level
# "runs" list with trajectory_path fields.
METADATA_PATH = EXAMPLES_DIR / "method_run_outputs" / "method_run_metadata.json"

# Methods to compare.  For a single-method batch, use one label, for example:
# GROUPS = ("ssa",)
GROUPS = ("ssa", "blended")

# Species can be ids or names.  These defaults match the A/B polymer species
# produced by examples/catalyst_run.py.
SPECIES = ("AB", "AAB", "ABB", "AAAA", "BBBB")

# Time points to sample.  Either use TIME_POINTS directly or leave it as None
# and use TIME_RANGE + N_TIME_POINTS.
TIME_POINTS = None
TIME_RANGE = (0.0, 0.2)
N_TIME_POINTS = 20

OUTPUT_DIR = EXAMPLES_DIR / "distribution_comparison"

# If a trajectory ends before a requested time point:
# - "nan": mark the value as missing
# - "hold": use the last recorded state
OUTSIDE = "nan"

ALPHA = 0.05
PRINT_MEMORY = True


def load_metadata_overview(path: Path) -> dict:
    """Read metadata and print a compact structure check."""

    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    shared = payload.get("shared", {})
    if not isinstance(runs, list):
        raise ValueError("metadata JSON must contain a top-level runs list")
    modes = {}
    missing_trajectory = 0
    for row in runs:
        mode = str(row.get("mode", row.get("stepper_method", "unknown")))
        modes[mode] = modes.get(mode, 0) + 1
        if not row.get("trajectory_path"):
            missing_trajectory += 1

    print("[batch metadata]")
    print(f"  path: {path}")
    print(f"  experiment: {payload.get('experiment')}")
    print(f"  n_runs: {len(runs)}")
    print(f"  modes: {modes}")
    print(f"  missing trajectory_path: {missing_trajectory}")
    print(f"  n_species: {shared.get('n_species')}")
    return payload


def validate_species(payload: dict, species) -> None:
    """Print selected species and warn early when a name is not available."""

    names = list(payload.get("shared", {}).get("species_names", []))
    print("[species selection]")
    print(f"  selected: {list(species) if isinstance(species, tuple) else species}")
    if not names:
        print("  metadata has no shared.species_names; validation skipped")
        return
    missing = [item for item in species if isinstance(item, str) and item not in names and not item.isdigit()]
    if missing:
        raise KeyError(f"species name(s) not found in metadata: {missing}")
    print(f"  available species example: {names[:10]}")


def main() -> None:
    payload = load_metadata_overview(METADATA_PATH)
    validate_species(payload, SPECIES)

    print("[distribution comparison]")
    print(f"  groups: {list(GROUPS)}")
    print(f"  time_points: {TIME_POINTS}")
    print(f"  time_range: {TIME_RANGE}")
    print(f"  n_time_points: {N_TIME_POINTS}")
    print(f"  output_dir: {OUTPUT_DIR}")

    result = compare_species_distributions(
        METADATA_PATH,
        species=SPECIES,
        time_points=TIME_POINTS,
        time_range=TIME_RANGE,
        n_time_points=N_TIME_POINTS,
        groups=GROUPS,
        output_dir=OUTPUT_DIR,
        outside=OUTSIDE,
        alpha=ALPHA,
        print_memory=PRINT_MEMORY,
    )

    print("[done]")
    print(f"  temp manifest: {result.extraction.manifest_path}")
    print(f"  within-group plots: {len(result.within_group_plots)}")
    print(f"  comparison plots: {len(result.comparison_plots)}")
    print(f"  statistics csv: {result.statistics_csv}")
    print(f"  statistics json: {result.statistics_json}")
    print(f"  non-significant ratio: {result.non_significant_ratio}")


if __name__ == "__main__":
    main()
