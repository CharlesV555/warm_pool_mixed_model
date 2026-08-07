from __future__ import annotations

"""Beta-hybrid local/global update test matrix.

This entry point runs the four beta-hybrid update combinations:
- global beta + global propensity
- global beta + local propensity
- local beta + global propensity
- local beta + local propensity

In the current ``BlendedHybridStepper`` configuration:
- beta is global when ``blended_beta_compute_mode="beta_fully_compute"``;
- beta is local when ``blended_beta_compute_mode="beta_compute_by_state_difference"``;
- observed propensity is global/uncached when ``blended_strict_int_for_CLE=False``;
- observed propensity uses the rounded-state cache and local update path when
  ``blended_strict_int_for_CLE=True``. It may still fall back to full recompute
  if too many channels are affected.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from common import DEFAULT_SETTINGS, json_ready
from experiment_matrix import (
    DEFAULT_OUTPUT_ROOT,
    make_timestamp,
    matrix_run_dir,
    parse_optional_float,
    run_experiment_matrix,
    write_config_files,
)


DEFAULT_BETA_HYBRID_NETWORKS = (
    "polymer_a_len5_a5_catalyzes_a_constant_food",
    "polymer_a_len6_a6_catalyzes_a_constant_food",
    "polymer_a_len8_a8_catalyzes_a_constant_food",
)


def create_beta_hybrid_update_config(
    *,
    networks: Sequence[str] = DEFAULT_BETA_HYBRID_NETWORKS,
    configs: Sequence[str] | None = None,
    wall_seconds: float = 10.0,
    max_steps: int = 100_000_000,
    seed: int = 123,
    t_end: float | None = None,
    food_supply_mode: str = "constant",
    blended_i1: float = DEFAULT_SETTINGS.blended_i1,
    blended_i2: float = DEFAULT_SETTINGS.blended_i2,
    blended_dt_cle: float = DEFAULT_SETTINGS.blended_dt_cle,
    blended_dt_macro: float = DEFAULT_SETTINGS.blended_dt_macro,
    blended_beta_species_mode: str = DEFAULT_SETTINGS.blended_beta_species_mode,
) -> pd.DataFrame:
    """Create a network x update-strategy matrix for beta-hybrid profiling."""

    base_settings: dict[str, Any] = {
        "seed": int(seed),
        "t_end": t_end,
        "max_steps": int(max_steps),
        "max_runtime_seconds": float(wall_seconds),
        "food_supply_mode": str(food_supply_mode),
        "blended_i1": float(blended_i1),
        "blended_i2": float(blended_i2),
        "blended_dt_cle": float(blended_dt_cle),
        "blended_dt_macro": float(blended_dt_macro),
        "blended_beta_species_mode": str(blended_beta_species_mode),
    }
    method_configs: list[tuple[str, dict[str, Any]]] = [
        (
            "beta_hybrid_global_beta_global_propensity",
            {
                **base_settings,
                "blended_beta_compute_mode": "beta_fully_compute",
                "blended_strict_int_for_CLE": False,
            },
        ),
        (
            "beta_hybrid_global_beta_local_propensity",
            {
                **base_settings,
                "blended_beta_compute_mode": "beta_fully_compute",
                "blended_strict_int_for_CLE": True,
            },
        ),
        (
            "beta_hybrid_local_beta_global_propensity",
            {
                **base_settings,
                "blended_beta_compute_mode": "beta_compute_by_state_difference",
                "blended_strict_int_for_CLE": False,
            },
        ),
        (
            "beta_hybrid_local_beta_local_propensity",
            {
                **base_settings,
                "blended_beta_compute_mode": "beta_compute_by_state_difference",
                "blended_strict_int_for_CLE": True,
            },
        ),
    ]
    selected_configs = None if configs is None else {str(config) for config in configs}
    if selected_configs is not None:
        known_configs = {config_id for config_id, _settings in method_configs}
        unknown = sorted(selected_configs - known_configs)
        if unknown:
            raise ValueError(f"unknown beta-hybrid configs: {unknown}; expected one of {sorted(known_configs)}")
        method_configs = [
            (config_id, settings)
            for config_id, settings in method_configs
            if config_id in selected_configs
        ]

    network_names = [str(network) for network in networks]
    rows: dict[str, dict[str, str]] = {}
    for config_id, settings in method_configs:
        rows[config_id] = {
            network: _json_cell(
                {
                    "enabled": True,
                    "function": "run_matrix_cell",
                    "method": "gillespie_cle_hybrid",
                    "settings": settings,
                }
            )
            for network in network_names
        }
    df = pd.DataFrame.from_dict(rows, orient="index", columns=network_names)
    df.index.name = "method_config_id"
    return df


def _json_cell(value: dict[str, Any]) -> str:
    return json.dumps(json_ready(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run beta-hybrid global/local beta and propensity update tests.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--timestamp", default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-run", action="store_true", help="Only write test_config.csv/xlsx.")
    p.add_argument("--networks", nargs="+", default=list(DEFAULT_BETA_HYBRID_NETWORKS))
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=(
            "beta_hybrid_global_beta_global_propensity",
            "beta_hybrid_global_beta_local_propensity",
            "beta_hybrid_local_beta_global_propensity",
            "beta_hybrid_local_beta_local_propensity",
        ),
        help="Subset of beta-hybrid update configs to run. Omit to run all four.",
    )
    p.add_argument(
        "--food-supply-mode",
        choices=("explicit_inflow", "constant", "upper_limit", "none"),
        default="constant",
    )
    p.add_argument("--wall-seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=DEFAULT_SETTINGS.seed)
    p.add_argument("--t-end", default="none")
    p.add_argument("--max-steps", type=int, default=DEFAULT_SETTINGS.max_steps)
    p.add_argument("--blended-i1", type=float, default=DEFAULT_SETTINGS.blended_i1)
    p.add_argument("--blended-i2", type=float, default=DEFAULT_SETTINGS.blended_i2)
    p.add_argument("--blended-dt-cle", type=float, default=DEFAULT_SETTINGS.blended_dt_cle)
    p.add_argument("--blended-dt-macro", type=float, default=DEFAULT_SETTINGS.blended_dt_macro)
    p.add_argument(
        "--blended-beta-species-mode",
        choices=("reactants", "products", "reactants_products"),
        default=DEFAULT_SETTINGS.blended_beta_species_mode,
    )
    p.add_argument("--profile", dest="profile", action="store_true")
    p.add_argument("--no-profile", dest="profile", action="store_false")
    p.set_defaults(profile=True)
    p.add_argument("--profile-limit", type=int, default=40)
    return p


def main() -> None:
    args = parser().parse_args()
    output_root = Path(args.output_root)
    timestamp = args.timestamp or make_timestamp()
    config = create_beta_hybrid_update_config(
        networks=tuple(args.networks),
        configs=None if args.configs is None else tuple(args.configs),
        wall_seconds=float(args.wall_seconds),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        t_end=parse_optional_float(args.t_end),
        food_supply_mode=str(args.food_supply_mode),
        blended_i1=float(args.blended_i1),
        blended_i2=float(args.blended_i2),
        blended_dt_cle=float(args.blended_dt_cle),
        blended_dt_macro=float(args.blended_dt_macro),
        blended_beta_species_mode=str(args.blended_beta_species_mode),
    )

    run_dir = matrix_run_dir(output_root, timestamp)
    config_paths = write_config_files(config, run_dir)
    print(f"[beta-hybrid matrix] wrote config csv: {config_paths['csv']}")
    print(f"[beta-hybrid matrix] wrote config xlsx: {config_paths['xlsx']}")
    print(f"[beta-hybrid matrix] networks={list(config.columns)}")
    print(f"[beta-hybrid matrix] method_configs={list(config.index)}")
    if args.no_run:
        return

    run_experiment_matrix(
        config,
        output_root=output_root,
        timestamp=timestamp,
        workers=int(args.workers),
        profile=bool(args.profile),
        profile_limit=int(args.profile_limit),
    )


if __name__ == "__main__":
    main()
