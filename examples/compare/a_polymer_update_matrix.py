from __future__ import annotations

"""Concrete A-polymer local/global update benchmark matrix.

Networks:
- polymer_a_len5_a5_catalyzes_a_constant_food: max length 5, AAAAA catalyzes A addition.
- polymer_a_len6_a6_catalyzes_a_constant_food: max length 6, AAAAAA catalyzes A addition.
- polymer_a_len8_a8_catalyzes_a_constant_food: max length 8, AAAAAAAA catalyzes A addition.
- polymer_a_len9_a9_catalyzes_a_constant_food: max length 9, AAAAAAAAA catalyzes A addition.
- polymer_a_len10_a10_catalyzes_a_constant_food: max length 10, AAAAAAAAAA catalyzes A addition.

Food handling:
- The monomer A is a food species.
- The network specs use food_supply_mode="constant", so explicit food inflow
  channels are disabled and the runner restriction restores A to its fixed
  chemostat count after each step.

Rows:
- ssa: SSAStepper.
- blended_global_beta_global_propensity: BlendedHybridStepper with full beta
  recomputation and uncached full propensity recomputation.
- blended_local_beta_global_propensity: BlendedHybridStepper with cached/local
  beta updates and uncached full propensity recomputation.
- blended_local_beta_local_propensity: BlendedHybridStepper with cached/local
  beta updates and cached observed propensity updates.  The current stepper may
  still fall back to full recomputation when too many channels are affected.
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


DEFAULT_A_POLYMER_NETWORKS = (
    # "polymer_a_len5_a5_catalyzes_a_constant_food",
    # "polymer_a_len6_a6_catalyzes_a_constant_food",
    "polymer_a_len8_a8_catalyzes_a_constant_food",
    # "polymer_a_len9_a9_catalyzes_a_constant_food",
    "polymer_a_len10_a10_catalyzes_a_constant_food",
    "polymer_a_len12_a12_catalyzes_a_constant_food",
)


def create_a_polymer_update_config(
    *,
    networks: Sequence[str] = DEFAULT_A_POLYMER_NETWORKS,
    wall_seconds: float = 10.0,
    max_steps: int = 100_000_000,
    seed: int = 123,
    t_end: float | None = None,
    blended_i1: float = DEFAULT_SETTINGS.blended_i1,
    blended_i2: float = DEFAULT_SETTINGS.blended_i2,
    blended_dt_cle: float = DEFAULT_SETTINGS.blended_dt_cle,
    blended_dt_macro: float = DEFAULT_SETTINGS.blended_dt_macro,
) -> pd.DataFrame:
    """Create the A-polymer network x method matrix."""

    base_settings: dict[str, Any] = {
        "seed": int(seed),
        "t_end": t_end,
        "max_steps": int(max_steps),
        "max_runtime_seconds": float(wall_seconds),
        "food_supply_mode": "constant",
    }
    blended_base: dict[str, Any] = {
        **base_settings,
        "blended_i1": float(blended_i1),
        "blended_i2": float(blended_i2),
        "blended_dt_cle": float(blended_dt_cle),
        "blended_dt_macro": float(blended_dt_macro),
        "blended_beta_species_mode": "reactants",
    }
    method_configs: list[tuple[str, str, dict[str, Any]]] = [
        (
            "ssa",
            "gillespie_ssa",
            {**base_settings, "ssa_use_local_propensity_updates": True},
        ),
        (
            "blended_global_beta_global_propensity",
            "gillespie_cle_hybrid",
            {
                **blended_base,
                "blended_beta_compute_mode": "beta_fully_compute",
                "blended_strict_int_for_CLE": False,
            },
        ),
        (
            "blended_local_beta_global_propensity",
            "gillespie_cle_hybrid",
            {
                **blended_base,
                "blended_beta_compute_mode": "beta_compute_by_state_difference",
                "blended_strict_int_for_CLE": False,
            },
        ),
        (
            "blended_local_beta_local_propensity",
            "gillespie_cle_hybrid",
            {
                **blended_base,
                "blended_beta_compute_mode": "beta_compute_by_state_difference",
                "blended_strict_int_for_CLE": True,
            },
        ),
    ]

    network_names = [str(network) for network in networks]
    rows: dict[str, dict[str, str]] = {}
    for config_id, method, settings in method_configs:
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


def _json_cell(value: dict[str, Any]) -> str:
    return json.dumps(json_ready(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the concrete A-polymer update-strategy matrix.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--timestamp", default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-run", action="store_true", help="Only write test_config.csv/xlsx.")
    p.add_argument("--networks", nargs="+", default=list(DEFAULT_A_POLYMER_NETWORKS))
    p.add_argument("--wall-seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=DEFAULT_SETTINGS.seed)
    p.add_argument("--t-end", default="none")
    p.add_argument("--max-steps", type=int, default=DEFAULT_SETTINGS.max_steps)
    p.add_argument("--blended-i1", type=float, default=DEFAULT_SETTINGS.blended_i1)
    p.add_argument("--blended-i2", type=float, default=DEFAULT_SETTINGS.blended_i2)
    p.add_argument("--blended-dt-cle", type=float, default=DEFAULT_SETTINGS.blended_dt_cle)
    p.add_argument("--blended-dt-macro", type=float, default=DEFAULT_SETTINGS.blended_dt_macro)
    return p


def main() -> None:
    args = parser().parse_args()
    output_root = Path(args.output_root)
    timestamp = args.timestamp or make_timestamp()
    config = create_a_polymer_update_config(
        networks=tuple(args.networks),
        wall_seconds=float(args.wall_seconds),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        t_end=parse_optional_float(args.t_end),
        blended_i1=float(args.blended_i1),
        blended_i2=float(args.blended_i2),
        blended_dt_cle=float(args.blended_dt_cle),
        blended_dt_macro=float(args.blended_dt_macro),
    )

    run_dir = matrix_run_dir(output_root, timestamp)
    config_paths = write_config_files(config, run_dir)
    print(f"[A-polymer matrix] wrote config csv: {config_paths['csv']}")
    print(f"[A-polymer matrix] wrote config xlsx: {config_paths['xlsx']}")
    print(f"[A-polymer matrix] networks={list(config.columns)}")
    print(f"[A-polymer matrix] method_configs={list(config.index)}")
    if args.no_run:
        return

    run_experiment_matrix(
        config,
        output_root=output_root,
        timestamp=timestamp,
        workers=int(args.workers),
    )


if __name__ == "__main__":
    main()
