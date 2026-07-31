from __future__ import annotations

"""Batch entry for fixed wall-clock cProfile comparisons.

Edit NETWORKS, METHODS, and SETTINGS below, or override the common options from
the command line.  The batch entry always writes:
- report/profile_top*_report.* for parsed cProfile entries;
- report/simulation_summary_table.* for simulation time, step count, and event
  count summaries across all network/method pairs.
"""

import argparse
from pathlib import Path

from common import (
    DEFAULT_NETWORKS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SETTINGS,
    METHOD_ORDER,
    NETWORK_SPECS,
    RunSettings,
    run_comparison,
    write_simulation_summary_tables,
)


NETWORKS = DEFAULT_NETWORKS
METHODS = METHOD_ORDER
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
SETTINGS = DEFAULT_SETTINGS


def main() -> None:
    args = _parser().parse_args()
    settings = RunSettings(
        seed=int(args.seed),
        t_end=None if str(args.t_end).lower() in {"none", "null", "inf", "infinity"} else float(args.t_end),
        max_steps=int(args.max_steps),
        max_runtime_seconds=float(args.wall_seconds),
        blended_i1=float(args.blended_i1),
        blended_i2=float(args.blended_i2),
        blended_dt_cle=float(args.blended_dt_cle),
        blended_dt_macro=float(args.blended_dt_macro),
        pdmp_ode_step=float(args.pdmp_ode_step),
    )
    records = run_comparison(
        networks=tuple(args.networks),
        methods=tuple(args.methods),
        settings=settings,
        output_dir=Path(args.output_dir),
        profile_dir=args.profile_dir,
        profile_limit=int(args.profile_limit),
    )
    write_simulation_summary_tables(records, Path(args.output_dir))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect cProfile top-time entries under a fixed wall-clock budget.")
    parser.add_argument("--networks", nargs="+", default=list(NETWORKS), choices=sorted(NETWORK_SPECS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHOD_ORDER))
    parser.add_argument("--wall-seconds", type=float, default=SETTINGS.max_runtime_seconds)
    parser.add_argument("--seed", type=int, default=SETTINGS.seed)
    parser.add_argument("--t-end", default="none", help="Use 'none' to stop by wall-clock/max-steps.")
    parser.add_argument("--max-steps", type=int, default=SETTINGS.max_steps)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--blended-i1", type=float, default=SETTINGS.blended_i1)
    parser.add_argument("--blended-i2", type=float, default=SETTINGS.blended_i2)
    parser.add_argument("--blended-dt-cle", type=float, default=SETTINGS.blended_dt_cle)
    parser.add_argument("--blended-dt-macro", type=float, default=SETTINGS.blended_dt_macro)
    parser.add_argument("--pdmp-ode-step", type=float, default=SETTINGS.pdmp_ode_step)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--profile-limit", type=int, default=40)
    return parser


if __name__ == "__main__":
    main()
