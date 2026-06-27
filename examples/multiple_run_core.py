from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

import catalyst_run
from compute_strategy import ComputeStrategy, apply_cpu_affinity, resolve_compute_strategy

from polymer_sim import (
    BaseRestriction,
    BlendedHybridConfig,
    BlendedHybridStepper,
    CLEStepper,
    ExperimentRunner,
    FixedPartitionStrategy,
    HybridStepper,
    ReactionNetworkData,
    SSAStepper,
    TrajectoryRecorder,
    save_trajectory_record,
)

# Food handling is inherited from catalyst_run.py: formal INFLOW channels plus
# FoodUpperLimitRestriction. This keeps every method on the same capped
# finite-reservoir model.

@dataclass(slots=True)
class MultipleRunConfig:
    """Configuration for a unified multi-method batch run.

    Main user-facing field:
    - ``methods``: pass ``("ssa",)`` for a single-method batch, or for example
      ``("ssa", "blended")`` for a method comparison.  The same seed is reused
      across all methods at the same ``run_order``.
    """

    methods: str | Sequence[str] = ("ssa", "blended")
    n_runs: int = 10
    base_seed: int = 20260524
    t_end: float | None = 0.2
    max_steps: int = 10_000_000
    max_runtime_seconds: float | None = 1800.0
    output_dir: Path | str = EXAMPLES_DIR / "method_run_outputs"
    trajectory_dir_name: str = "trajectories"
    metadata_filename: str = "method_run_metadata.json"
    save_trajectories: bool = True
    compute_strategy: ComputeStrategy = ComputeStrategy(
        backend="process",
        n_workers=None,
        use_gpu=False,
        reserve_logical_cpus=0,
    )

    stepper_dt: float | None = None
    cle_fast_channel_ids: tuple[int, ...] | str | None = None
    hybrid_fast_channel_ids: tuple[int, ...] | str = ()

    blended_i1: float = 10.0
    blended_i2: float = 30.0
    blended_dt_cle: float = 0.01
    blended_dt_macro: float | None = None
    blended_use_reaction_interval_dt: bool = True
    blended_reaction_interval_update_steps: int = 100


_SHARED_NETWORK: ReactionNetworkData | None = None
_SHARED_RESTRICTION: BaseRestriction | None = None


def run_methods(
    methods: str | Sequence[str] = ("ssa", "blended"),
    *,
    n_runs: int = 10,
    base_seed: int = 20260524,
    t_end: float | None = 0.2,
    max_steps: int = 10_000_000,
    max_runtime_seconds: float | None = 1800.0,
    output_dir: Path | str = EXAMPLES_DIR / "method_run_outputs",
    save_trajectories: bool = True,
    compute_strategy: ComputeStrategy | None = None,
    stepper_dt: float | None = None,
    blended_dt_cle: float = 0.01,
    blended_dt_macro: float | None = None,
    blended_use_reaction_interval_dt: bool = True,
    blended_reaction_interval_update_steps: int = 100,
    blended_i1: float = 10.0,
    blended_i2: float = 30.0,
) -> dict[str, object]:
    """Run one or more methods on one shared random catalytic network.

    Examples
    --------
    ``run_methods("ssa", n_runs=10)``
        Run a single-method SSA batch.

    ``run_methods(["ssa", "blended"], n_runs=10)``
        Run a comparison.  For each ``run_order``, SSA and blended receive the
        same random seed while different run orders receive independent seeds.
    """

    config = MultipleRunConfig(
        methods=methods,
        n_runs=int(n_runs),
        base_seed=int(base_seed),
        t_end=None if t_end is None else float(t_end),
        max_steps=int(max_steps),
        max_runtime_seconds=max_runtime_seconds,
        output_dir=output_dir,
        save_trajectories=bool(save_trajectories),
        compute_strategy=compute_strategy or MultipleRunConfig().compute_strategy,
        stepper_dt=stepper_dt,
        blended_dt_cle=float(blended_dt_cle),
        blended_dt_macro=blended_dt_macro,
        blended_use_reaction_interval_dt=bool(blended_use_reaction_interval_dt),
        blended_reaction_interval_update_steps=int(blended_reaction_interval_update_steps),
        blended_i1=float(blended_i1),
        blended_i2=float(blended_i2),
    )
    return run_config(config)


def run_config(config: MultipleRunConfig) -> dict[str, object]:
    """Run a ``MultipleRunConfig`` and write metadata plus optional trajectories."""

    methods = normalize_methods(config.methods)
    if int(config.n_runs) <= 0:
        raise ValueError("n_runs must be > 0")

    network, catalysis_result, restriction = build_shared_objects()
    seeds = make_run_seeds(config.base_seed, config.n_runs)
    output_dir = Path(config.output_dir)
    trajectory_dir = output_dir / config.trajectory_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.save_trajectories:
        trajectory_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(config, methods, seeds, trajectory_dir)
    strategy = resolve_compute_strategy(config.compute_strategy, task_count=len(tasks))
    apply_cpu_affinity(strategy)

    started_at = perf_counter()
    run_records = run_tasks(network, restriction, tasks, strategy)
    total_wall_runtime = perf_counter() - started_at
    run_records = sorted(run_records, key=lambda item: (int(item["run_order"]), int(item["method_order"])))

    payload = metadata_payload(
        config=config,
        methods=methods,
        network=network,
        catalysis_result=catalysis_result,
        seeds=seeds,
        run_records=run_records,
        compute_strategy=strategy,
        total_wall_runtime_seconds=total_wall_runtime,
    )
    metadata_path = output_dir / config.metadata_filename
    metadata_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print_run_summary(payload, metadata_path, trajectory_dir)
    return payload


def normalize_methods(methods: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(methods, str):
        values = (methods,)
    else:
        values = tuple(str(method) for method in methods)
    normalized = tuple(method.lower() for method in values)
    if not normalized:
        raise ValueError("methods must contain at least one method")
    allowed = {"ssa", "cle", "hybrid", "blended"}
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"unknown method(s): {unknown}; allowed methods are {sorted(allowed)}")
    return normalized


def build_shared_objects() -> tuple[ReactionNetworkData, dict, BaseRestriction]:
    network, catalysis_result = catalyst_run.build_random_catalyst_network()
    restriction = catalyst_run.build_food_upper_limit_restriction(network)
    return network, catalysis_result, restriction


def make_run_seeds(base_seed: int, n_runs: int) -> list[int]:
    if int(n_runs) <= 0:
        raise ValueError("n_runs must be > 0")
    seed_sequence = np.random.SeedSequence(int(base_seed))
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(int(n_runs))
    ]


def build_tasks(
    config: MultipleRunConfig,
    methods: Sequence[str],
    seeds: Sequence[int],
    trajectory_dir: Path,
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for run_order, seed in enumerate(seeds):
        for method_order, method in enumerate(methods):
            tasks.append(
                {
                    "run_order": int(run_order),
                    "pair_order": int(run_order),  # compatibility with older paired metadata readers
                    "method_order": int(method_order),
                    "mode": str(method),
                    "seed": int(seed),
                    "base_seed": int(config.base_seed),
                    "t_end": _json_float_or_none(config.t_end),
                    "max_steps": int(config.max_steps),
                    "max_runtime_seconds": config.max_runtime_seconds,
                    "save_trajectories": bool(config.save_trajectories),
                    "trajectory_dir": str(trajectory_dir),
                    "trajectory_name": f"{method}_{int(run_order):03d}.npz",
                    "stepper_dt": config.stepper_dt,
                    "cle_fast_channel_ids": config.cle_fast_channel_ids,
                    "hybrid_fast_channel_ids": config.hybrid_fast_channel_ids,
                    "blended_i1": float(config.blended_i1),
                    "blended_i2": float(config.blended_i2),
                    "blended_dt_cle": float(config.blended_dt_cle),
                    "blended_dt_macro": config.blended_dt_macro,
                    "blended_use_reaction_interval_dt": bool(config.blended_use_reaction_interval_dt),
                    "blended_reaction_interval_update_steps": int(config.blended_reaction_interval_update_steps),
                }
            )
    return tasks


def run_tasks(
    network: ReactionNetworkData,
    restriction: BaseRestriction,
    tasks: Sequence[dict[str, object]],
    compute_strategy: ComputeStrategy,
) -> list[dict[str, object]]:
    backend = str(compute_strategy.backend).lower()
    workers = max(int(compute_strategy.n_workers or 1), 1)
    if backend == "serial" or workers == 1:
        _initialize_worker(network, restriction)
        return [_run_one_task(task) for task in tasks]
    if backend == "thread":
        _initialize_worker(network, restriction)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_run_one_task, tasks))
    if backend == "process":
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(network, restriction),
        ) as executor:
            return list(executor.map(_run_one_task, tasks))
    raise ValueError("compute_strategy.backend must be 'process', 'thread', or 'serial'")


def _initialize_worker(network: ReactionNetworkData, restriction: BaseRestriction) -> None:
    global _SHARED_NETWORK, _SHARED_RESTRICTION
    _SHARED_NETWORK = network
    _SHARED_RESTRICTION = restriction


def _run_one_task(task: dict[str, object]) -> dict[str, object]:
    if _SHARED_NETWORK is None or _SHARED_RESTRICTION is None:
        raise RuntimeError("worker has not been initialized")
    network = _SHARED_NETWORK
    restriction = _SHARED_RESTRICTION

    method = str(task["mode"]).lower()
    stepper, partition_strategy, dt = make_stepper(
        method,
        task["stepper_dt"],
        network,
        task["cle_fast_channel_ids"],
        task["hybrid_fast_channel_ids"],
        task["blended_i1"],
        task["blended_i2"],
        task["blended_dt_cle"],
        task["blended_dt_macro"],
        task["blended_use_reaction_interval_dt"],
        task["blended_reaction_interval_update_steps"],
    )

    recorder = TrajectoryRecorder() if bool(task["save_trajectories"]) else None
    started_at = perf_counter()
    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=_runner_t_end(task["t_end"]),
        seed=int(task["seed"]),
        dt=dt,
        recorder=recorder,
        restriction=restriction,
        partition_strategy=partition_strategy,
        max_steps=int(task["max_steps"]),
        max_runtime_seconds=task["max_runtime_seconds"],
    )
    wall_runtime = perf_counter() - started_at

    summary = result.summary
    trajectory_path: Path | None = None
    if recorder is not None:
        trajectory_record = recorder.finalize()
        trajectory_record.run_metadata.update(_trajectory_metadata(task, method, dt, summary.metadata, wall_runtime))
        trajectory_path = Path(str(task["trajectory_dir"])) / str(task["trajectory_name"])
        save_trajectory_record(trajectory_path, trajectory_record)

    final_state = np.asarray(summary.final_state, dtype=float)
    record = {
        "run_order": int(task["run_order"]),
        "pair_order": int(task["pair_order"]),
        "method_order": int(task["method_order"]),
        "mode": method,
        "stepper_method": method,
        "seed": int(task["seed"]),
        "pair_seed": int(task["seed"]),
        "base_seed": int(task["base_seed"]),
        "requested_t_end": _json_float_or_none(task["t_end"]),
        "simulation_final_time": float(summary.final_time),
        "wall_runtime_seconds": float(wall_runtime),
        "n_steps": int(summary.n_steps),
        "n_events": int(summary.n_events),
        "stop_reason": summary.metadata.get("stop_reason"),
        "final_total_abundance": float(final_state.sum()),
        "max_species_count": float(final_state.max()) if final_state.size else 0.0,
        "state_shape": [int(v) for v in final_state.shape],
    }
    if trajectory_path is not None:
        record["trajectory_path"] = str(trajectory_path)
    return record


def _trajectory_metadata(
    task: dict[str, object],
    method: str,
    dt: float | None,
    summary_metadata: dict,
    wall_runtime: float,
) -> dict[str, object]:
    return {
        "run_order": int(task["run_order"]),
        "pair_order": int(task["pair_order"]),
        "method_order": int(task["method_order"]),
        "mode": method,
        "stepper_method": method,
        "seed": int(task["seed"]),
        "pair_seed": int(task["seed"]),
        "base_seed": int(task["base_seed"]),
        "requested_t_end": _json_float_or_none(task["t_end"]),
        "max_steps": int(task["max_steps"]),
        "max_runtime_seconds": task["max_runtime_seconds"],
        "wall_runtime_seconds": float(wall_runtime),
        "stepper_dt": None if dt is None else float(dt),
        "stop_reason": summary_metadata.get("stop_reason"),
    }


def make_stepper(
    method: str,
    stepper_dt: float | None,
    network: ReactionNetworkData,
    cle_fast_channel_ids,
    hybrid_fast_channel_ids,
    blended_i1: float,
    blended_i2: float,
    blended_dt_cle: float,
    blended_dt_macro: float | None,
    blended_use_reaction_interval_dt: bool,
    blended_reaction_interval_update_steps: int,
):
    name = str(method).lower()
    if name == "ssa":
        return SSAStepper(), None, None if stepper_dt is None else float(stepper_dt)
    if name == "cle":
        dt = _require_dt(name, stepper_dt)
        return CLEStepper(), _fixed_partition(network, cle_fast_channel_ids), dt
    if name == "hybrid":
        dt = _require_dt(name, stepper_dt)
        return HybridStepper(), _fixed_partition(network, hybrid_fast_channel_ids), dt
    if name == "blended":
        config = BlendedHybridConfig(
            i1=blended_i1,
            i2=blended_i2,
            dt_cle=blended_dt_cle,
            dt_macro=blended_dt_macro,
            use_reaction_interval_dt=blended_use_reaction_interval_dt,
            reaction_interval_update_steps=blended_reaction_interval_update_steps,
        )
        return BlendedHybridStepper(config), None, None
    raise ValueError("method must be 'ssa', 'cle', 'hybrid', or 'blended'")


def _require_dt(method: str, stepper_dt: float | None) -> float:
    if stepper_dt is None:
        raise ValueError(f"stepper_dt must be set for {method}")
    dt = float(stepper_dt)
    if dt <= 0.0:
        raise ValueError("stepper_dt must be > 0")
    return dt


def _fixed_partition(network: ReactionNetworkData, fast_channel_ids):
    if fast_channel_ids is None:
        return None
    if isinstance(fast_channel_ids, str):
        if fast_channel_ids.lower() != "all":
            raise ValueError("fast channel ids string must be 'all'")
        return FixedPartitionStrategy(np.arange(network.n_channels, dtype=np.int64))
    return FixedPartitionStrategy(fast_channel_ids)


def _runner_t_end(value: object) -> float:
    # T_END=None means "run until max_runtime_seconds or max_steps".
    return float("inf") if value is None else float(value)


def _json_float_or_none(value: object) -> float | None:
    return None if value is None else float(value)


def metadata_payload(
    *,
    config: MultipleRunConfig,
    methods: Sequence[str],
    network: ReactionNetworkData,
    catalysis_result: dict,
    seeds: Sequence[int],
    run_records: Sequence[dict[str, object]],
    compute_strategy: ComputeStrategy,
    total_wall_runtime_seconds: float,
) -> dict[str, object]:
    return {
        "experiment": "multi_method_run",
        "generated_by": "examples.multiple_run_core.run_methods",
        "shared": {
            "methods": list(methods),
            "n_runs": int(config.n_runs),
            "base_seed": int(config.base_seed),
            "run_seeds": [int(seed) for seed in seeds],
            "requested_t_end": _json_float_or_none(config.t_end),
            "max_steps": int(config.max_steps),
            "max_runtime_seconds": None if config.max_runtime_seconds is None else float(config.max_runtime_seconds),
            "save_trajectories": bool(config.save_trajectories),
            "total_wall_runtime_seconds": float(total_wall_runtime_seconds),
            "compute_strategy": compute_strategy.as_metadata(),
            "n_species": int(network.n_species),
            "n_channels": int(network.n_channels),
            "species_names": list(network.species_names),
            "example_parameters": catalyst_run.example_parameters(),
            "catalysis_assignment": catalyst_run.json_ready(catalysis_result),
            "catalyst_species_names": catalyst_run.catalyst_species_names(network),
            "restriction": {
                "type": "FoodUpperLimitRestriction",
                "food_species": list(catalyst_run.ALPHABET),
                "initial_food_count": float(catalyst_run.INITIAL_FOOD_COUNT),
                "effective_initial_counts": dict(catalyst_run.INITIAL_COUNTS),
                "food_inflow_rate": float(catalyst_run.FOOD_INFLOW_RATE),
                "food_max_count": float(catalyst_run.FOOD_MAX_COUNT),
            },
            "blended_config": {
                "i1": float(config.blended_i1),
                "i2": float(config.blended_i2),
                "dt_cle": float(config.blended_dt_cle),
                "dt_macro": None if config.blended_dt_macro is None else float(config.blended_dt_macro),
                "use_reaction_interval_dt": bool(config.blended_use_reaction_interval_dt),
                "reaction_interval_update_steps": int(config.blended_reaction_interval_update_steps),
            },
        },
        "runs": list(run_records),
    }


def print_run_summary(payload: dict[str, object], metadata_path: Path, trajectory_dir: Path) -> None:
    shared = payload["shared"]
    runs = payload["runs"]
    final_times = np.asarray([float(item["simulation_final_time"]) for item in runs], dtype=float)
    wall_times = np.asarray([float(item["wall_runtime_seconds"]) for item in runs], dtype=float)
    n_events = np.asarray([int(item["n_events"]) for item in runs], dtype=float)
    print("\nMulti-method run:")
    print(
        f"  methods={shared['methods']}, n_runs={shared['n_runs']}, "
        f"backend={shared['compute_strategy']['backend']}, "
        f"workers={shared['compute_strategy']['n_workers']}"
    )
    print(f"  requested_t_end={shared['requested_t_end']}, max_runtime_seconds={shared['max_runtime_seconds']}")
    print(
        f"  simulation_final_time: min={final_times.min():.4f}, "
        f"mean={final_times.mean():.4f}, max={final_times.max():.4f}"
    )
    print(
        f"  wall_runtime_seconds: min={wall_times.min():.3f}, "
        f"mean={wall_times.mean():.3f}, max={wall_times.max():.3f}"
    )
    print(f"  n_events: min={n_events.min():.0f}, mean={n_events.mean():.2f}, max={n_events.max():.0f}")
    print("  by method:")
    for method in shared["methods"]:
        selected = [item for item in runs if item["mode"] == method]
        if not selected:
            continue
        method_final_times = np.asarray([float(item["simulation_final_time"]) for item in selected], dtype=float)
        method_wall_times = np.asarray([float(item["wall_runtime_seconds"]) for item in selected], dtype=float)
        method_events = np.asarray([int(item["n_events"]) for item in selected], dtype=float)
        print(
            f"    {method}: "
            f"simulation_time_mean={method_final_times.mean():.6g}, "
            f"wall_time_mean={method_wall_times.mean():.3f}, "
            f"events_mean={method_events.mean():.2f}"
        )
    if shared["save_trajectories"]:
        print(f"  trajectories saved under: {trajectory_dir}")
    print(f"  metadata saved to: {metadata_path}")
