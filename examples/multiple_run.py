from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
    RunSummary,
    SSAStepper,
    TrajectoryRecorder,
    save_summary,
    save_trajectory_record,
)
from polymer_sim.simulation.restriction import build_restriction


BATCH_SIZE = 8
BASE_RUN_SEED = 123
N_WORKERS = 4
PARALLEL_BACKEND = "process"  # "process", "thread", or "serial"

STEPPER_METHOD = "ssa"  # "ssa", "cle", "hybrid", or "blended"
STEPPER_DT = None  # Set to a positive value for "cle" or "hybrid", for example 0.01.
CLE_FAST_CHANNEL_IDS = None  # None means all channels are treated as CLE channels.
HYBRID_FAST_CHANNEL_IDS: tuple[int, ...] | str = ()  # Use "all" or a tuple of channel ids.
BLENDED_I1 = 10.0
BLENDED_I2 = 30.0
BLENDED_DT_CLE = 0.01 if STEPPER_DT is None else STEPPER_DT
BLENDED_DT_MACRO = STEPPER_DT
BLENDED_USE_REACTION_INTERVAL_DT = True
BLENDED_REACTION_INTERVAL_UPDATE_STEPS = 100

SAVE_TRAJECTORIES = False
KEEP_CHANNEL_LABELS_IN_SUMMARY = False
OUTPUT_DIR = EXAMPLES_DIR / "multiple_run_outputs"
SUMMARY_FILENAME = "multiple_run_summary.json"
METADATA_FILENAME = "multiple_run_metadata.json"
TRAJECTORY_DIR_NAME = "trajectories"

MAIN_RUN_MODE = "batch"  # "batch" or "paired_method_test"

PAIRED_N_PAIRS = 10
PAIRED_BASE_SEED = 20260524
PAIRED_T_END = 0.2
PAIRED_MAX_STEPS = 10_000_000
PAIRED_MAX_RUNTIME_SECONDS = 1800.0
PAIRED_OUTPUT_DIR = EXAMPLES_DIR / "paired_method_outputs"
PAIRED_METADATA_FILENAME = "paired_method_metadata.json"
PAIRED_TRAJECTORY_DIR_NAME = "trajectories"
PAIRED_COMPUTE_STRATEGY = ComputeStrategy(
    backend="process",
    n_workers=None,  # None means auto-detect logical CPUs, then cap by task count.
    use_gpu=False,
    reserve_logical_cpus=0,
)


_SHARED_NETWORK: ReactionNetworkData | None = None
_SHARED_RESTRICTION: BaseRestriction | None = None


def build_shared_objects() -> tuple[ReactionNetworkData, dict, BaseRestriction]:
    network, catalysis_result = catalyst_run.build_random_catalyst_network()
    restriction = build_restriction(
        network,
        food_species=catalyst_run.ALPHABET,
        food_count=catalyst_run.FOOD_COUNT,
    )
    return network, catalysis_result, restriction


def make_run_seeds(base_seed: int, batch_size: int) -> list[int]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    seed_sequence = np.random.SeedSequence(int(base_seed))
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(int(batch_size))
    ]


def run_batch(
    network: ReactionNetworkData,
    restriction: BaseRestriction,
    *,
    seeds: Sequence[int] | None = None,
    batch_size: int = BATCH_SIZE,
    base_seed: int = BASE_RUN_SEED,
    stepper_method: str = STEPPER_METHOD,
    stepper_dt: float | None = STEPPER_DT,
    n_workers: int = N_WORKERS,
    parallel_backend: str = PARALLEL_BACKEND,
    save_trajectories: bool = SAVE_TRAJECTORIES,
) -> list[RunSummary]:
    run_seeds = [int(seed) for seed in seeds] if seeds is not None else make_run_seeds(base_seed, batch_size)
    tasks = [
        {
            "run_index": idx,
            "seed": seed,
            "base_run_seed": int(base_seed),
            "stepper_method": stepper_method,
            "stepper_dt": stepper_dt,
            "max_steps": catalyst_run.MAX_STEPS,
            "max_runtime_seconds": catalyst_run.MAX_TIMES,
            "save_trajectories": bool(save_trajectories),
            "trajectory_dir": str(OUTPUT_DIR / TRAJECTORY_DIR_NAME),
            "keep_channel_labels": bool(KEEP_CHANNEL_LABELS_IN_SUMMARY),
            "cle_fast_channel_ids": CLE_FAST_CHANNEL_IDS,
            "hybrid_fast_channel_ids": HYBRID_FAST_CHANNEL_IDS,
            "blended_i1": BLENDED_I1,
            "blended_i2": BLENDED_I2,
            "blended_dt_cle": BLENDED_DT_CLE,
            "blended_dt_macro": BLENDED_DT_MACRO,
            "blended_use_reaction_interval_dt": BLENDED_USE_REACTION_INTERVAL_DT,
            "blended_reaction_interval_update_steps": BLENDED_REACTION_INTERVAL_UPDATE_STEPS,
        }
        for idx, seed in enumerate(run_seeds)
    ]

    backend = str(parallel_backend).lower()
    workers = max(int(n_workers), 1)
    if backend == "serial" or workers == 1:
        _initialize_worker(network, restriction)
        return [_run_one_from_task(task) for task in tasks]
    if backend == "thread":
        _initialize_worker(network, restriction)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_run_one_from_task, tasks))
    if backend == "process":
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(network, restriction),
        ) as executor:
            return list(executor.map(_run_one_from_task, tasks))
    raise ValueError("parallel_backend must be 'process', 'thread', or 'serial'")


def run_paired_ssa_blended_test(
    *,
    n_pairs: int = PAIRED_N_PAIRS,
    base_seed: int = PAIRED_BASE_SEED,
    t_end: float = PAIRED_T_END,
    max_steps: int = PAIRED_MAX_STEPS,
    max_runtime_seconds: float | None = PAIRED_MAX_RUNTIME_SECONDS,
    output_dir: Path | str = PAIRED_OUTPUT_DIR,
    compute_strategy: ComputeStrategy | None = None,
) -> dict[str, object]:
    """Run paired SSA and blended trajectories on one shared random network.

    Each pair uses the same seed for SSA and blended.  Different pairs use
    independent child seeds derived from ``base_seed``.
    """

    pair_count = int(n_pairs)
    if pair_count <= 0:
        raise ValueError("n_pairs must be > 0")
    output_path = Path(output_dir)
    trajectory_dir = output_path / PAIRED_TRAJECTORY_DIR_NAME

    network, catalysis_result, restriction = build_shared_objects()
    seeds = make_run_seeds(base_seed, pair_count)
    task_count = pair_count * 2
    resolved_strategy = resolve_compute_strategy(
        compute_strategy or PAIRED_COMPUTE_STRATEGY,
        task_count=task_count,
    )
    apply_cpu_affinity(resolved_strategy)

    tasks = []
    for pair_order, seed in enumerate(seeds):
        for mode in ("ssa", "blended"):
            tasks.append(
                {
                    "pair_order": int(pair_order),
                    "mode": mode,
                    "seed": int(seed),
                    "base_seed": int(base_seed),
                    "t_end": float(t_end),
                    "max_steps": int(max_steps),
                    "max_runtime_seconds": None if max_runtime_seconds is None else float(max_runtime_seconds),
                    "trajectory_dir": str(trajectory_dir),
                    "trajectory_name": f"{mode}_{int(pair_order):03d}.npz",
                    "blended_i1": BLENDED_I1,
                    "blended_i2": BLENDED_I2,
                    "blended_dt_cle": BLENDED_DT_CLE,
                    "blended_dt_macro": BLENDED_DT_MACRO,
                    "blended_use_reaction_interval_dt": BLENDED_USE_REACTION_INTERVAL_DT,
                    "blended_reaction_interval_update_steps": BLENDED_REACTION_INTERVAL_UPDATE_STEPS,
                }
            )

    output_path.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    started_at = perf_counter()
    run_records = _run_paired_tasks(network, restriction, tasks, resolved_strategy)
    total_wall_runtime = perf_counter() - started_at
    run_records = sorted(run_records, key=lambda item: (int(item["pair_order"]), str(item["mode"])))

    payload = _paired_metadata_payload(
        network=network,
        catalysis_result=catalysis_result,
        seeds=seeds,
        run_records=run_records,
        compute_strategy=resolved_strategy,
        n_pairs=pair_count,
        base_seed=base_seed,
        t_end=t_end,
        max_steps=max_steps,
        max_runtime_seconds=max_runtime_seconds,
        total_wall_runtime_seconds=total_wall_runtime,
    )
    metadata_path = output_path / PAIRED_METADATA_FILENAME
    metadata_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print_paired_test_summary(payload, metadata_path, trajectory_dir)
    return payload


def _run_paired_tasks(
    network: ReactionNetworkData,
    restriction: BaseRestriction,
    tasks: Sequence[dict],
    compute_strategy: ComputeStrategy,
) -> list[dict[str, object]]:
    backend = compute_strategy.backend
    workers = max(int(compute_strategy.n_workers or 1), 1)
    if backend == "serial" or workers == 1:
        _initialize_worker(network, restriction)
        return [_run_paired_method_task(task) for task in tasks]
    if backend == "thread":
        _initialize_worker(network, restriction)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_run_paired_method_task, tasks))
    if backend == "process":
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(network, restriction),
        ) as executor:
            return list(executor.map(_run_paired_method_task, tasks))
    raise ValueError("compute_strategy.backend must be 'process', 'thread', or 'serial'")


def _run_paired_method_task(task: dict) -> dict[str, object]:
    if _SHARED_NETWORK is None or _SHARED_RESTRICTION is None:
        raise RuntimeError("worker has not been initialized")
    network = _SHARED_NETWORK
    restriction = _SHARED_RESTRICTION

    mode = str(task["mode"]).lower()
    if mode not in {"ssa", "blended"}:
        raise ValueError("paired method mode must be 'ssa' or 'blended'")
    stepper, partition_strategy, dt = make_stepper(
        mode,
        None,
        network,
        None,
        (),
        task["blended_i1"],
        task["blended_i2"],
        task["blended_dt_cle"],
        task["blended_dt_macro"],
        task["blended_use_reaction_interval_dt"],
        task["blended_reaction_interval_update_steps"],
    )

    recorder = TrajectoryRecorder()
    started_at = perf_counter()
    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=float(task["t_end"]),
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
    trajectory_record = recorder.finalize()
    trajectory_metadata = {
        "pair_order": int(task["pair_order"]),
        "mode": mode,
        "pair_seed": int(task["seed"]),
        "base_seed": int(task["base_seed"]),
        "requested_t_end": float(task["t_end"]),
        "max_steps": int(task["max_steps"]),
        "max_runtime_seconds": task["max_runtime_seconds"],
        "wall_runtime_seconds": float(wall_runtime),
        "stepper_method": mode,
        "stepper_dt": None if dt is None else float(dt),
        "stop_reason": summary.metadata.get("stop_reason"),
    }
    trajectory_record.run_metadata.update(trajectory_metadata)
    trajectory_path = Path(str(task["trajectory_dir"])) / str(task["trajectory_name"])
    save_trajectory_record(trajectory_path, trajectory_record)

    final_state = np.asarray(summary.final_state, dtype=float)
    return {
        "pair_order": int(task["pair_order"]),
        "mode": mode,
        "seed": int(task["seed"]),
        "trajectory_path": str(trajectory_path),
        "requested_t_end": float(task["t_end"]),
        "simulation_final_time": float(summary.final_time),
        "wall_runtime_seconds": float(wall_runtime),
        "n_steps": int(summary.n_steps),
        "n_events": int(summary.n_events),
        "stop_reason": summary.metadata.get("stop_reason"),
        "final_total_abundance": float(final_state.sum()),
        "max_species_count": float(final_state.max()) if final_state.size else 0.0,
        "n_recorded_points": int(trajectory_record.times.shape[0]),
        "state_shape": [int(v) for v in trajectory_record.states.shape],
    }


def _paired_metadata_payload(
    *,
    network: ReactionNetworkData,
    catalysis_result: dict,
    seeds: Sequence[int],
    run_records: Sequence[dict[str, object]],
    compute_strategy: ComputeStrategy,
    n_pairs: int,
    base_seed: int,
    t_end: float,
    max_steps: int,
    max_runtime_seconds: float | None,
    total_wall_runtime_seconds: float,
) -> dict[str, object]:
    return {
        "experiment": "paired_ssa_blended",
        "generated_by": "examples.multiple_run.run_paired_ssa_blended_test",
        "shared": {
            "n_pairs": int(n_pairs),
            "base_seed": int(base_seed),
            "pair_seeds": [int(seed) for seed in seeds],
            "requested_t_end": float(t_end),
            "max_steps": int(max_steps),
            "max_runtime_seconds": None if max_runtime_seconds is None else float(max_runtime_seconds),
            "total_wall_runtime_seconds": float(total_wall_runtime_seconds),
            "compute_strategy": compute_strategy.as_metadata(),
            "n_species": int(network.n_species),
            "n_channels": int(network.n_channels),
            "species_names": list(network.species_names),
            "example_parameters": catalyst_run.example_parameters(),
            "catalysis_assignment": catalyst_run.json_ready(catalysis_result),
            "catalyst_species_names": catalyst_run.catalyst_species_names(network),
            "restriction": {
                "food_species": list(catalyst_run.ALPHABET),
                "food_count": float(catalyst_run.FOOD_COUNT),
            },
            "paired_modes": ["ssa", "blended"],
            "blended_config": {
                "i1": float(BLENDED_I1),
                "i2": float(BLENDED_I2),
                "dt_cle": float(BLENDED_DT_CLE),
                "dt_macro": None if BLENDED_DT_MACRO is None else float(BLENDED_DT_MACRO),
                "use_reaction_interval_dt": bool(BLENDED_USE_REACTION_INTERVAL_DT),
                "reaction_interval_update_steps": int(BLENDED_REACTION_INTERVAL_UPDATE_STEPS),
            },
        },
        "runs": list(run_records),
    }


def print_paired_test_summary(
    payload: dict[str, object],
    metadata_path: Path,
    trajectory_dir: Path,
) -> None:
    shared = payload["shared"]
    runs = payload["runs"]
    final_times = np.asarray([float(item["simulation_final_time"]) for item in runs], dtype=float)
    wall_times = np.asarray([float(item["wall_runtime_seconds"]) for item in runs], dtype=float)
    print("\nPaired SSA/blended test:")
    print(
        f"  pairs={shared['n_pairs']}, "
        f"backend={shared['compute_strategy']['backend']}, "
        f"workers={shared['compute_strategy']['n_workers']}"
    )
    print(
        f"  simulation_final_time: min={final_times.min():.4f}, "
        f"mean={final_times.mean():.4f}, max={final_times.max():.4f}"
    )
    print(
        f"  wall_runtime_seconds: min={wall_times.min():.3f}, "
        f"mean={wall_times.mean():.3f}, max={wall_times.max():.3f}"
    )
    print(f"  trajectories saved under: {trajectory_dir}")
    print(f"  metadata saved to: {metadata_path}")


def _initialize_worker(network: ReactionNetworkData, restriction: BaseRestriction) -> None:
    global _SHARED_NETWORK, _SHARED_RESTRICTION
    _SHARED_NETWORK = network
    _SHARED_RESTRICTION = restriction


def _run_one_from_task(task: dict) -> RunSummary:
    if _SHARED_NETWORK is None or _SHARED_RESTRICTION is None:
        raise RuntimeError("worker has not been initialized")
    network = _SHARED_NETWORK
    restriction = _SHARED_RESTRICTION

    stepper, partition_strategy, dt = make_stepper(
        task["stepper_method"],
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
    recorder = TrajectoryRecorder() if task["save_trajectories"] else None
    result = ExperimentRunner().run_one(
        network,
        stepper,
        t_end=catalyst_run.T_END,
        seed=int(task["seed"]),
        dt=dt,
        recorder=recorder,
        restriction=restriction,
        partition_strategy=partition_strategy,
        max_steps=int(task["max_steps"]),
        max_runtime_seconds=task["max_runtime_seconds"],
    )

    summary = result.summary
    summary.metadata.update(
        {
            "run_index": int(task["run_index"]),
            "seed": int(task["seed"]),
            "base_run_seed": int(task["base_run_seed"]),
            "stepper_method": str(task["stepper_method"]).lower(),
            "stepper_dt": None if dt is None else float(dt),
        }
    )
    if not task["keep_channel_labels"]:
        summary.metadata.pop("channel_labels", None)

    if recorder is not None:
        record = recorder.finalize()
        record.run_metadata.update(summary.metadata)
        trajectory_dir = Path(str(task["trajectory_dir"]))
        trajectory_path = trajectory_dir / f"run_{int(task['run_index']):04d}_seed_{int(task['seed'])}.npz"
        save_trajectory_record(trajectory_path, record)

    return summary


def make_stepper(
    stepper_method: str,
    stepper_dt: float | None,
    network: ReactionNetworkData,
    cle_fast_channel_ids,
    hybrid_fast_channel_ids,
    blended_i1: float = BLENDED_I1,
    blended_i2: float = BLENDED_I2,
    blended_dt_cle: float = BLENDED_DT_CLE,
    blended_dt_macro: float | None = BLENDED_DT_MACRO,
    blended_use_reaction_interval_dt: bool = BLENDED_USE_REACTION_INTERVAL_DT,
    blended_reaction_interval_update_steps: int = BLENDED_REACTION_INTERVAL_UPDATE_STEPS,
):
    method = str(stepper_method).lower()
    if method == "ssa":
        return SSAStepper(), None, None if stepper_dt is None else float(stepper_dt)
    if method == "cle":
        dt = _require_dt(method, stepper_dt)
        return CLEStepper(), _fixed_partition(network, cle_fast_channel_ids), dt
    if method == "hybrid":
        dt = _require_dt(method, stepper_dt)
        return HybridStepper(), _fixed_partition(network, hybrid_fast_channel_ids), dt
    if method == "blended":
        config = BlendedHybridConfig(
            i1=blended_i1,
            i2=blended_i2,
            dt_cle=blended_dt_cle,
            dt_macro=blended_dt_macro,
            use_reaction_interval_dt=blended_use_reaction_interval_dt,
            reaction_interval_update_steps=blended_reaction_interval_update_steps,
        )
        return BlendedHybridStepper(config), None, None
    raise ValueError("stepper_method must be 'ssa', 'cle', 'hybrid', or 'blended'")


def _require_dt(stepper_method: str, stepper_dt: float | None) -> float:
    if stepper_dt is None:
        raise ValueError(f"STEPPER_DT must be set for {stepper_method}")
    dt = float(stepper_dt)
    if dt <= 0.0:
        raise ValueError("STEPPER_DT must be > 0")
    return dt


def _fixed_partition(network: ReactionNetworkData, fast_channel_ids):
    if fast_channel_ids is None:
        return None
    if isinstance(fast_channel_ids, str):
        if fast_channel_ids.lower() != "all":
            raise ValueError("fast channel ids string must be 'all'")
        return FixedPartitionStrategy(np.arange(network.n_channels, dtype=np.int64))
    return FixedPartitionStrategy(fast_channel_ids)


def save_batch_metadata(
    path: Path,
    *,
    network: ReactionNetworkData,
    catalysis_result: dict,
    seeds: Sequence[int],
    summaries: Sequence[RunSummary],
    stepper_method: str,
    stepper_dt: float | None,
    parallel_backend: str,
    n_workers: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_size": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "stepper_method": str(stepper_method),
        "stepper_dt": None if stepper_dt is None else float(stepper_dt),
        "blended_i1": BLENDED_I1,
        "blended_i2": BLENDED_I2,
        "blended_dt_cle": BLENDED_DT_CLE,
        "blended_dt_macro": BLENDED_DT_MACRO,
        "blended_use_reaction_interval_dt": BLENDED_USE_REACTION_INTERVAL_DT,
        "blended_reaction_interval_update_steps": BLENDED_REACTION_INTERVAL_UPDATE_STEPS,
        "parallel_backend": str(parallel_backend),
        "n_workers": int(n_workers),
        "n_species": int(network.n_species),
        "n_channels": int(network.n_channels),
        "species_names": list(network.species_names),
        "example_parameters": catalyst_run.example_parameters(),
        "catalysis_assignment": catalyst_run.json_ready(catalysis_result),
        "restriction": {
            "food_species": list(catalyst_run.ALPHABET),
            "food_count": float(catalyst_run.FOOD_COUNT),
        },
        "n_events": [int(summary.n_events) for summary in summaries],
        "final_times": [float(summary.final_time) for summary in summaries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def print_batch_summary(summaries: Sequence[RunSummary]) -> None:
    n_events = np.asarray([summary.n_events for summary in summaries], dtype=float)
    final_times = np.asarray([summary.final_time for summary in summaries], dtype=float)
    print("\nMultiple-run summary:")
    print(f"  runs={len(summaries)}, stepper={STEPPER_METHOD}, backend={PARALLEL_BACKEND}, workers={N_WORKERS}")
    print(f"  final_time: min={final_times.min():.4f}, mean={final_times.mean():.4f}, max={final_times.max():.4f}")
    print(f"  n_events: min={n_events.min():.0f}, mean={n_events.mean():.2f}, max={n_events.max():.0f}")


def main() -> None:
    if MAIN_RUN_MODE == "paired_method_test":
        run_paired_ssa_blended_test()
        return
    if MAIN_RUN_MODE != "batch":
        raise ValueError("MAIN_RUN_MODE must be 'batch' or 'paired_method_test'")

    network, catalysis_result, restriction = build_shared_objects()
    seeds = make_run_seeds(BASE_RUN_SEED, BATCH_SIZE)

    print("Multiple random-catalyst runs")
    print(f"n_species={network.n_species}, n_channels={network.n_channels}")
    print(f"shared catalyst assignment={catalyst_run.CATALYST_ASSIGNMENT_MODE}")
    print(f"seeds={seeds}")

    summaries = run_batch(
        network,
        restriction,
        seeds=seeds,
        stepper_method=STEPPER_METHOD,
        stepper_dt=STEPPER_DT,
        base_seed=BASE_RUN_SEED,
        n_workers=N_WORKERS,
        parallel_backend=PARALLEL_BACKEND,
        save_trajectories=SAVE_TRAJECTORIES,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / SUMMARY_FILENAME
    metadata_path = OUTPUT_DIR / METADATA_FILENAME
    save_summary(summary_path, list(summaries))
    save_batch_metadata(
        metadata_path,
        network=network,
        catalysis_result=catalysis_result,
        seeds=seeds,
        summaries=summaries,
        stepper_method=STEPPER_METHOD,
        stepper_dt=STEPPER_DT,
        parallel_backend=PARALLEL_BACKEND,
        n_workers=N_WORKERS,
    )

    print_batch_summary(summaries)
    print(f"  summary saved to: {summary_path}")
    print(f"  metadata saved to: {metadata_path}")
    if SAVE_TRAJECTORIES:
        print(f"  trajectories saved under: {OUTPUT_DIR / TRAJECTORY_DIR_NAME}")


if __name__ == "__main__":
    main()
