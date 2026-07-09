from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import importlib
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
    network_source: str = "random"
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
    blended_dt_cle: float | Sequence[float] = 0.01
    blended_dt_macro: float | Sequence[float | None] | None = None
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
    network_source: str = "random",
    save_trajectories: bool = True,
    compute_strategy: ComputeStrategy | None = None,
    stepper_dt: float | None = None,
    blended_dt_cle: float | Sequence[float] = 0.01,
    blended_dt_macro: float | Sequence[float | None] | None = None,
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

    ``run_methods("blended", blended_dt_cle=[1e-4, 1e-3], blended_dt_macro=[1e-3, 1e-2])``
        Run every ``dt_cle x dt_macro`` pair satisfying ``dt_macro >= dt_cle``.
        Set ``blended_use_reaction_interval_dt=False`` for a fixed-dt sweep.

    ``run_methods(["ssa", "blended"], network_source="oscillator")``
        Use examples/oscillator.py instead of the default random catalyst
        network.  Built-in sources are ``"random"``, ``"oscillator"``, and
        ``"cross_catalysis"``.
    """

    config = MultipleRunConfig(
        methods=methods,
        n_runs=int(n_runs),
        base_seed=int(base_seed),
        t_end=None if t_end is None else float(t_end),
        max_steps=int(max_steps),
        max_runtime_seconds=max_runtime_seconds,
        output_dir=output_dir,
        network_source=str(network_source),
        save_trajectories=bool(save_trajectories),
        compute_strategy=compute_strategy or MultipleRunConfig().compute_strategy,
        stepper_dt=stepper_dt,
        blended_dt_cle=blended_dt_cle,
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

    blended_dt_pairs = build_blended_dt_pairs(config) if "blended" in methods else []
    network, catalysis_result, restriction, network_metadata = build_shared_objects(config.network_source)
    seeds = make_run_seeds(config.base_seed, config.n_runs)
    output_dir = Path(config.output_dir)
    trajectory_dir = output_dir / config.trajectory_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.save_trajectories:
        trajectory_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(config, methods, seeds, trajectory_dir, blended_dt_pairs)
    strategy = resolve_compute_strategy(config.compute_strategy, task_count=len(tasks))
    apply_cpu_affinity(strategy)

    started_at = perf_counter()
    run_records = run_tasks(network, restriction, tasks, strategy)
    total_wall_runtime = perf_counter() - started_at
    run_records = sorted(
        run_records,
        key=lambda item: (
            int(item["run_order"]),
            int(item["method_order"]),
            _sort_optional_int(item.get("blended_dt_pair_order")),
        ),
    )

    payload = metadata_payload(
        config=config,
        methods=methods,
        network=network,
        catalysis_result=catalysis_result,
        network_metadata=network_metadata,
        seeds=seeds,
        blended_dt_pairs=blended_dt_pairs,
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


def build_shared_objects(network_source: str = "random") -> tuple[ReactionNetworkData, dict, BaseRestriction, dict]:
    source_key, module_name, builder_name = _network_source_spec(network_source)
    module = importlib.import_module(module_name)
    builder = getattr(module, builder_name)
    network, catalysis_result = builder()
    restriction = module.build_food_upper_limit_restriction(network)
    metadata = _network_source_metadata(source_key, module, network, catalysis_result)
    return network, catalysis_result, restriction, metadata


def _network_source_spec(network_source: str) -> tuple[str, str, str]:
    key = str(network_source).lower()
    aliases = {
        "random": ("random", "catalyst_run", "build_random_catalyst_network"),
        "random_catalyst": ("random", "catalyst_run", "build_random_catalyst_network"),
        "catalyst_run": ("random", "catalyst_run", "build_random_catalyst_network"),
        "oscillator": ("oscillator", "oscillator", "build_oscillator_network"),
        "cross": ("cross_catalysis", "cross_catalysis", "build_cross_catalysis_network"),
        "cross_catalysis": ("cross_catalysis", "cross_catalysis", "build_cross_catalysis_network"),
    }
    if key not in aliases:
        raise ValueError(
            "network_source must be one of "
            "'random', 'oscillator', or 'cross_catalysis'"
        )
    return aliases[key]


def _network_source_metadata(
    source_key: str,
    module,
    network: ReactionNetworkData,
    catalysis_result: dict,
) -> dict[str, object]:
    json_ready = getattr(module, "json_ready", catalyst_run.json_ready)
    example_parameters = getattr(module, "example_parameters", lambda: {})()
    catalyst_species_names = getattr(module, "catalyst_species_names", lambda _: [])(network)
    return {
        "network_source": str(source_key),
        "example_parameters": json_ready(example_parameters),
        "catalysis_assignment": json_ready(catalysis_result),
        "catalyst_species_names": json_ready(catalyst_species_names),
        "restriction": _restriction_metadata_from_module(module),
    }


def _restriction_metadata_from_module(module) -> dict[str, object]:
    return {
        "type": "FoodUpperLimitRestriction",
        "food_species": list(getattr(module, "ALPHABET", ())),
        "initial_food_count": _optional_float_attr(module, "INITIAL_FOOD_COUNT"),
        "effective_initial_counts": dict(getattr(module, "INITIAL_COUNTS", {})),
        "food_inflow_rate": _optional_float_attr(module, "FOOD_INFLOW_RATE"),
        "food_max_count": _optional_float_attr(module, "FOOD_MAX_COUNT"),
    }


def _optional_float_attr(module, name: str) -> float | None:
    if not hasattr(module, name):
        return None
    value = getattr(module, name)
    return None if value is None else float(value)


def make_run_seeds(base_seed: int, n_runs: int) -> list[int]:
    if int(n_runs) <= 0:
        raise ValueError("n_runs must be > 0")
    seed_sequence = np.random.SeedSequence(int(base_seed))
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(int(n_runs))
    ]


def build_blended_dt_pairs(config: MultipleRunConfig) -> list[dict[str, object]]:
    dt_cle_values = _normalize_positive_float_values(config.blended_dt_cle, "blended_dt_cle")
    dt_macro_values = _normalize_optional_positive_float_values(config.blended_dt_macro, "blended_dt_macro")
    pairs: list[dict[str, object]] = []
    for cle_order, dt_cle in enumerate(dt_cle_values):
        for macro_order, dt_macro in enumerate(dt_macro_values):
            if dt_macro is not None and dt_macro < dt_cle:
                continue
            pairs.append(
                {
                    "blended_dt_pair_order": len(pairs),
                    "blended_dt_cle_order": int(cle_order),
                    "blended_dt_macro_order": int(macro_order),
                    "blended_dt_cle": float(dt_cle),
                    "blended_dt_macro": None if dt_macro is None else float(dt_macro),
                    "blended_dt_label": _blended_dt_label(dt_cle, dt_macro),
                }
            )
    if not pairs:
        raise ValueError("no valid blended dt pairs; require dt_macro >= dt_cle")
    if len(pairs) > 1 and bool(config.blended_use_reaction_interval_dt):
        raise ValueError(
            "fixed blended dt sweep requires blended_use_reaction_interval_dt=False; "
            "otherwise reaction-interval dt overrides dt_cle/dt_macro"
        )
    return pairs


def _normalize_positive_float_values(value: float | Sequence[float], name: str) -> tuple[float, ...]:
    values = _as_sequence(value)
    normalized = tuple(float(item) for item in values)
    if not normalized:
        raise ValueError(f"{name} must contain at least one value")
    if any(item <= 0.0 for item in normalized):
        raise ValueError(f"{name} values must be > 0")
    return normalized


def _normalize_optional_positive_float_values(
    value: float | Sequence[float | None] | None,
    name: str,
) -> tuple[float | None, ...]:
    if value is None:
        return (None,)
    values = _as_sequence(value)
    normalized = tuple(None if item is None else float(item) for item in values)
    if not normalized:
        raise ValueError(f"{name} must contain at least one value")
    numeric = [item for item in normalized if item is not None]
    if any(item <= 0.0 for item in numeric):
        raise ValueError(f"{name} values must be > 0 when provided")
    return normalized


def _as_sequence(value):
    if isinstance(value, (str, bytes)):
        raise TypeError("dt values must be numeric scalars or numeric sequences")
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _blended_dt_label(dt_cle: float, dt_macro: float | None) -> str:
    macro = "auto" if dt_macro is None else _dt_label_number(dt_macro)
    return f"dtcle_{_dt_label_number(dt_cle)}_dtmacro_{macro}"


def _dt_label_number(value: float) -> str:
    return f"{float(value):.12g}".replace("+", "").replace("-", "m").replace(".", "p")


def _sort_optional_int(value: object) -> int:
    return -1 if value is None else int(value)


def build_tasks(
    config: MultipleRunConfig,
    methods: Sequence[str],
    seeds: Sequence[int],
    trajectory_dir: Path,
    blended_dt_pairs: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    use_blended_dt_labels = len(blended_dt_pairs) > 1
    for run_order, seed in enumerate(seeds):
        for method_order, method in enumerate(methods):
            if method == "blended":
                for pair in blended_dt_pairs:
                    tasks.append(
                        _task_record(
                            config,
                            method=str(method),
                            run_order=run_order,
                            method_order=method_order,
                            seed=seed,
                            trajectory_dir=trajectory_dir,
                            blended_dt_pair=pair,
                            use_blended_dt_label=use_blended_dt_labels,
                        )
                    )
                continue
            tasks.append(
                _task_record(
                    config,
                    method=str(method),
                    run_order=run_order,
                    method_order=method_order,
                    seed=seed,
                    trajectory_dir=trajectory_dir,
                    blended_dt_pair=None,
                    use_blended_dt_label=False,
                )
            )
    return tasks


def _task_record(
    config: MultipleRunConfig,
    *,
    method: str,
    run_order: int,
    method_order: int,
    seed: int,
    trajectory_dir: Path,
    blended_dt_pair: dict[str, object] | None,
    use_blended_dt_label: bool,
) -> dict[str, object]:
    dt_label = None if blended_dt_pair is None else str(blended_dt_pair["blended_dt_label"])
    trajectory_name = (
        f"{method}_{dt_label}_{int(run_order):03d}.npz"
        if use_blended_dt_label and dt_label is not None
        else f"{method}_{int(run_order):03d}.npz"
    )
    return {
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
        "trajectory_name": trajectory_name,
        "stepper_dt": config.stepper_dt,
        "cle_fast_channel_ids": config.cle_fast_channel_ids,
        "hybrid_fast_channel_ids": config.hybrid_fast_channel_ids,
        "blended_i1": float(config.blended_i1),
        "blended_i2": float(config.blended_i2),
        "blended_dt_pair_order": None
        if blended_dt_pair is None
        else int(blended_dt_pair["blended_dt_pair_order"]),
        "blended_dt_cle_order": None
        if blended_dt_pair is None
        else int(blended_dt_pair["blended_dt_cle_order"]),
        "blended_dt_macro_order": None
        if blended_dt_pair is None
        else int(blended_dt_pair["blended_dt_macro_order"]),
        "blended_dt_cle": None if blended_dt_pair is None else float(blended_dt_pair["blended_dt_cle"]),
        "blended_dt_macro": None if blended_dt_pair is None else blended_dt_pair["blended_dt_macro"],
        "blended_dt_label": dt_label,
        "blended_use_reaction_interval_dt": bool(config.blended_use_reaction_interval_dt),
        "blended_reaction_interval_update_steps": int(config.blended_reaction_interval_update_steps),
    }


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
        "blended_dt_pair_order": _json_int_or_none(task["blended_dt_pair_order"]),
        "blended_dt_cle_order": _json_int_or_none(task["blended_dt_cle_order"]),
        "blended_dt_macro_order": _json_int_or_none(task["blended_dt_macro_order"]),
        "blended_dt_cle": _json_float_or_none(task["blended_dt_cle"]),
        "blended_dt_macro": _json_float_or_none(task["blended_dt_macro"]),
        "blended_dt_label": task["blended_dt_label"],
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
        "blended_dt_pair_order": _json_int_or_none(task["blended_dt_pair_order"]),
        "blended_dt_cle_order": _json_int_or_none(task["blended_dt_cle_order"]),
        "blended_dt_macro_order": _json_int_or_none(task["blended_dt_macro_order"]),
        "blended_dt_cle": _json_float_or_none(task["blended_dt_cle"]),
        "blended_dt_macro": _json_float_or_none(task["blended_dt_macro"]),
        "blended_dt_label": task["blended_dt_label"],
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
    blended_dt_cle: float | None,
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
        if blended_dt_cle is None:
            raise ValueError("blended_dt_cle must be set for blended tasks")
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


def _json_int_or_none(value: object) -> int | None:
    return None if value is None else int(value)


def metadata_payload(
    *,
    config: MultipleRunConfig,
    methods: Sequence[str],
    network: ReactionNetworkData,
    catalysis_result: dict,
    network_metadata: dict,
    seeds: Sequence[int],
    blended_dt_pairs: Sequence[dict[str, object]],
    run_records: Sequence[dict[str, object]],
    compute_strategy: ComputeStrategy,
    total_wall_runtime_seconds: float,
) -> dict[str, object]:
    return {
        "experiment": "multi_method_run",
        "generated_by": "examples.multiple_run_core.run_methods",
        "shared": {
            "network_source": network_metadata.get("network_source", "random"),
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
            "example_parameters": network_metadata.get("example_parameters", {}),
            "catalysis_assignment": network_metadata.get("catalysis_assignment", catalysis_result),
            "catalyst_species_names": network_metadata.get("catalyst_species_names", []),
            "restriction": network_metadata.get("restriction", {}),
            "blended_config": {
                "i1": float(config.blended_i1),
                "i2": float(config.blended_i2),
                "requested_dt_cle_values": list(
                    _normalize_positive_float_values(config.blended_dt_cle, "blended_dt_cle")
                ),
                "requested_dt_macro_values": list(
                    _normalize_optional_positive_float_values(config.blended_dt_macro, "blended_dt_macro")
                ),
                "dt_pair_filter": "dt_macro is None or dt_macro >= dt_cle",
                "dt_pairs": [dict(pair) for pair in blended_dt_pairs],
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
    blended_runs = [item for item in runs if item["mode"] == "blended" and item.get("blended_dt_pair_order") is not None]
    if blended_runs:
        print("  by blended dt pair:")
        pair_orders = sorted({int(item["blended_dt_pair_order"]) for item in blended_runs})
        for pair_order in pair_orders:
            selected = [item for item in blended_runs if int(item["blended_dt_pair_order"]) == pair_order]
            if not selected:
                continue
            pair_final_times = np.asarray([float(item["simulation_final_time"]) for item in selected], dtype=float)
            pair_wall_times = np.asarray([float(item["wall_runtime_seconds"]) for item in selected], dtype=float)
            pair_events = np.asarray([int(item["n_events"]) for item in selected], dtype=float)
            dt_cle = selected[0]["blended_dt_cle"]
            dt_macro = selected[0]["blended_dt_macro"]
            print(
                f"    dt_cle={dt_cle}, dt_macro={dt_macro}: "
                f"simulation_time_mean={pair_final_times.mean():.6g}, "
                f"wall_time_mean={pair_wall_times.mean():.3f}, "
                f"events_mean={pair_events.mean():.2f}"
            )
    if shared["save_trajectories"]:
        print(f"  trajectories saved under: {trajectory_dir}")
    print(f"  metadata saved to: {metadata_path}")
