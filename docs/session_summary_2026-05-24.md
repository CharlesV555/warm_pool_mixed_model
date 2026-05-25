# Session Summary - 2026-05-24

This document summarizes the project state after the latest implementation
round.  It is intended to be loaded at the start of the next session.

## 1. Current Goal

The project is now set up to compare exact SSA trajectories with a minimal
Duncan-Erban-Zygalakis-style blended hybrid stepper on the same random
catalytic polymer reaction network.

The current recommended workflow is split into two stages:

1. Batch simulation: generate paired SSA / blended trajectories and metadata.
2. Offline analysis: load saved trajectories, resample them, then perform
   per-mode and combined time-resolved PCA.

PCA analysis has not been implemented in this round.  The batch generation
side is implemented.

## 2. Major New Implementation

### 2.1 Blended Hybrid Stepper

`polymer_sim/simulation/stepper.py` now contains:

- `BlendedHybridConfig`
- `BlendedHybridStepper`
- `estimate_mean_reaction_interval(...)`

The stepper is a minimal runnable blended hybrid method:

- reaction-level beta is computed from reactant copy numbers;
- `beta = 1` means pure SSA jump;
- `beta = 0` means pure CLE;
- mixed channels are split into jump and CLE contributions;
- CLE uses Euler-Maruyama;
- propensities are always routed through `ReactionNetworkData.compute_all_propensities(...)`;
- substrate-saturating catalysis is not duplicated in the stepper;
- negative values from CLE are clipped when configured;
- no Next Reaction Method, thinning, adaptive timestep, or weak trapezoidal
  method is implemented.

`BlendedHybridConfig` currently supports:

- `i1`
- `i2`
- `dt_cle`
- `dt_macro`
- `beta_tol`
- `round_mode`
- `clip_negative`
- `beta_species_mode`
- `use_reaction_interval_dt`
- `reaction_interval_update_steps`
- `reaction_interval_scale`

The reaction interval option estimates the mean reaction interval as
`1 / total_propensity` and refreshes it every configured number of steps.

### 2.2 Public Exports

The new stepper objects are exported from:

- `polymer_sim/simulation/__init__.py`
- `polymer_sim/__init__.py`

Available public names include:

- `BlendedHybridConfig`
- `BlendedHybridStepper`
- `estimate_mean_reaction_interval`

### 2.3 Minimal Blended Example

`examples/blended_hybrid_minimal.py` uses the same reaction network,
catalyst assignment, and food restriction as `examples/catalyst_run.py`.

It saves:

- `examples/blended_hybrid_minimal_trajectory.npz`

The example uses the blended stepper and keeps a short default runtime so it is
safe for local smoke testing.

### 2.4 Paired SSA / Blended Batch Runner

`examples/multiple_run.py` now contains:

- `run_paired_ssa_blended_test(...)`

This function:

- builds one shared random catalytic network;
- builds one shared restriction;
- uses the same `network`, `catalysis_result`, and `restriction` for all runs;
- runs paired SSA and blended simulations;
- gives both modes in the same pair the same seed;
- gives different pairs different seeds;
- runs tasks in serial, thread, or process mode;
- saves full trajectories;
- writes a JSON metadata file with shared configuration and per-run results.

Default paired settings:

- `PAIRED_N_PAIRS = 10`
- `PAIRED_T_END = 0.2`
- `PAIRED_MAX_STEPS = 10_000_000`
- `PAIRED_MAX_RUNTIME_SECONDS = 1800.0`
- output directory: `examples/paired_method_outputs`
- metadata file: `paired_method_metadata.json`
- trajectory names: `ssa_000.npz`, `blended_000.npz`, etc.

`MAIN_RUN_MODE` remains `"batch"` by default.  To run the paired comparison by
executing the script directly, set:

```python
MAIN_RUN_MODE = "paired_method_test"
```

Then run:

```bash
python examples/multiple_run.py
```

Alternatively, call the function directly:

```python
from pathlib import Path

from compute_strategy import ComputeStrategy
from multiple_run import run_paired_ssa_blended_test

run_paired_ssa_blended_test(
    n_pairs=10,
    t_end=0.2,
    max_steps=10_000_000,
    max_runtime_seconds=1800.0,
    output_dir=Path("examples/paired_method_outputs"),
    compute_strategy=ComputeStrategy(backend="process", n_workers=64, use_gpu=False),
)
```

### 2.5 Hardware Strategy Interface

New file:

- `examples/compute_strategy.py`

It defines:

- `ComputeStrategy`
- `resolve_compute_strategy(...)`
- `apply_cpu_affinity(...)`

Supported settings:

- `backend`: `"process"`, `"thread"`, or `"serial"`
- `n_workers`: explicit worker count, or `None` to auto-detect logical CPUs
- `use_gpu`
- `gpu_device`
- `reserve_logical_cpus`
- `cpu_affinity`

Important limitation:

- GPU acceleration is not implemented.
- If `use_gpu=True`, the strategy raises `NotImplementedError`.
- Current steppers are NumPy CPU implementations.

For the target Linux server with 128 logical CPUs, a practical starting
configuration is:

```python
PAIRED_COMPUTE_STRATEGY = ComputeStrategy(
    backend="process",
    n_workers=64,
    use_gpu=False,
)
```

The paired runner caps workers by task count, so 10 pairs produce 20 tasks and
will use at most 20 workers unless more tasks are requested.

## 3. Existing Chemistry / Network State

The current reaction architecture uses `ReactionNetworkData` as the centralized
network and propensity implementation.

Current relevant behavior:

- full polymer species space up to `max_len=5`;
- two-monomer alphabet in current examples: `("A", "B")`;
- 62 species for length 1 through 5;
- formal channel blocks:
  - `LEFT_ADD`
  - `RIGHT_ADD`
  - `LEFT_SPLIT`
  - `RIGHT_SPLIT`
  - `OUTFLOW`
- outflow channels are represented as normal reaction channels;
- non-food species in `catalyst_run.py` have outflow channels;
- food replenishment is handled by `build_restriction(...)`;
- catalyst assignment can mirror catalysis onto reverse reactions;
- catalysis modes:
  - `"linear"`
  - `"substrate_saturating"`

`substrate_saturating` is implemented centrally in `network.py` and applies to
add reactions only.  Outflow does not use the saturating two-substrate formula.

## 4. Current Important Example Files

### `examples/catalyst_run.py`

Builds a random catalytic network with:

- `MAX_LEN = 5`
- `ALPHABET = ("A", "B")`
- `FOOD_COUNT = 100.0`
- `CATALYSIS_MODE = "substrate_saturating"`
- `SATURATION_ALPHA = 0.01`
- non-food outflow enabled

It saves:

- `examples/catalyst_run_trajectory.npz`

### `examples/without_catalyst.py`

Builds a comparable max-length-5 system without catalysts.

It saves:

- `examples/without_catalyst_trajectory.npz`

### `examples/multiple_run.py`

Now supports two workflows:

1. Existing generic multi-run batch path via `run_batch(...)`.
2. New paired SSA/blended comparison via `run_paired_ssa_blended_test(...)`.

The paired function is the intended entry point for the next method-comparison
experiment.

## 5. Recording and Plotting State

Trajectory recording supports:

- full time series;
- species names;
- run metadata;
- channel trigger counts;
- channel event times;
- channel event ids;
- tracked outflow metadata.

Plotting helpers include:

- species time series;
- reaction trigger frequency;
- reaction frequency over time;
- channel propensity time series.

`examples/plot.ipynb` has been extended in earlier steps for:

- `without_catalyst_trajectory.npz`;
- `catalyst_run_trajectory.npz`;
- reaction frequency blocks;
- all-reaction propensity blocks.

## 6. Verification Run in This Round

Static compile check:

```powershell
python -m py_compile examples\compute_strategy.py examples\multiple_run.py
```

Blended stepper tests:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CONDA_NO_PLUGINS='true'
conda run -n warm_pool python -m pytest tests\test_blended_hybrid_stepper.py
```

Result:

- 5 tests passed.
- Pytest emitted a cache permission warning for `.pytest_cache`; this did not
  affect test success.

Smoke run for the paired runner:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CONDA_NO_PLUGINS='true'
conda run -n warm_pool python -c "import sys; from pathlib import Path; sys.path.insert(0, r'examples'); from compute_strategy import ComputeStrategy; from multiple_run import run_paired_ssa_blended_test; run_paired_ssa_blended_test(n_pairs=1, t_end=0.0001, max_steps=200, max_runtime_seconds=10.0, output_dir=Path(r'examples/paired_method_outputs_smoke'), compute_strategy=ComputeStrategy(backend='serial', n_workers=1))"
```

Result:

- generated `ssa_000.npz`;
- generated `blended_000.npz`;
- generated `paired_method_metadata.json`;
- confirmed that trajectory and metadata saving works.

## 7. Known Limitations / Next Steps

The blended stepper is intentionally minimal:

- Euler-Maruyama CLE only;
- beta uses reactant max rule;
- mixed-region propensities are fixed at the start of each small step;
- no Algorithm 2 / Next Reaction Method;
- no Algorithm 3 / thinning;
- no weak trapezoidal method;
- no adaptive timestep beyond the simple mean reaction interval helper;
- negative CLE values are clipped, not handled by a strict boundary method.

The paired batch runner does not yet do PCA.  The next clean step is to add a
separate analysis script or notebook block that:

1. loads `paired_method_metadata.json`;
2. loads the listed `.npz` trajectories;
3. resamples concentrations onto a common time grid;
4. computes PCA within each mode;
5. computes combined PCA across both modes;
6. saves figures and PCA metadata separately from the raw trajectories.
