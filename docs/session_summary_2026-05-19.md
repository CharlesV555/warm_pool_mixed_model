# Session Summary - 2026-05-22

This document summarizes the current project state and is intended to be loaded
at the start of the next session.

## 1. Project Overview

The project is a structured polymer reaction simulation framework. It is not a
general CRN package. The numerical core is built around fixed species spaces,
integer species ids, NumPy arrays, rule lookup tables, and block-contiguous
reaction channel ids.

Current focus:

- fixed full-space linear polymer chemistry
- SSA / CLE / Hybrid stepper comparison
- formal reaction channels, including outflow
- deterministic and random catalysis assignment
- linear and substrate-saturating catalytic propensity modes
- trajectory and summary recording
- offline plotting for species counts, reaction frequencies, and propensities
- single-run and multi-run example workflows

## 2. Current Package Layout

Main package:

- `polymer_sim/core/`
  - `enums.py`: channel block enum definitions
  - `network.py`: `ReactionNetworkData`, channel mapping, propensity logic,
    catalysis API, and state updates
  - `state.py`: lightweight mutable `SystemState`
- `polymer_sim/model/`
  - `species.py`: fixed species-space generation
  - `rules.py`: terminal join/split lookup table generation
  - `catalysis.py`: random catalysis helpers and catalysis clearing
  - `wills_henderson.py`: HS2014 / Wills-Henderson formal example helpers
- `polymer_sim/simulation/`
  - `stepper.py`: `SSAStepper`, minimal `CLEStepper`, and `HybridStepper`
  - `restriction.py`: external restrictions such as food replenishment
  - `propensity.py`: compatibility wrapper around network propensity calls
- `polymer_sim/partition/`
  - `strategies.py`: fixed fast/slow partition and blending interfaces
- `polymer_sim/experiment/`
  - `runner.py`: `ExperimentRunner.run_one(...)` / `run_many(...)`
- `polymer_sim/recording/`
  - `summary.py`: lightweight run summaries
  - `trajectory.py`: full trajectory recording and `.npz` persistence
  - `plot_single_run.py`: trajectory-level plotting helpers
  - `plot_summary.py`: summary-level plotting helpers
  - `timing.py`: timing recorder utilities
- `polymer_sim/analysis/`
  - `raf.py`: closure, RAF, maxRAF, and irrRAF utilities

Examples:

- `examples/minimal_run.py`
- `examples/hs2014_formal_example.py`
- `examples/without_catalyst.py`
- `examples/catalyst_run.py`
- `examples/multiple_run.py`
- `examples/plot.ipynb`

Tests:

- `tests/test_species.py`
- `tests/test_rules_and_channels.py`
- `tests/test_run.py`
- `tests/test_catalysis_assignment.py`
- `tests/test_hs2014_formal_example.py`
- `tests/test_restriction.py`
- `tests/test_recording_plots.py`
- `tests/test_updates_and_propensity.py`

## 3. Data Model and Reaction Network

Species are represented by arrays and integer ids:

- `species_names: list[str]`
- `name_to_idx: dict[str, int]`
- `x0: np.ndarray`
- `lengths: np.ndarray`
- `n_monomers: int`
- `max_len: int`

Convention:

- `sid < n_monomers` means the species is a monomer.

Reaction channels are represented as block-local arrays plus a global
`channel_id -> (block_type, local_id)` mapping. Current channel blocks are:

- `LEFT_ADD`
- `RIGHT_ADD`
- `LEFT_SPLIT`
- `RIGHT_SPLIT`
- `OUTFLOW`

Important rule:

- If a process should compete inside SSA propensity sampling, it should be a
  formal channel.
- If a process is an external forced constraint, it should be a restriction.

Current application of that rule:

- non-food natural dissipation is implemented as formal `OUTFLOW` channels
- food replenishment is implemented as a `restriction`

## 4. Propensity and Catalysis

Centralized propensity calculation lives in:

- `polymer_sim/core/network.py`
- `ReactionNetworkData.compute_base_propensity(...)`
- `ReactionNetworkData.compute_propensity(...)`
- `ReactionNetworkData.compute_all_propensities(...)`

The old simulation-level `propensity.py` is only a wrapper.

Current catalysis modes:

- `catalysis_mode = "linear"`
- `catalysis_mode = "substrate_saturating"`

Default remains:

- `linear`

`substrate_saturating` mode applies only to addition reactions
(`LEFT_ADD`, `RIGHT_ADD`). It does not apply to outflow or split reactions.

For an addition reaction, substrate capacity is:

- if `A != B`: `S_mu = min(X_A, X_B)`
- if `A == B`: `S_mu = floor(X_A / 2)`

Per-catalyst saturated contribution is:

```text
strength * S_mu * X_c / (saturation_alpha * S_mu + X_c)
```

Multiple catalysts are summed per catalyst, not pooled before saturation.

Current saturation parameter:

- `saturation_alpha: float`
- must be `> 0`

Current implementation also mirrors catalytic assignment to reverse reactions
when a reverse channel exists. Outflow has no reverse and is not mirrored.

## 5. Restrictions

Main file:

- `polymer_sim/simulation/restriction.py`

Current public builder:

- `build_restriction(...)`

Backward-compatible alias:

- `build_hs2014_restriction = build_restriction`

Current `build_restriction(...)` behavior:

- creates a `RestrictionController`
- applies `FoodReplenishmentRestriction`
- keeps configured food species at fixed counts after each step

Current food replenishment use:

- HS2014 example: food species `("0", "1")`
- random catalyst examples: food species `("A", "B")`

`TrimerOutflowRestriction` still exists, but the main examples now use formal
`OUTFLOW` channels instead of this post-step approximation.

## 6. Steppers and Runner

Main stepper file:

- `polymer_sim/simulation/stepper.py`

Current steppers:

- `SSAStepper`
  - full propensity recomputation
  - linear scan channel sampling
- `CLEStepper`
  - minimal Euler-Maruyama style implementation
  - uses selected fast channels or all channels
- `HybridStepper`
  - fixed fast/slow partition skeleton
  - slow reactions sampled by SSA
  - fast reactions advanced by CLE increments

Runner:

- `polymer_sim/experiment/runner.py`
- `ExperimentRunner.run_one(...)`

Important `run_one(...)` parameters:

- `network`
- `stepper`
- `t_end`
- `seed`
- `dt`
- `recorder`
- `restriction`
- `partition_strategy`
- `blending_strategy`
- `max_steps`
- `max_runtime_seconds`

Stop reasons:

- `reached_t_end`
- `max_steps`
- `max_runtime_seconds`
- `no_progress`

## 7. Recording and Plotting

Full trajectory recording:

- `polymer_sim/recording/trajectory.py`
- `TrajectoryRecorder`
- `save_trajectory_record(...)`
- `load_trajectory_record(...)`

Current trajectory metadata includes:

- `seed`
- `n_channels`
- `channel_labels`
- `channel_trigger_counts`
- `channel_event_times`
- `channel_event_ids`
- `tracked_outflow`
- `n_steps`
- `n_events`

`tracked_outflow` now follows all formal `OUTFLOW` source species found in
channel labels, not only hard-coded trimers.

Plotting functions:

- `plot_time_series(...)`
- `plot_species_with_outflow(...)`
- `plot_reaction_trigger_frequency(...)`
- `plot_reaction_frequency_over_time(...)`
- `plot_channel_propensity_time_series(...)`

`plot_reaction_frequency_over_time(...)`:

- uses saved `channel_event_times` and `channel_event_ids`
- default `n_bins = 100`
- draws adaptive-width bar plots over simulation time

`plot_channel_propensity_time_series(...)`:

- recomputes propensities along a saved trajectory
- requires the matching `ReactionNetworkData`
- can plot all channels, selected channel ids, selected block type, or top-N

Notebook:

- `examples/plot.ipynb`

Current notebook blocks include:

- load HS2014 trajectory
- species time series
- outflow-aware species plotting
- reaction trigger frequency
- propensity-by-block plotting
- `without_catalyst_trajectory.npz` species and reaction frequency analysis
- `catalyst_run_trajectory.npz` species, total reaction frequency, and
  100-bin reaction-frequency-over-time analysis
- all-reaction propensity-over-time block with comments explaining how to switch
  to another simulation by changing:
  - `PROPENSITY_TRAJECTORY_PATH`
  - `PROPENSITY_BUILDER_PATH`
  - `PROPENSITY_BUILDER_NAME`
  - optional channel/time downsampling parameters

## 8. Example Scripts

### 8.1 `examples/hs2014_formal_example.py`

Purpose:

- HS2014 / Wills-Henderson formal n=3 example
- binary alphabet `("0", "1")`
- deterministic paper-minimal catalysis
- static RAF analysis
- SSA trajectory recording

Current high-level behavior:

- builds n=3 WH network
- assigns paper-minimal catalysis
- computes maxRAF and irrRAFs
- applies food replenishment through `build_restriction(...)`
- non-food outflow is represented by formal `OUTFLOW` channels
- saves `examples/hs2014_formal_example_trajectory.npz`

Current key parameters:

- `T_END = 1000.0`
- `SEED = 123`
- `MAX_STEPS = 100_000_000`
- `MAX_TIMES = 30.0`
- `K_RIGHT_ADD = 0.1`
- `K_NONFOOD_OUTFLOW = 0.5`
- `FOOD_COUNT = 10.0`
- `CATALYSIS_MODE = "substrate_saturating"`
- `SATURATION_ALPHA = 0.01`

### 8.2 `examples/without_catalyst.py`

Purpose:

- max length 5 binary polymer system
- no catalysts
- food species are monomers `("A", "B")`
- all non-food species have formal outflow
- saves full trajectory

Current output:

- `examples/without_catalyst_trajectory.npz`

Key parameters:

- `MAX_LEN = 5`
- `ALPHABET = ("A", "B")`
- `T_END = 1000.0`
- `SEED = 123`
- `MAX_STEPS = 100_000_000`
- `MAX_TIMES = 30.0`
- `K_RIGHT_ADD = 0.1`
- `K_NONFOOD_OUTFLOW = 0.5`
- `FOOD_COUNT = 100.0`
- `CATALYSIS_MODE = "substrate_saturating"`
- `SATURATION_ALPHA = 0.01`

Species count for max length 5 binary full space:

- `2 + 4 + 8 + 16 + 32 = 62`

### 8.3 `examples/catalyst_run.py`

Purpose:

- max length 5 binary polymer system
- random catalysis
- food species are monomers `("A", "B")`
- all non-food species have formal outflow
- food is replenished by `build_restriction(...)`
- saves full trajectory

Current output:

- `examples/catalyst_run_trajectory.npz`

Key parameters:

- `MAX_LEN = 5`
- `ALPHABET = ("A", "B")`
- `T_END = 4.0`
- `SEED = 123`
- `MAX_STEPS = 100_000_000`
- `MAX_TIMES = 600.0`
- `K_LEFT_ADD = 0.1`
- `K_RIGHT_ADD = 0.1`
- `K_LEFT_SPLIT = 0.1`
- `K_RIGHT_SPLIT = 0.1`
- `K_NONFOOD_OUTFLOW = 0.5`
- `FOOD_COUNT = 100.0`
- `CATALYSIS_MODE = "substrate_saturating"`
- `SATURATION_ALPHA = 0.01`
- `CATALYST_SEED = 2026`
- `CATALYST_ASSIGNMENT_MODE = "single_longest_all_channels"`
- `N_RANDOM_CATALYSTS = 16`
- `CATALYST_LOG_MEAN = 0.0`
- `CATALYST_LOG_SIGMA = 1.0`

Supported random catalysis modes:

- `single_longest_all_channels`
- `distinct_longest_channels`

Note:

- The reaction network and catalysis assignment are controlled separately from
  the SSA run seed. `CATALYST_SEED` controls network catalysis; `SEED` controls
  the stochastic simulation trajectory.

### 8.4 `examples/multiple_run.py`

Purpose:

- batch runner based on `catalyst_run.py`
- shared reaction network, catalysis assignment, and restriction
- different simulation seeds per run
- supports parallel execution
- leaves stepper selection configurable for comparing methods

Main shared builder:

- `build_shared_objects() -> (network, catalysis_result, restriction)`

Main batch entry:

- `run_batch(...)`

Configurable top-level parameters:

- `BATCH_SIZE`
- `BASE_RUN_SEED`
- `N_WORKERS`
- `PARALLEL_BACKEND = "process" | "thread" | "serial"`
- `STEPPER_METHOD = "ssa" | "cle" | "hybrid"`
- `STEPPER_DT`
- `CLE_FAST_CHANNEL_IDS`
- `HYBRID_FAST_CHANNEL_IDS`
- `SAVE_TRAJECTORIES`
- `KEEP_CHANNEL_LABELS_IN_SUMMARY`
- output paths

Default output:

- `examples/multiple_run_outputs/multiple_run_summary.json`
- `examples/multiple_run_outputs/multiple_run_metadata.json`

If `SAVE_TRAJECTORIES = True`, per-run trajectories are written under:

- `examples/multiple_run_outputs/trajectories/`

## 9. Static RAF / HS2014 Status

The static n=3 WH example still supports:

- `build_n3_wh_species(...)`
- `build_n3_wh_network(...)`
- `build_n3_wh_reactions(...)`
- `classify_wh_reaction_category(...)`
- `assign_paper_minimal_catalysis(...)`
- `closure(...)`
- `compute_max_raf(...)`
- `enumerate_irr_rafs(...)`

Expected reproduced maxRAF:

- `0+0->00`
- `00+0->000`
- `1+1->11`
- `11+1->111`

Expected irrRAFs:

- `{0+0->00, 00+0->000}`
- `{1+1->11, 11+1->111}`

Still not implemented:

- paper-specific repeated-run dynamic outcome classifier
- formal dynamic categories such as `none`, `only R0`, `only R1`, `both`
- automated statistical report comparing those categories

## 10. Current Public Exports

Top-level `polymer_sim` now exports the main user-facing builders, steppers,
recorders, plotting helpers, and restrictions, including:

- `ReactionNetworkData`
- `SSAStepper`
- `CLEStepper`
- `HybridStepper`
- `ExperimentRunner`
- `TrajectoryRecorder`
- `SummaryRecorder`
- `build_restriction`
- `build_n3_wh_network`
- `assign_paper_minimal_catalysis`
- `assign_random_longest_catalyst_to_all_channels`
- `assign_random_longest_catalysts_to_distinct_channels`
- `plot_time_series`
- `plot_species_with_outflow`
- `plot_reaction_trigger_frequency`
- `plot_reaction_frequency_over_time`
- `plot_channel_propensity_time_series`

## 11. Dependencies and Environment

Project metadata:

- `pyproject.toml`
- project name: `polymer-sim`
- Python: `>=3.10`
- dependencies:
  - `numpy`
  - `matplotlib`

Test config:

- pytest test path: `tests`
- pytest python path: `.`

Known local environment notes:

- The current default shell Python reported missing `numpy` and `matplotlib`.
- Earlier work used a Conda environment named `warm_pool`.
- `pytest-cache-files-*` directories in the workspace can emit permission
  warnings during git/status/search operations.

## 12. Current Validation Status

Recent checks performed:

- `python -m py_compile` passed for recently edited files:
  - `examples/multiple_run.py`
  - `examples/catalyst_run.py`
  - `examples/hs2014_formal_example.py`
  - `polymer_sim/simulation/restriction.py`
  - recording modules
  - selected tests
- `examples/plot.ipynb` JSON parses successfully after appended blocks.

Not recently run in the current default shell:

- full `pytest`
- actual `examples/catalyst_run.py`
- actual `examples/multiple_run.py`

Reason:

- current default shell Python is missing required runtime dependencies.

## 13. Known Cleanup Items

Potential cleanup / next tasks:

- install or select a Python environment with `numpy` and `matplotlib`, then run
  full tests again
- run `examples/catalyst_run.py` to regenerate `catalyst_run_trajectory.npz`
  after the latest recorder/metadata changes
- run `examples/multiple_run.py` and inspect generated summaries
- clean generated `__pycache__` and trajectory artifacts if a clean repository
  snapshot is needed
- consider moving shared example parameters into a small config helper to reduce
  duplication between `catalyst_run.py` and `multiple_run.py`
- add tests for `multiple_run.py` in serial mode
- add a paper-specific repeated-run dynamic outcome classifier
- improve plotting ergonomics for hundreds of reaction channels

## 14. Recommended Next Entry Points

For simulation mechanics:

- `polymer_sim/core/network.py`
- `polymer_sim/simulation/stepper.py`
- `polymer_sim/experiment/runner.py`

For restrictions and food replenishment:

- `polymer_sim/simulation/restriction.py`

For random catalytic systems:

- `examples/catalyst_run.py`
- `examples/multiple_run.py`
- `polymer_sim/model/catalysis.py`

For plotting / offline analysis:

- `examples/plot.ipynb`
- `polymer_sim/recording/plot_single_run.py`
- `polymer_sim/recording/trajectory.py`

For HS2014 / RAF logic:

- `polymer_sim/model/wills_henderson.py`
- `polymer_sim/analysis/raf.py`
