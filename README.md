
# Structured Polymer Simulation Skeleton

This is a first-pass numerical skeleton for a rule-constrained linear polymer
reaction system. It deliberately avoids a general Species / Reaction / Catalyst
object graph. The hot path uses integer species ids, NumPy arrays, rule lookup
tables, block-contiguous channel ids, and O(1) local state updates.

## Requirement

`pip install numpy pytest`

## Project Layout

```text
polymer_sim/
  core/
    enums.py        Channel block ids.
    network.py      ReactionNetworkData and unified channel/catalysis API.
    state.py        Lightweight SystemState.
  model/
    species.py      Fixed full-space species generator.
    rules.py        Join/split lookup table builder.
    catalysis.py    Dense catalysis block helper.
  simulation/
    propensity.py   Propensity wrapper.
    stepper.py      BaseStepper, SSA, CLE placeholder, Hybrid skeleton.
  partition/
    strategies.py   Fixed partition and blending interfaces.
  recording/
    base.py         Recording 抽象接口。
    trajectory.py   完整轨迹记录与保存/读取。
    summary.py      默认轻量 summary 路径。
    timing.py       低侵入耗时统计。
    plot_single_run.py
    plot_summary.py
  experiment/
    runner.py       run_one and run_many orchestration.
examples/
  minimal_run.py
tests/
docs/
```

## Architecture

```text
SpeciesSpace
  -> ReactionRuleTables
      -> ReactionNetworkData
          -> StepperContext
              -> SSAStepper / CLEStepper / HybridStepper
                  -> SummaryRecorder / TrajectoryRecorder
                      -> ExperimentRunner
```

`ReactionNetworkData` is the numerical core. It stores species arrays, join/split
lookup tables, block-local channel arrays, dense catalysis blocks, global channel
mapping arrays, and sparse dependency indexes.

## Species Data

Fixed full-space species are generated from a monomer alphabet and `max_len`.
The default generator sorts the alphabet, then emits:

1. all monomers first,
2. length 2 polymers,
3. length 3 polymers,
4. and so on through `max_len`,
5. with lexicographic order within each length.

Core arrays:

```text
species_names[i]       sequence string
name_to_idx[name]      stable species id
x0[i]                  initial count / concentration
lengths[i]             sequence length
n_monomers             number of monomer species
```

The convention is fixed: `sid < n_monomers` means the species is a monomer.
There is no `monomer_ids` array.

## Rule Tables

`build_reaction_rule_tables(space)` precomputes:

```text
left_join[m, i]          idx(m + species[i]) or -1
right_join[i, m]         idx(species[i] + m) or -1
split_left_monomer[i]    left terminal monomer or -1
split_left_rest[i]       remaining suffix or -1
split_right_rest[i]      remaining prefix or -1
split_right_monomer[i]   right terminal monomer or -1
```

Only terminal monomer addition and terminal monomer cleavage are represented.
No internal cleavage channels are generated.

## Channel Encoding

Global channel ids are block-contiguous:

```text
LEFT_ADD, RIGHT_ADD, LEFT_SPLIT, RIGHT_SPLIT
```

`channel_block_type[channel_id]` and `channel_local_id[channel_id]` map a global
id back to a block and local row. The local block arrays are:

```text
left_add_target, left_add_monomer, left_add_species
right_add_target, right_add_species, right_add_monomer
left_split_source, left_split_monomer, left_split_rest
right_split_source, right_split_rest, right_split_monomer
```

Length-2 split left/right duplicates are merged into a single `LEFT_SPLIT`
channel with multiplicity 2.

Unified API:

```python
network.get_channel_block(channel_id)
network.get_channel_local_id(channel_id)
network.get_channel_reactants(channel_id)
network.get_channel_products(channel_id)
network.get_channel_main_species(channel_id)
network.apply_channel_update(state, channel_id)
```

## Catalysis

Catalysis is stored internally by reaction block:

```text
cat_left_add[local_channel, catalyst_sid]
cat_right_add[local_channel, catalyst_sid]
cat_left_split[local_channel, catalyst_sid]
cat_right_split[local_channel, catalyst_sid]
```

The public interface is global-channel based:

```python
network.get_channel_catalysts(channel_id)
network.get_catalytic_strength(channel_id, catalyst_sid)
network.get_catalytic_factor(channel_id, state)
```

Current catalytic factor:

```text
1 + sum(strength[channel, catalyst] * x[catalyst])
```

Catalysis only multiplies propensity. It does not change reactants, products,
channel ids, or O(1) state updates.

## Running

Run the minimal example:

```powershell
python examples/minimal_run.py
```

Run tests:

```powershell
python -m pytest
```

## Implemented

- Fixed full-space species generation.
- Join/split lookup tables.
- Block-contiguous channel encoding.
- Unified channel semantic API.
- Dense block catalysis with unified query API.
- O(1) local state update functions.
- Base propensity and catalytic propensity.
- Sparse dependency indexes:
  - `channel_to_species`
  - `species_to_channels`
  - `channel_to_catalysts`
- Lightweight `SystemState`.
- `SSAStepper` with full propensity recompute and linear scan sampling.
- Minimal `CLEStepper`.
- Fixed split `HybridStepper` skeleton.
- `FixedPartitionStrategy` and blending interface.
- Basic recorder, statistics collector, and experiment runner.
- Minimal example and tests.

## Placeholders

- `CLEStepper` is a minimal Euler-Maruyama placeholder.
- `HybridStepper` only supports fixed fast/slow partition semantics.
- `BlendingStrategy` has an interface plus `NoBlendingStrategy`.
- Catalyst assignment strategies live in [catalysis.py](<C:/Users/33973/Documents/New project/polymer_sim/model/catalysis.py>).
- There is no adaptive repartitioning yet.
- There is no PDMP or LP-based partitioning.
- There is no 2016-style blending formula yet.
- There is no particle filtering or inference.
- There is no GPU, JIT, Fenwick tree, alias sampler, or sparse catalysis backend yet.
