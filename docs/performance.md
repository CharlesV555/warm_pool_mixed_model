# Complexity and Performance Notes

Let:

```text
M = number of monomers
L = max_len
N = number of species = sum_{ell=1..L} M^ell
C = number of channels
```

## Current Complexity

Species full-space generation:

```text
O(N * average_length)
```

The generator emits all valid strings once and builds `name_to_idx`.

Reaction table construction:

```text
left/right join: O(M * N)
split tables:    O(N)
```

Channel construction:

```text
add channels:   O(M * N)
split channels: O(N)
```

The add channel count is bounded by valid target length. Split channels include
one merged length-2 split plus left/right terminal splits for longer polymers.

SSA step, current version:

```text
full propensity recompute: O(C * avg_catalysts_per_channel)
linear reaction sampling:  O(C)
state update:              O(1)
```

The dependency indexes are built, but SSA does not yet use incremental
propensity updates.

CLE step, current version:

```text
O(F * avg_catalysts_per_channel)
```

where `F` is the number of selected fast channels.

Recorder overhead:

```text
selected species values: O(K) per recorded step
event count update:      O(1) per event
event detail record:     O(reactants + products + catalysts) per event
```

`K` is the number of selected species. The recorder does not store the full
state trajectory by default.

## Main Bottlenecks

- Full propensity recompute in every SSA step.
- Linear scan reaction sampling.
- Dense catalysis blocks when the catalyst relation is sparse.
- Python loops over channels.
- Optional event detail recording can become expensive for very long runs.

## Planned Optimizations

1. Local propensity updates.
   Use `species_to_channels`, `channel_to_species`, and `channel_to_catalysts`
   to recompute only affected channels after an O(1) state update.

2. Faster reaction sampler.
   Replace linear scan with Fenwick tree, segment tree, alias tables, or
   composition-rejection variants depending on update pattern.

3. Sparse catalysis backend.
   Keep the same public API while replacing dense block matrices with CSR/CSC
   blocks for large sparse catalysis maps.

4. Lower overhead recorder.
   Add sampling intervals, ring buffers, and block-level counters for large
   repeat simulations.

5. Hybrid and blending refinement.
   Add a numerically careful fixed hybrid method first, then a 2016-style
   reaction-wise blending implementation behind `BlendingStrategy`.

6. Parallel batch execution.
   Keep per-run RNG independence, then add multiprocessing or job-array friendly
   wrappers around `ExperimentRunner.run_many`.
