from __future__ import annotations

import numpy as np


def dense_catalysis_block(n_local_channels: int, n_species: int) -> np.ndarray:
    return np.zeros((int(n_local_channels), int(n_species)), dtype=float)


def clear_all_catalysis(network, *, rebuild: bool = True) -> None:
    """Clear every catalytic strength in every block."""

    network.cat_left_add.fill(0.0)
    network.cat_right_add.fill(0.0)
    network.cat_left_split.fill(0.0)
    network.cat_right_split.fill(0.0)
    network.cat_outflow.fill(0.0)
    if hasattr(network, "cat_inflow"):
        network.cat_inflow.fill(0.0)
    if rebuild:
        network.rebuild_dependency_indices()
    else:
        if hasattr(network, "_invalidate_catalysis_runtime_caches"):
            network._invalidate_catalysis_runtime_caches()
        if hasattr(network, "channel_has_catalysts"):
            network.channel_has_catalysts.fill(False)
        if hasattr(network, "dependency_indices_dirty"):
            network.dependency_indices_dirty = True


def set_catalytic_strengths_for_channels(
    network,
    channel_ids,
    catalyst_sids,
    strengths,
    *,
    mirror_reverse: bool = True,
    rebuild: bool = True,
) -> None:
    """Assign catalytic strengths to many channels and rebuild indices once.

    Use this for deterministic catalysis construction in examples/tests instead
    of calling ``network.set_catalytic_strength`` inside a large loop.  Reverse
    mirroring uses the network's precomputed reverse-channel cache.
    """

    if hasattr(network, "set_catalytic_strengths"):
        network.set_catalytic_strengths(
            channel_ids,
            catalyst_sids=catalyst_sids,
            strengths=strengths,
            mirror_reverse=mirror_reverse,
            rebuild=rebuild,
        )
        return

    for channel_id, catalyst_sid, strength in _iter_catalysis_assignments(channel_ids, catalyst_sids, strengths):
        network.set_catalytic_strength(
            int(channel_id),
            catalyst_sid=int(catalyst_sid),
            strength=float(strength),
            mirror_reverse=mirror_reverse,
            rebuild=False,
        )
    if rebuild:
        network.rebuild_dependency_indices()
    else:
        network.dependency_indices_dirty = True


def scale_all_catalytic_strengths(network, factor: float, *, rebuild: bool = True) -> None:
    """Multiply every existing catalytic matrix entry by one global factor.

    This is intended for controlled parameter sweeps and profiling.  It does
    not change which catalyst-channel pairs exist; it only rescales their
    strengths.
    """

    scale = float(factor)
    for block in _catalysis_blocks(network):
        block *= scale
    _finalize_global_catalysis_edit(network, rebuild=rebuild)


def set_all_existing_catalytic_strengths(network, strength: float, *, rebuild: bool = True) -> None:
    """Set all currently nonzero catalytic entries to the same value.

    This testing helper preserves the sparse catalytic topology and replaces
    only the coefficient values, which is useful when comparing algorithms
    under identical catalytic graphs.
    """

    value = float(strength)
    for block in _catalysis_blocks(network):
        active = block != 0.0
        if np.any(active):
            block[active] = value
    _finalize_global_catalysis_edit(network, rebuild=rebuild)


def longest_polymer_species_ids(network) -> np.ndarray:
    """Return species ids of the longest polymers in the fixed species space."""

    longest = int(np.max(network.lengths))
    return np.flatnonzero(network.lengths == longest).astype(np.int64, copy=False)


def assign_random_longest_catalyst_to_all_channels(
    network,
    *,
    rng: np.random.Generator,
    log_mean: float = 0.0,
    log_sigma: float = 1.0,
    reset_existing: bool = True,
) -> dict[str, np.ndarray | int | float]:
    """Method 1.

    Randomly choose one catalyst from the longest polymers, then let it catalyze
    every global channel. Each channel receives an independent log-distributed
    catalytic strength sampled as exp(N(log_mean, log_sigma^2)).
    """

    candidates = longest_polymer_species_ids(network)
    if candidates.size == 0:
        raise ValueError("no longest polymer candidates found")

    if reset_existing:
        clear_all_catalysis(network, rebuild=False)

    catalyst_sid = int(rng.choice(candidates))
    strengths = np.exp(rng.normal(loc=float(log_mean), scale=float(log_sigma), size=network.n_channels))
    set_catalytic_strengths_for_channels(
        network,
        np.arange(network.n_channels, dtype=np.int64),
        catalyst_sid,
        strengths,
        mirror_reverse=False,
        rebuild=True,
    )
    return {
        "method": "random_longest_catalyst_to_all_channels",
        "catalyst_sid": catalyst_sid,
        "channel_ids": np.arange(network.n_channels, dtype=np.int64),
        "strengths": strengths,
    }


def assign_random_longest_catalysts_to_distinct_channels(
    network,
    n_catalysts: int,
    *,
    rng: np.random.Generator,
    log_mean: float = 0.0,
    log_sigma: float = 1.0,
    reset_existing: bool = True,
) -> dict[str, np.ndarray | int | float]:
    """Method 2.

    Randomly choose `n_catalysts` distinct catalysts from the longest polymers.
    Then randomly choose the same number of distinct channels and assign one
    catalyst to one different channel. Each assigned catalytic strength is
    sampled independently from a log distribution.
    """

    n = int(n_catalysts)
    if n <= 0:
        raise ValueError("n_catalysts must be > 0")

    candidates = longest_polymer_species_ids(network)
    if n > candidates.size:
        raise ValueError("n_catalysts exceeds number of longest polymer candidates")
    if n > network.n_channels:
        raise ValueError("n_catalysts exceeds number of available channels")

    if reset_existing:
        clear_all_catalysis(network, rebuild=False)

    catalyst_sids = np.asarray(rng.choice(candidates, size=n, replace=False), dtype=np.int64)
    channel_ids = np.asarray(rng.choice(network.n_channels, size=n, replace=False), dtype=np.int64)
    strengths = np.exp(rng.normal(loc=float(log_mean), scale=float(log_sigma), size=n))

    set_catalytic_strengths_for_channels(
        network,
        channel_ids,
        catalyst_sids,
        strengths,
        mirror_reverse=True,
        rebuild=True,
    )
    return {
        "method": "random_longest_catalysts_to_distinct_channels",
        "catalyst_sids": catalyst_sids,
        "channel_ids": channel_ids,
        "strengths": strengths,
    }


def _catalysis_blocks(network) -> tuple[np.ndarray, ...]:
    blocks = [
        network.cat_left_add,
        network.cat_right_add,
        network.cat_left_split,
        network.cat_right_split,
        network.cat_outflow,
    ]
    if hasattr(network, "cat_inflow"):
        blocks.append(network.cat_inflow)
    return tuple(blocks)


def _finalize_global_catalysis_edit(network, *, rebuild: bool) -> None:
    if rebuild:
        network.rebuild_dependency_indices()
        return
    if hasattr(network, "_invalidate_catalysis_runtime_caches"):
        network._invalidate_catalysis_runtime_caches()
    if hasattr(network, "dependency_indices_dirty"):
        network.dependency_indices_dirty = True


def _iter_catalysis_assignments(channel_ids, catalyst_sids, strengths):
    channels = np.asarray(channel_ids, dtype=np.int64)
    if channels.ndim != 1:
        raise ValueError("channel_ids must be a 1D array")
    catalysts = _broadcast_1d(catalyst_sids, channels.size, "catalyst_sids", np.int64)
    values = _broadcast_1d(strengths, channels.size, "strengths", float)
    return zip(channels, catalysts, values)


def _broadcast_1d(value, n: int, name: str, dtype) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full(int(n), arr.item(), dtype=dtype)
    if arr.shape != (int(n),):
        raise ValueError(f"{name} must be scalar or shape ({int(n)},)")
    return np.asarray(arr, dtype=dtype)
