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
    if rebuild:
        network.rebuild_dependency_indices()


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
    for channel_id, strength in enumerate(strengths):
        network.set_catalytic_strength(channel_id, catalyst_sid=catalyst_sid, strength=float(strength), rebuild=False)
    network.rebuild_dependency_indices()
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

    for catalyst_sid, channel_id, strength in zip(catalyst_sids, channel_ids, strengths):
        network.set_catalytic_strength(int(channel_id), catalyst_sid=int(catalyst_sid), strength=float(strength), rebuild=False)
    network.rebuild_dependency_indices()
    return {
        "method": "random_longest_catalysts_to_distinct_channels",
        "catalyst_sids": catalyst_sids,
        "channel_ids": channel_ids,
        "strengths": strengths,
    }
