from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability depends on the local environment.
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent.
    njit = None
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _elementary_propensity(
        channel_id,
        x,
        rate_constants,
        reaction_order,
        reactant1,
        reactant2,
        homo_second_order,
    ):
        rate = rate_constants[channel_id]
        if rate <= 0.0:
            return 0.0
        order = reaction_order[channel_id]
        if order == 0:
            return rate
        sid1 = reactant1[channel_id]
        count1 = x[sid1]
        if count1 < 0.0:
            count1 = 0.0
        if order == 1:
            return rate * count1
        if homo_second_order[channel_id]:
            value = rate * 0.5 * count1 * (count1 - 1.0)
            if value > 0.0:
                return value
            return 0.0
        sid2 = reactant2[channel_id]
        count2 = x[sid2]
        if count2 < 0.0:
            count2 = 0.0
        return rate * count1 * count2

    @njit(cache=True)
    def _elementary_reactants_available(
        channel_id,
        x,
        reaction_order,
        reactant1,
        reactant2,
        homo_second_order,
        tol,
    ):
        order = reaction_order[channel_id]
        if order == 0:
            return True
        sid1 = reactant1[channel_id]
        if order == 1:
            return x[sid1] >= 1.0 - tol
        if homo_second_order[channel_id]:
            return x[sid1] >= 2.0 - tol
        sid2 = reactant2[channel_id]
        return x[sid1] >= 1.0 - tol and x[sid2] >= 1.0 - tol

    @njit(cache=True)
    def collect_affected_channels_numba(
        changed_species,
        fired_channel,
        species_to_channels_indptr,
        species_to_channels_indices,
        marker,
        scratch,
        n_species,
        n_channels,
    ):
        """Collect affected channel ids into scratch and return count.

        Inputs are plain arrays so the PDMP stepper can avoid Python list
        traversal and np.unique in the event hot path.
        """

        count = 0
        for pos in range(changed_species.size):
            sid = changed_species[pos]
            if sid < 0 or sid >= n_species:
                continue
            start = species_to_channels_indptr[sid]
            end = species_to_channels_indptr[sid + 1]
            for ptr in range(start, end):
                channel_id = species_to_channels_indices[ptr]
                if channel_id < 0 or channel_id >= n_channels:
                    continue
                if not marker[channel_id]:
                    marker[channel_id] = True
                    scratch[count] = channel_id
                    count += 1

        if fired_channel >= 0 and fired_channel < n_channels:
            if not marker[fired_channel]:
                marker[fired_channel] = True
                scratch[count] = fired_channel
                count += 1

        for pos in range(count):
            marker[scratch[pos]] = False
        return count

    @njit(cache=True)
    def rebuild_elementary_gillespie_cache_numba(
        x,
        rate_constants,
        reaction_order,
        reactant1,
        reactant2,
        homo_second_order,
        discrete_mask,
        cached_propensities,
        available_propensities,
        available_mask,
        hazard_tol,
    ):
        """Rebuild available RD propensity cache from a current propensity vector."""

        total = 0.0
        bad_count = 0
        n_channels = cached_propensities.size
        for channel_id in range(n_channels):
            available_propensities[channel_id] = 0.0
            available_mask[channel_id] = False

        for channel_id in range(n_channels):
            if not discrete_mask[channel_id]:
                continue
            prop = cached_propensities[channel_id]
            if not np.isfinite(prop):
                bad_count += 1
                prop = 0.0
            if prop < 0.0:
                prop = 0.0
            if _elementary_reactants_available(
                channel_id,
                x,
                reaction_order,
                reactant1,
                reactant2,
                homo_second_order,
                hazard_tol,
            ):
                available_propensities[channel_id] = prop
                if prop > hazard_tol:
                    available_mask[channel_id] = True
                total += prop

        if total < 0.0 or not np.isfinite(total):
            bad_count += 1
            total = 0.0
        return total, bad_count

    @njit(cache=True)
    def refresh_elementary_gillespie_cache_numba(
        affected_channels,
        x,
        rate_constants,
        reaction_order,
        reactant1,
        reactant2,
        homo_second_order,
        discrete_mask,
        cached_propensities,
        available_propensities,
        available_mask,
        previous_total,
        hazard_tol,
    ):
        """Refresh affected elementary propensities and RD cache in one pass."""

        total = previous_total
        if total < 0.0 or not np.isfinite(total):
            total = 0.0
        bad_count = 0
        n_channels = cached_propensities.size

        for pos in range(affected_channels.size):
            channel_id = affected_channels[pos]
            if channel_id < 0 or channel_id >= n_channels:
                continue

            if discrete_mask[channel_id]:
                total -= available_propensities[channel_id]

            prop = _elementary_propensity(
                channel_id,
                x,
                rate_constants,
                reaction_order,
                reactant1,
                reactant2,
                homo_second_order,
            )
            if not np.isfinite(prop):
                bad_count += 1
                prop = 0.0
            if prop < 0.0:
                prop = 0.0
            cached_propensities[channel_id] = prop

            if discrete_mask[channel_id]:
                available = _elementary_reactants_available(
                    channel_id,
                    x,
                    reaction_order,
                    reactant1,
                    reactant2,
                    homo_second_order,
                    hazard_tol,
                )
                if available:
                    available_propensities[channel_id] = prop
                    available_mask[channel_id] = prop > hazard_tol
                    total += prop
                else:
                    available_propensities[channel_id] = 0.0
                    available_mask[channel_id] = False

        if total < 0.0:
            total = 0.0
        if not np.isfinite(total):
            bad_count += 1
            total = 0.0
        return total, bad_count

else:

    def _numba_unavailable(*args, **kwargs):
        raise RuntimeError("numba is not available")

    collect_affected_channels_numba = _numba_unavailable
    rebuild_elementary_gillespie_cache_numba = _numba_unavailable
    refresh_elementary_gillespie_cache_numba = _numba_unavailable
