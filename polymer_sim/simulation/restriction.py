from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from polymer_sim.core.state import SystemState
from polymer_sim.simulation.stepper import StepResult


@dataclass(slots=True)
class RestrictionContext:
    network: Any
    rng: np.random.Generator


class BaseRestriction(ABC):
    @abstractmethod
    def apply(
        self,
        state: SystemState,
        dt: float,
        context: RestrictionContext,
        step_result: StepResult,
    ) -> None:
        raise NotImplementedError

    def metadata(self) -> dict[str, object]:
        return {}


class RestrictionController(BaseRestriction):
    def __init__(self, restrictions: list[BaseRestriction] | tuple[BaseRestriction, ...]):
        self.restrictions = list(restrictions)

    def apply(
        self,
        state: SystemState,
        dt: float,
        context: RestrictionContext,
        step_result: StepResult,
    ) -> None:
        for restriction in self.restrictions:
            restriction.apply(state, dt, context, step_result)

    def metadata(self) -> dict[str, object]:
        merged: dict[str, object] = {}
        for restriction in self.restrictions:
            merged.update(restriction.metadata())
        return merged


class FoodReplenishmentRestriction(BaseRestriction):
    """Keep selected food species fixed at target counts.

    This restriction represents a chemostat-style boundary condition rather
    than a chemical reaction.  It is applied by ``ExperimentRunner`` after a
    stepper advances the state.  For event steppers whose ``step`` fires one
    event, this means food is restored after every event; for larger hybrid or
    PDMP steps it is a step-boundary projection.
    """

    def __init__(self, target_counts: dict[int, float]):
        self.target_counts = {int(sid): float(value) for sid, value in target_counts.items()}
        if any(not np.isfinite(value) or value < 0.0 for value in self.target_counts.values()):
            raise ValueError("food target counts must be finite values >= 0")
        self._species_names: list[str] | None = None

    def apply(
        self,
        state: SystemState,
        dt: float,
        context: RestrictionContext,
        step_result: StepResult,
    ) -> None:
        if self._species_names is None:
            self._species_names = [
                context.network.species_names[int(sid)]
                for sid in self.target_counts
            ]
        for sid, target in self.target_counts.items():
            state.x[int(sid)] = float(target)

    def metadata(self) -> dict[str, object]:
        return {
            "food_replenishment": {
                "mode": "constant",
                "species_ids": [int(sid) for sid in self.target_counts],
                "species_names": list(self._species_names or []),
                "target_counts": [float(value) for value in self.target_counts.values()],
            }
        }


class FoodUpperLimitRestriction(BaseRestriction):
    def __init__(self, max_counts: dict[int, float]):
        self.max_counts = {int(sid): float(value) for sid, value in max_counts.items()}
        if any(not np.isfinite(value) or value < 0.0 for value in self.max_counts.values()):
            raise ValueError("food upper limits must be finite values >= 0")
        self._species_names: list[str] | None = None

    def apply(
        self,
        state: SystemState,
        dt: float,
        context: RestrictionContext,
        step_result: StepResult,
    ) -> None:
        if self._species_names is None:
            self._species_names = [
                context.network.species_names[int(sid)]
                for sid in self.max_counts
            ]
        for sid, maximum in self.max_counts.items():
            state.x[int(sid)] = min(float(state.x[int(sid)]), float(maximum))

    def metadata(self) -> dict[str, object]:
        return {
            "food_upper_limits": {
                "species_ids": [int(sid) for sid in self.max_counts],
                "species_names": list(self._species_names or []),
                "max_counts": [float(value) for value in self.max_counts.values()],
            }
        }


class TrimerOutflowRestriction(BaseRestriction):
    def __init__(
        self,
        rate: float,
        species_ids: np.ndarray | list[int] | tuple[int, ...] | None = None,
        tracked_species_ids: np.ndarray | list[int] | tuple[int, ...] | None = None,
    ):
        self.rate = float(rate)
        self.species_ids = None if species_ids is None else np.asarray(species_ids, dtype=np.int64)
        self.tracked_species_ids = None if tracked_species_ids is None else np.asarray(tracked_species_ids, dtype=np.int64)
        self._history_times: list[float] = []
        self._history_removed: list[list[float]] = []
        self._tracked_species_names: list[str] | None = None

    def apply(
        self,
        state: SystemState,
        dt: float,
        context: RestrictionContext,
        step_result: StepResult,
    ) -> None:
        if dt <= 0.0 or self.rate <= 0.0:
            return

        network = context.network
        species_ids = self.species_ids
        if species_ids is None:
            species_ids = np.flatnonzero(network.lengths == 3).astype(np.int64, copy=False)

        removal_prob = 1.0 - float(np.exp(-self.rate * dt))
        removed_by_species: dict[int, float] = {}
        for sid in species_ids:
            value = float(state.x[int(sid)])
            if value <= 0.0:
                removed_by_species[int(sid)] = 0.0
                continue
            rounded = int(np.rint(value))
            if np.isclose(value, rounded):
                removed = int(context.rng.binomial(rounded, removal_prob))
                state.x[int(sid)] = float(max(rounded - removed, 0))
                removed_by_species[int(sid)] = float(removed)
            else:
                new_value = max(value * (1.0 - removal_prob), 0.0)
                removed_by_species[int(sid)] = float(max(value - new_value, 0.0))
                state.x[int(sid)] = new_value

        if self.tracked_species_ids is not None:
            if self._tracked_species_names is None:
                self._tracked_species_names = [network.species_names[int(sid)] for sid in self.tracked_species_ids]
            self._history_times.append(float(state.t))
            self._history_removed.append(
                [float(removed_by_species.get(int(sid), 0.0)) for sid in self.tracked_species_ids]
            )

    def metadata(self) -> dict[str, object]:
        if self.tracked_species_ids is None:
            return {}
        return {
            "trimer_outflow": {
                "times": list(self._history_times),
                "species_ids": [int(sid) for sid in self.tracked_species_ids.tolist()],
                "species_names": list(self._tracked_species_names or []),
                "removed": [list(row) for row in self._history_removed],
                "rate": float(self.rate),
            }
        }


def build_restriction(
    network: Any,
    *,
    food_species: tuple[str, ...] = ("0", "1"),
    food_count: float = 10.0,
) -> RestrictionController:
    food_targets = {network.species_idx(name): float(food_count) for name in food_species}
    return RestrictionController([FoodReplenishmentRestriction(target_counts=food_targets)])


_FOOD_SUPPLY_MODE_ALIASES = {
    "explicit": "explicit_inflow",
    "explicit_inflow": "explicit_inflow",
    "inflow": "explicit_inflow",
    "hill": "explicit_inflow",
    "hill_inflow": "explicit_inflow",
    "none": "none",
    "off": "none",
    "constant": "constant",
    "fixed": "constant",
    "chemostat": "constant",
    "upper_limit": "upper_limit",
    "cap": "upper_limit",
    "capped": "upper_limit",
}


def normalize_food_supply_mode(mode: str | None) -> str:
    """Normalize the public food-supply mode string.

    Supported canonical modes:
    - ``explicit_inflow``: food supply is represented by formal INFLOW channels.
    - ``constant``: food species are configured as network-level chemostats.
    - ``upper_limit``: food is capped from above but not replenished.
    - ``none``: no food-supply restriction is applied.
    """

    key = "explicit_inflow" if mode is None else str(mode).strip().lower().replace("-", "_")
    try:
        return _FOOD_SUPPLY_MODE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(set(_FOOD_SUPPLY_MODE_ALIASES.values())))
        raise ValueError(f"unsupported food supply mode {mode!r}; expected one of: {valid}") from exc


def build_food_supply_restriction(
    network: Any,
    *,
    mode: str | None,
    food_species: tuple[str | int, ...] = ("0", "1"),
    food_count: float = 10.0,
    food_counts: dict[str | int, float] | None = None,
) -> RestrictionController | None:
    """Build the optional restriction for a selected food-supply mode.

    ``constant`` is implemented by configuring the network itself: food counts
    become fixed parameters in propensity evaluation and are excluded from
    dynamic state deltas/dependency propagation.  The helper returns ``None`` in
    that mode so ``ExperimentRunner`` does not apply a post-step projection and
    does not invalidate stepper caches.
    """

    normalized = normalize_food_supply_mode(mode)
    if normalized in {"explicit_inflow", "none"}:
        return None

    targets = _resolve_species_counts(
        network,
        food_species=food_species,
        default_count=float(food_count),
        counts=food_counts,
    )
    if normalized == "constant":
        if not hasattr(network, "set_chemostat_species"):
            raise TypeError("constant food mode requires a network with set_chemostat_species(...)")
        network.set_chemostat_species(targets)
        return None
    if normalized == "upper_limit":
        return RestrictionController([FoodUpperLimitRestriction(max_counts=targets)])
    raise AssertionError(f"unhandled food supply mode: {normalized}")


def _resolve_species_counts(
    network: Any,
    *,
    food_species: tuple[str | int, ...],
    default_count: float,
    counts: dict[str | int, float] | None,
) -> dict[int, float]:
    if counts is None:
        return {
            _resolve_species_id(network, species): float(default_count)
            for species in food_species
        }
    return {
        _resolve_species_id(network, species): float(value)
        for species, value in counts.items()
    }


def _resolve_species_id(network: Any, species: str | int) -> int:
    if isinstance(species, (int, np.integer)):
        sid = int(species)
    else:
        sid = int(network.species_idx(str(species)))
    if sid < 0 or sid >= int(network.n_species):
        raise IndexError(f"food species id out of range: {sid}")
    return sid


# Backward-compatible alias for older examples/notebooks.
build_hs2014_restriction = build_restriction
