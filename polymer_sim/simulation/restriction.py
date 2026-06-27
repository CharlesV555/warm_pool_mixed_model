from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState
from polymer_sim.simulation.stepper import StepResult


@dataclass(slots=True)
class RestrictionContext:
    network: ReactionNetworkData
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
    def __init__(self, target_counts: dict[int, float]):
        self.target_counts = {int(sid): float(value) for sid, value in target_counts.items()}

    def apply(
        self,
        state: SystemState,
        dt: float,
        context: RestrictionContext,
        step_result: StepResult,
    ) -> None:
        for sid, target in self.target_counts.items():
            state.x[sid] = float(target)


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
    network: ReactionNetworkData,
    *,
    food_species: tuple[str, ...] = ("0", "1"),
    food_count: float = 10.0,
) -> RestrictionController:
    food_targets = {network.species_idx(name): float(food_count) for name in food_species}
    return RestrictionController([FoodReplenishmentRestriction(target_counts=food_targets)])


# Backward-compatible alias for older examples/notebooks.
build_hs2014_restriction = build_restriction
