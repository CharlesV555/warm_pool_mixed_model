from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState


@dataclass(slots=True)
class PartitionResult:
    fast_channels: np.ndarray
    slow_channels: np.ndarray
    weights: np.ndarray | None = None


class PartitionStrategy(ABC):
    @abstractmethod
    def partition(
        self,
        network: ReactionNetworkData,
        state: SystemState,
        propensities: np.ndarray | None = None,
    ) -> PartitionResult:
        raise NotImplementedError


class FixedPartitionStrategy(PartitionStrategy):
    def __init__(self, fast_channels: np.ndarray | list[int] | tuple[int, ...]):
        self.fast_channels = np.asarray(fast_channels, dtype=np.int64)

    def partition(
        self,
        network: ReactionNetworkData,
        state: SystemState,
        propensities: np.ndarray | None = None,
    ) -> PartitionResult:
        fast_mask = np.zeros(network.n_channels, dtype=bool)
        if self.fast_channels.size:
            if np.any(self.fast_channels < 0) or np.any(self.fast_channels >= network.n_channels):
                raise IndexError("fast channel id out of range")
            fast_mask[self.fast_channels] = True
        fast_channels = np.flatnonzero(fast_mask).astype(np.int64, copy=False)
        slow_channels = np.flatnonzero(~fast_mask).astype(np.int64, copy=False)
        return PartitionResult(fast_channels=fast_channels, slow_channels=slow_channels)


class BlendingStrategy(ABC):
    @abstractmethod
    def weights(
        self,
        network: ReactionNetworkData,
        state: SystemState,
        propensities: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class NoBlendingStrategy(BlendingStrategy):
    def weights(
        self,
        network: ReactionNetworkData,
        state: SystemState,
        propensities: np.ndarray,
    ) -> np.ndarray:
        return np.ones(network.n_channels, dtype=float)
