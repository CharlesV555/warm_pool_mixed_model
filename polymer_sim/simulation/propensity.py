from __future__ import annotations

import numpy as np

from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState


def compute_all_propensities(
    network: ReactionNetworkData,
    state: SystemState,
    out: np.ndarray | None = None,
) -> np.ndarray:
    return network.compute_all_propensities(state, out=out)
