from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SystemState:
    t: float
    x: np.ndarray
    step_count: int = 0
    event_count: int = 0
    partition_state: Any = None

    def copy(self) -> "SystemState":
        return SystemState(
            t=float(self.t),
            x=np.array(self.x, copy=True),
            step_count=int(self.step_count),
            event_count=int(self.event_count),
            partition_state=self.partition_state,
        )

    @classmethod
    def from_x0(cls, x0: np.ndarray, t0: float = 0.0) -> "SystemState":
        return cls(t=float(t0), x=np.array(x0, dtype=float, copy=True))
