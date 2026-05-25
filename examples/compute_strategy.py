from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence


_ALLOWED_BACKENDS = {"process", "thread", "serial"}


@dataclass(frozen=True, slots=True)
class ComputeStrategy:
    """Small hardware policy object for batch examples.

    The current simulation kernels are NumPy CPU implementations.  The GPU flag
    is kept explicit so a server config can fail early instead of silently
    pretending to use a GPU.
    """

    backend: str = "process"
    n_workers: int | None = None
    use_gpu: bool = False
    gpu_device: int | None = None
    reserve_logical_cpus: int = 0
    cpu_affinity: tuple[int, ...] | None = None

    def as_metadata(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "n_workers": self.n_workers,
            "use_gpu": self.use_gpu,
            "gpu_device": self.gpu_device,
            "reserve_logical_cpus": self.reserve_logical_cpus,
            "cpu_affinity": None if self.cpu_affinity is None else list(self.cpu_affinity),
            "logical_cpu_count": os.cpu_count(),
        }


def resolve_compute_strategy(
    strategy: ComputeStrategy | None = None,
    *,
    task_count: int | None = None,
) -> ComputeStrategy:
    raw = strategy or ComputeStrategy()
    backend = str(raw.backend).lower()
    if backend not in _ALLOWED_BACKENDS:
        raise ValueError("backend must be 'process', 'thread', or 'serial'")
    if raw.use_gpu:
        raise NotImplementedError(
            "GPU acceleration is not implemented for the current NumPy steppers; "
            "set use_gpu=False for this batch runner."
        )

    logical_cpus = os.cpu_count() or 1
    reserve = max(int(raw.reserve_logical_cpus), 0)
    if raw.n_workers is None:
        n_workers = max(logical_cpus - reserve, 1)
    else:
        n_workers = int(raw.n_workers)
    if n_workers <= 0:
        raise ValueError("n_workers must be > 0 when provided")
    if task_count is not None:
        n_workers = min(n_workers, max(int(task_count), 1))
    if backend == "serial":
        n_workers = 1

    affinity = _normalize_affinity(raw.cpu_affinity)
    return ComputeStrategy(
        backend=backend,
        n_workers=n_workers,
        use_gpu=False,
        gpu_device=raw.gpu_device,
        reserve_logical_cpus=reserve,
        cpu_affinity=affinity,
    )


def apply_cpu_affinity(strategy: ComputeStrategy) -> bool:
    """Apply process CPU affinity on Linux when requested.

    Returns True if affinity was applied.  Child processes inherit the parent
    affinity on Linux.
    """

    if strategy.cpu_affinity is None:
        return False
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("cpu_affinity requires os.sched_setaffinity, which is unavailable on this platform")
    os.sched_setaffinity(0, set(strategy.cpu_affinity))
    return True


def _normalize_affinity(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    affinity = tuple(sorted({int(cpu_id) for cpu_id in value}))
    if not affinity:
        raise ValueError("cpu_affinity must not be empty when provided")
    if affinity[0] < 0:
        raise ValueError("cpu_affinity ids must be nonnegative")
    return affinity
