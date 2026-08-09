"""Lightweight CLE sparsity sampler for BlendedHybridStepper diagnostics.

删除指引：
- 核心插入点 1：`BlendedHybridConfig.cle_sparsity_sampling` 显式开启采样。
- 核心插入点 2：`BlendedHybridStepper.__init__` 在开启时创建本 sampler。
- 核心插入点 3：`BlendedHybridStepper._cle_increment` 在 amounts 与 S 都已得到后调用 `sample(...)`。
- 核心插入点 4：`ExperimentRunner.run_one` 在 finalize 后读取 stepper summary metadata。

本文件只负责探测数据结构和绘图，不写 trajectory，也不参与模拟状态推进。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse

from polymer_sim.recording.base import PathLike


@dataclass(slots=True)
class CLESparsitySample:
    sample_index: int
    cle_increment_call: int
    amounts_shape: tuple[int, ...]
    stoichiometry_shape: tuple[int, ...]
    csr_stoichiometry_shape: tuple[int, ...]
    amounts_zero_fraction: float
    amounts_nonzero_fraction: float
    stoichiometry_zero_fraction: float
    stoichiometry_nonzero_fraction: float


@dataclass(slots=True)
class CLESparsitySampler:
    """Sample `amounts` and stoichiometry sparsity every `sample_interval` CLE calls."""

    sample_interval: int = 100
    plot_path: str | None = None
    _call_count: int = 0
    _samples: list[CLESparsitySample] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sample_interval = int(self.sample_interval)
        if self.sample_interval <= 0:
            raise ValueError("sample_interval must be > 0")

    @property
    def call_count(self) -> int:
        return int(self._call_count)

    @property
    def samples(self) -> tuple[CLESparsitySample, ...]:
        return tuple(self._samples)

    def sample(self, amounts: np.ndarray, stoichiometry: np.ndarray, csr_stoichiometry: Any | None = None) -> None:
        self._call_count += 1
        if self._call_count % self.sample_interval != 0:
            return

        amounts_values = np.asarray(amounts)
        stoich_values = np.asarray(stoichiometry)
        csr_stoich = csr_stoichiometry if csr_stoichiometry is not None else csr_matrix(stoich_values)
        if not issparse(csr_stoich):
            csr_stoich = csr_matrix(csr_stoich)

        amounts_total = int(amounts_values.size)
        stoich_total = int(stoich_values.size)
        amounts_nonzero = int(np.count_nonzero(amounts_values)) if amounts_total else 0
        stoich_nonzero = int(np.count_nonzero(stoich_values)) if stoich_total else 0

        self._samples.append(
            CLESparsitySample(
                sample_index=len(self._samples) + 1,
                cle_increment_call=int(self._call_count),
                amounts_shape=tuple(int(value) for value in amounts_values.shape),
                stoichiometry_shape=tuple(int(value) for value in stoich_values.shape),
                csr_stoichiometry_shape=tuple(int(value) for value in csr_stoich.shape),
                amounts_zero_fraction=_zero_fraction(amounts_total, amounts_nonzero),
                amounts_nonzero_fraction=_nonzero_fraction(amounts_total, amounts_nonzero),
                stoichiometry_zero_fraction=_zero_fraction(stoich_total, stoich_nonzero),
                stoichiometry_nonzero_fraction=_nonzero_fraction(stoich_total, stoich_nonzero),
            )
        )

    def summary_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": True,
            "sample_interval": int(self.sample_interval),
            "cle_increment_calls": int(self._call_count),
            "n_samples": int(len(self._samples)),
            "samples": [_sample_to_dict(sample) for sample in self._samples],
        }
        if self._samples and self.plot_path:
            path = save_cle_sparsity_plot(self, self.plot_path)
            payload["plot_path"] = str(path)
        return payload


def save_cle_sparsity_plot(sampler: CLESparsitySampler, path: PathLike) -> Path:
    """Save line plot of amounts and stoichiometry sparsity over CLE call count."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = sampler.samples
    x = np.asarray([sample.cle_increment_call for sample in samples], dtype=float)
    amounts_zero = np.asarray([sample.amounts_zero_fraction for sample in samples], dtype=float)
    stoich_zero = np.asarray([sample.stoichiometry_zero_fraction for sample in samples], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(x, amounts_zero, marker="o", linewidth=1.5, label="amounts zero fraction")
    ax.plot(x, stoich_zero, marker="s", linewidth=1.5, label="S zero fraction")
    ax.set_xlabel("CLE increment call count")
    ax.set_ylabel("zero fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _zero_fraction(total: int, nonzero: int) -> float:
    return 0.0 if int(total) == 0 else float(1.0 - float(nonzero) / float(total))


def _nonzero_fraction(total: int, nonzero: int) -> float:
    return 0.0 if int(total) == 0 else float(nonzero) / float(total)


def _sample_to_dict(sample: CLESparsitySample) -> dict[str, Any]:
    return {
        "sample_index": int(sample.sample_index),
        "cle_increment_call": int(sample.cle_increment_call),
        "amounts_shape": list(sample.amounts_shape),
        "stoichiometry_shape": list(sample.stoichiometry_shape),
        "csr_stoichiometry_shape": list(sample.csr_stoichiometry_shape),
        "amounts_zero_fraction": float(sample.amounts_zero_fraction),
        "amounts_nonzero_fraction": float(sample.amounts_nonzero_fraction),
        "stoichiometry_zero_fraction": float(sample.stoichiometry_zero_fraction),
        "stoichiometry_nonzero_fraction": float(sample.stoichiometry_nonzero_fraction),
    }


__all__ = ["CLESparsitySample", "CLESparsitySampler", "save_cle_sparsity_plot"]
