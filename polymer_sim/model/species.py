from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class SpeciesSpace:
    species_names: list[str]
    name_to_idx: dict[str, int]
    x0: np.ndarray
    lengths: np.ndarray
    n_monomers: int
    alphabet: tuple[str, ...]
    max_len: int

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    def idx(self, name: str) -> int:
        return self.name_to_idx[name]

    def is_monomer(self, sid: int) -> bool:
        return int(sid) < self.n_monomers

    def with_initial_counts(self, initial_counts: Mapping[str, float] | Sequence[float] | np.ndarray) -> "SpeciesSpace":
        x0 = _make_x0(self.species_names, initial_counts)
        return SpeciesSpace(
            species_names=list(self.species_names),
            name_to_idx=dict(self.name_to_idx),
            x0=x0,
            lengths=np.array(self.lengths, copy=True),
            n_monomers=self.n_monomers,
            alphabet=self.alphabet,
            max_len=self.max_len,
        )


def generate_fixed_species_space(
    alphabet: Sequence[str],
    max_len: int,
    initial_counts: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    *,
    sort_alphabet: bool = True,
) -> SpeciesSpace:
    if max_len < 1:
        raise ValueError("max_len must be >= 1")

    letters = tuple(sorted(alphabet) if sort_alphabet else alphabet)
    if len(letters) == 0:
        raise ValueError("alphabet must not be empty")
    if len(set(letters)) != len(letters):
        raise ValueError("alphabet entries must be unique")
    if any(len(m) != 1 for m in letters):
        raise ValueError("this fixed-space generator expects single-character monomers")

    species_names: list[str] = list(letters)
    for length in range(2, max_len + 1):
        species_names.extend("".join(chars) for chars in product(letters, repeat=length))

    name_to_idx = {name: idx for idx, name in enumerate(species_names)}
    lengths = np.fromiter((len(name) for name in species_names), dtype=np.int16)
    x0 = np.zeros(len(species_names), dtype=float) if initial_counts is None else _make_x0(species_names, initial_counts)

    return SpeciesSpace(
        species_names=species_names,
        name_to_idx=name_to_idx,
        x0=x0,
        lengths=lengths,
        n_monomers=len(letters),
        alphabet=letters,
        max_len=max_len,
    )


def _make_x0(
    species_names: Sequence[str],
    initial_counts: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> np.ndarray:
    if isinstance(initial_counts, Mapping):
        x0 = np.zeros(len(species_names), dtype=float)
        name_to_idx = {name: idx for idx, name in enumerate(species_names)}
        for name, value in initial_counts.items():
            if name not in name_to_idx:
                raise KeyError(f"unknown species in initial_counts: {name}")
            x0[name_to_idx[name]] = float(value)
        return x0

    x0 = np.asarray(initial_counts, dtype=float)
    if x0.shape != (len(species_names),):
        raise ValueError(f"initial_counts must have shape ({len(species_names)},)")
    return np.array(x0, dtype=float, copy=True)
