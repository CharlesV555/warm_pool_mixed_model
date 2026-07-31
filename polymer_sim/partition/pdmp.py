from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np

from polymer_sim.core.elementary import ElementaryMassActionNetwork
from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState


PDMPNetwork = ElementaryMassActionNetwork | ReactionNetworkData
_FAST_NETWORK_CANDIDATE_CACHE: dict[tuple[int, int, int, int], "_FastNetworkCandidateCache"] = {}


@dataclass(slots=True)
class PDMPPartitionResult:
    """Partition data used by PDMP steppers.

    It is intentionally richer than the generic ``PartitionResult`` because
    adaptive PDMP needs species classes, reaction classes, scaling exponents,
    and optional fast-subnetwork metadata.
    """

    continuous_channels: np.ndarray
    discrete_channels: np.ndarray
    continuous_species: np.ndarray
    discrete_species: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    zeta: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    rq_channels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    fast_subnetworks: list["FastSubnetwork"] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def fast_channels(self) -> np.ndarray:
        return self.continuous_channels

    @property
    def slow_channels(self) -> np.ndarray:
        return self.discrete_channels

    def is_within_bounds(self, x: np.ndarray) -> bool:
        values = np.asarray(x, dtype=float)
        return bool(np.all(values >= self.lower_bounds) and np.all(values <= self.upper_bounds))


class PDMPPartitionStrategy:
    def partition(
        self,
        network: PDMPNetwork,
        state: SystemState,
        propensities: np.ndarray | None = None,
    ) -> PDMPPartitionResult:
        raise NotImplementedError


class FixedPDMPPartitionStrategy(PDMPPartitionStrategy):
    def __init__(
        self,
        continuous_channels: Sequence[int] | np.ndarray = (),
        continuous_species: Sequence[int] | np.ndarray | None = None,
    ):
        self.continuous_channels = np.asarray(continuous_channels, dtype=np.int64)
        self.continuous_species = None if continuous_species is None else np.asarray(continuous_species, dtype=np.int64)

    def partition(
        self,
        network: PDMPNetwork,
        state: SystemState,
        propensities: np.ndarray | None = None,
    ) -> PDMPPartitionResult:
        channel_mask = np.zeros(network.n_channels, dtype=bool)
        if self.continuous_channels.size:
            _validate_ids(self.continuous_channels, network.n_channels, "continuous_channels")
            channel_mask[self.continuous_channels] = True
        continuous_channels = np.flatnonzero(channel_mask).astype(np.int64, copy=False)
        discrete_channels = np.flatnonzero(~channel_mask).astype(np.int64, copy=False)

        if self.continuous_species is None:
            species_mask = np.zeros(network.n_species, dtype=bool)
            changed = _changed_species_for_channels(network, continuous_channels)
            species_mask[changed] = True
        else:
            _validate_ids(self.continuous_species, network.n_species, "continuous_species")
            species_mask = np.zeros(network.n_species, dtype=bool)
            species_mask[self.continuous_species] = True
        continuous_species = np.flatnonzero(species_mask).astype(np.int64, copy=False)
        discrete_species = np.flatnonzero(~species_mask).astype(np.int64, copy=False)
        alpha = np.where(species_mask, 1.0, 0.0)
        beta = np.where(channel_mask, 1.0, 0.0)
        zeta = beta + network.nu_minus @ alpha
        lower = np.full(network.n_species, -np.inf, dtype=float)
        upper = np.full(network.n_species, np.inf, dtype=float)
        rq_channels = _rq_channels(network, discrete_channels, continuous_species)
        return PDMPPartitionResult(
            continuous_channels=continuous_channels,
            discrete_channels=discrete_channels,
            continuous_species=continuous_species,
            discrete_species=discrete_species,
            alpha=alpha,
            beta=beta,
            zeta=zeta,
            lower_bounds=lower,
            upper_bounds=upper,
            rq_channels=rq_channels,
            metadata={"method": "fixed"},
        )


@dataclass(slots=True)
class ScalingPDMPConfig:
    """Configuration for the paper-style scaling parameter selection."""

    N0: float = 100.0
    species_exponent_threshold: float = 0.5
    reaction_exponent_threshold: float = 0.5
    bound_factor: float = 4.0
    continuous_copy_number_scale_threshold_mu: float | None = None
    adaptation_scale_threshold_eta: float | None = None
    reaction_relaxation_delta: float | None = None
    min_rate: float = 1e-300
    exponent_floor: float = -12.0
    use_lp: bool = True
    enable_fast_subnetworks: bool = False
    fast_subnetwork_threshold: float = 1.0
    fast_subnetwork_max_size: int = 3

    def __post_init__(self) -> None:
        self.N0 = float(self.N0)
        self.species_exponent_threshold = float(self.species_exponent_threshold)
        self.reaction_exponent_threshold = float(self.reaction_exponent_threshold)
        self.bound_factor = float(self.bound_factor)
        if self.continuous_copy_number_scale_threshold_mu is None:
            self.continuous_copy_number_scale_threshold_mu = self.species_exponent_threshold
        else:
            self.continuous_copy_number_scale_threshold_mu = float(
                self.continuous_copy_number_scale_threshold_mu
            )
            self.species_exponent_threshold = self.continuous_copy_number_scale_threshold_mu
        if self.adaptation_scale_threshold_eta is None:
            self.adaptation_scale_threshold_eta = float(np.log(self.bound_factor) / np.log(self.N0))
        else:
            self.adaptation_scale_threshold_eta = float(self.adaptation_scale_threshold_eta)
            self.bound_factor = float(self.N0 ** self.adaptation_scale_threshold_eta)
        if self.reaction_relaxation_delta is None:
            self.reaction_relaxation_delta = self.reaction_exponent_threshold
        else:
            self.reaction_relaxation_delta = float(self.reaction_relaxation_delta)
            self.reaction_exponent_threshold = self.reaction_relaxation_delta
        self.min_rate = float(self.min_rate)
        self.exponent_floor = float(self.exponent_floor)
        self.use_lp = bool(self.use_lp)
        self.enable_fast_subnetworks = bool(self.enable_fast_subnetworks)
        self.fast_subnetwork_threshold = float(self.fast_subnetwork_threshold)
        self.fast_subnetwork_max_size = int(self.fast_subnetwork_max_size)
        if self.N0 <= 1.0:
            raise ValueError("N0 must be > 1")
        if self.bound_factor <= 1.0:
            raise ValueError("bound_factor must be > 1")
        if self.continuous_copy_number_scale_threshold_mu < 0.0:
            raise ValueError("continuous_copy_number_scale_threshold_mu must be >= 0")
        if self.adaptation_scale_threshold_eta < 0.0:
            raise ValueError("adaptation_scale_threshold_eta must be >= 0")
        if self.reaction_relaxation_delta < 0.0:
            raise ValueError("reaction_relaxation_delta must be >= 0")
        if self.min_rate <= 0.0:
            raise ValueError("min_rate must be > 0")
        if self.fast_subnetwork_max_size <= 0:
            raise ValueError("fast_subnetwork_max_size must be > 0")


class ScalingPDMPPartitionStrategy(PDMPPartitionStrategy):
    """Algorithm-3-style adaptive PDMP partition.

    The LP follows the paper's scaling constraints for elementary mass-action
    reactions.  If SciPy is unavailable, the class falls back to the same
    exponent definitions without solving the LP and records that fallback in
    metadata.
    """

    def __init__(self, config: ScalingPDMPConfig | None = None):
        self.config = config or ScalingPDMPConfig()
        self.fast_selector = FastSubnetworkSelector(
            threshold=self.config.fast_subnetwork_threshold,
            max_size=self.config.fast_subnetwork_max_size,
        )

    def partition(
        self,
        network: PDMPNetwork,
        state: SystemState,
        propensities: np.ndarray | None = None,
    ) -> PDMPPartitionResult:
        # Algorithm 2 "adaptation" body, implemented with Algorithm 3 scaling.
        # Inputs are the current elementary network and current state z.
        # Outputs map to the Algorithm-2 pseudocode as:
        #   RC -> continuous_channels
        #   RD -> discrete_channels
        #   SC -> continuous_species
        #   SD -> discrete_species
        #   bounds -> lower_bounds / upper_bounds
        #   RQ -> rq_channels
        x = np.maximum(np.asarray(state.x, dtype=float), 0.0)

        # Algorithm 3 scale estimation.
        # alpha_i estimates the copy-number scale of species i relative to N0.
        # beta_k estimates the reaction-rate scale of channel k relative to N0.
        alpha_cap = np.log(np.maximum(x, 1.0)) / np.log(self.config.N0)
        beta_cap = np.log(np.maximum(network.rate_constants, self.config.min_rate)) / np.log(self.config.N0)
        beta_cap = np.maximum(beta_cap, self.config.exponent_floor)

        # Optional LP pass for Algorithm 3 consistency constraints.
        alpha, beta, lp_metadata = self._solve_or_fallback(network, alpha_cap, beta_cap)
        zeta = beta + network.nu_minus @ alpha

        # Algorithm 3 thresholding:
        #   mu chooses continuous species;
        #   delta is the relaxed changed-species threshold for continuous
        #   reactions;
        #   eta is used below to build adaptation bounds.
        mu = float(self.config.continuous_copy_number_scale_threshold_mu)
        eta = float(self.config.adaptation_scale_threshold_eta)
        delta = float(self.config.reaction_relaxation_delta)
        continuous_species_mask = alpha >= mu
        changed_by_channel = np.abs(network.nu) > 0.0
        continuous_channel_mask = np.all(~changed_by_channel | (alpha[np.newaxis, :] > delta), axis=1)

        # Optional Algorithm-4-style fast-subnetwork marking.  This is not part
        # of the minimal Algorithm-2 loop; it only modifies RC/SC before the
        # partition result is returned.
        fast_subnetworks: list[FastSubnetwork] = []
        if self.config.enable_fast_subnetworks:
            fast_subnetworks = self.fast_selector.select(network, zeta)
            for subnetwork in fast_subnetworks:
                continuous_channel_mask[subnetwork.channels] = True
                continuous_species_mask[subnetwork.changed_species] = True

        continuous_channels = np.flatnonzero(continuous_channel_mask).astype(np.int64, copy=False)
        discrete_channels = np.flatnonzero(~continuous_channel_mask).astype(np.int64, copy=False)
        continuous_species = np.flatnonzero(continuous_species_mask).astype(np.int64, copy=False)
        discrete_species = np.flatnonzero(~continuous_species_mask).astype(np.int64, copy=False)
        rq_channels = _rq_channels(network, discrete_channels, continuous_species)

        # Algorithm 2 scale-validity bounds checked by
        # PDMPStepper.step(...).  When violated, the stepper calls this
        # adaptation method again.
        lower, upper = _bounds_from_algorithm3(alpha, continuous_species_mask, self.config.N0, mu, eta)
        return PDMPPartitionResult(
            continuous_channels=continuous_channels,
            discrete_channels=discrete_channels,
            continuous_species=continuous_species,
            discrete_species=discrete_species,
            alpha=alpha,
            beta=beta,
            zeta=zeta,
            lower_bounds=lower,
            upper_bounds=upper,
            rq_channels=rq_channels,
            fast_subnetworks=fast_subnetworks,
            metadata={
                "method": "scaling_lp",
                "N0": float(self.config.N0),
                "continuous_copy_number_scale_threshold_mu": mu,
                "adaptation_scale_threshold_eta": eta,
                "reaction_relaxation_delta": delta,
                "species_exponent_threshold": float(self.config.species_exponent_threshold),
                "reaction_exponent_threshold": float(self.config.reaction_exponent_threshold),
                "bound_factor": float(self.config.bound_factor),
                "fast_subnetwork_count": int(len(fast_subnetworks)),
                "rq_channel_count": int(rq_channels.size),
                **lp_metadata,
            },
        )

    def _solve_or_fallback(
        self,
        network: PDMPNetwork,
        alpha_cap: np.ndarray,
        beta_cap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        if not self.config.use_lp:
            return alpha_cap.copy(), beta_cap.copy(), {"lp_used": False, "lp_status": "disabled"}
        try:
            from scipy.optimize import linprog
        except Exception:
            return alpha_cap.copy(), beta_cap.copy(), {"lp_used": False, "lp_status": "scipy_unavailable"}

        n_species = network.n_species
        n_channels = network.n_channels
        n_vars = n_species + n_channels
        objective = -np.ones(n_vars, dtype=float)
        bounds = [(0.0, float(max(cap, 0.0))) for cap in alpha_cap]
        bounds.extend((self.config.exponent_floor, float(cap)) for cap in beta_cap)

        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for channel_id in range(n_channels):
            reactant = network.nu_minus[channel_id]
            changed = np.flatnonzero(network.nu[channel_id] != 0.0)
            for species_id in changed:
                row = np.zeros(n_vars, dtype=float)
                row[:n_species] = reactant
                row[n_species + channel_id] = 1.0
                row[int(species_id)] -= 1.0
                rows.append(row)
                rhs.append(0.0)

        result = linprog(
            objective,
            A_ub=np.vstack(rows) if rows else None,
            b_ub=np.asarray(rhs, dtype=float) if rhs else None,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return alpha_cap.copy(), beta_cap.copy(), {
                "lp_used": False,
                "lp_status": str(result.message),
            }
        values = np.asarray(result.x, dtype=float)
        return values[:n_species], values[n_species:], {
            "lp_used": True,
            "lp_status": "optimal",
            "lp_objective": float(-result.fun),
        }


class LinearCatalysisScalingPDMPPartitionStrategy(ScalingPDMPPartitionStrategy):
    """Scaling partition specialized for direct effective catalysis.

    For ``ReactionNetworkData(catalysis_mode="linear")`` with propensity

        base_r(x) * (1 + sum_c strength_{r,c} * x_c),

    the effective reaction exponent is the maximum of the uncatalyzed term and
    every catalytic term.  For ``catalysis_mode="substrate_saturating"``,
    addition-channel catalytic terms use the implemented saturated contribution

        strength * S_r(x) * x_c / (saturation_alpha * S_r(x) + x_c),

    where ``S_r`` is the channel substrate capacity.  Its scaling exponent is
    the minimum of linear branches, so zeta is computed from that exact minimum
    while the LP uses a selected linear upper branch for each saturated term.
    """

    def partition(
        self,
        network: PDMPNetwork,
        state: SystemState,
        propensities: np.ndarray | None = None,
    ) -> PDMPPartitionResult:
        self._validate_linear_catalysis_network(network)
        x = np.maximum(np.asarray(state.x, dtype=float), 0.0)

        alpha_cap = np.log(np.maximum(x, 1.0)) / np.log(self.config.N0)
        beta_cap = np.log(np.maximum(network.rate_constants, self.config.min_rate)) / np.log(self.config.N0)
        beta_cap = np.maximum(beta_cap, self.config.exponent_floor)

        alpha, beta, lp_metadata = self._solve_or_fallback_linear_catalysis(network, alpha_cap, beta_cap)
        zeta = self._effective_linear_catalysis_zeta(network, alpha, beta)

        mu = float(self.config.continuous_copy_number_scale_threshold_mu)
        eta = float(self.config.adaptation_scale_threshold_eta)
        delta = float(self.config.reaction_relaxation_delta)
        continuous_species_mask = alpha >= mu
        changed_by_channel = np.abs(network.nu) > 0.0
        continuous_channel_mask = np.all(~changed_by_channel | (alpha[np.newaxis, :] > delta), axis=1)

        fast_subnetworks: list[FastSubnetwork] = []
        if self.config.enable_fast_subnetworks:
            fast_subnetworks = self.fast_selector.select(network, zeta)
            for subnetwork in fast_subnetworks:
                continuous_channel_mask[subnetwork.channels] = True
                continuous_species_mask[subnetwork.changed_species] = True

        continuous_channels = np.flatnonzero(continuous_channel_mask).astype(np.int64, copy=False)
        discrete_channels = np.flatnonzero(~continuous_channel_mask).astype(np.int64, copy=False)
        continuous_species = np.flatnonzero(continuous_species_mask).astype(np.int64, copy=False)
        discrete_species = np.flatnonzero(~continuous_species_mask).astype(np.int64, copy=False)
        rq_channels = _rq_channels(network, discrete_channels, continuous_species)

        lower, upper = _bounds_from_algorithm3(alpha, continuous_species_mask, self.config.N0, mu, eta)
        catalytic_terms = [
            term
            for channel_id in range(network.n_channels)
            for term in self._linear_catalysis_terms(network, channel_id, alpha_cap)
        ]
        saturated_term_count = sum(1 for term in catalytic_terms if len(term.branches) > 1)
        selected_upper_branch_count = sum(
            1
            for term in catalytic_terms
            if len(term.branches) > 1 and len(term.lp_branches) == 1
        )
        catalysis_mode = getattr(network, "catalysis_mode", "mass_action")
        saturation_alpha = getattr(network, "saturation_alpha", None)
        saturation_alpha_exponent = (
            None
            if saturation_alpha is None
            else float(np.log(max(float(saturation_alpha), self.config.min_rate)) / np.log(self.config.N0))
        )
        return PDMPPartitionResult(
            continuous_channels=continuous_channels,
            discrete_channels=discrete_channels,
            continuous_species=continuous_species,
            discrete_species=discrete_species,
            alpha=alpha,
            beta=beta,
            zeta=zeta,
            lower_bounds=lower,
            upper_bounds=upper,
            rq_channels=rq_channels,
            fast_subnetworks=fast_subnetworks,
            metadata={
                "method": "linear_catalysis_scaling_lp",
                "N0": float(self.config.N0),
                "continuous_copy_number_scale_threshold_mu": mu,
                "adaptation_scale_threshold_eta": eta,
                "reaction_relaxation_delta": delta,
                "species_exponent_threshold": float(self.config.species_exponent_threshold),
                "reaction_exponent_threshold": float(self.config.reaction_exponent_threshold),
                "bound_factor": float(self.config.bound_factor),
                "fast_subnetwork_count": int(len(fast_subnetworks)),
                "rq_channel_count": int(rq_channels.size),
                "catalysis_mode": str(catalysis_mode),
                "saturation_alpha": None if saturation_alpha is None else float(saturation_alpha),
                "saturation_alpha_exponent": saturation_alpha_exponent,
                "linear_catalysis_term_count": int(len(catalytic_terms)),
                "saturating_catalysis_term_count": int(saturated_term_count),
                "saturating_selected_upper_branch_count": int(selected_upper_branch_count),
                **lp_metadata,
            },
        )

    def _validate_linear_catalysis_network(self, network: PDMPNetwork) -> None:
        if not isinstance(network, ReactionNetworkData):
            return
        if network.catalysis_mode not in {"linear", "substrate_saturating"}:
            raise ValueError(
                "LinearCatalysisScalingPDMPPartitionStrategy requires "
                "ReactionNetworkData(catalysis_mode='linear' or 'substrate_saturating')"
            )

    def _solve_or_fallback_linear_catalysis(
        self,
        network: PDMPNetwork,
        alpha_cap: np.ndarray,
        beta_cap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        if not self.config.use_lp:
            return alpha_cap.copy(), beta_cap.copy(), {"lp_used": False, "lp_status": "disabled"}
        try:
            from scipy.optimize import linprog
        except Exception:
            return alpha_cap.copy(), beta_cap.copy(), {"lp_used": False, "lp_status": "scipy_unavailable"}

        n_species = network.n_species
        n_channels = network.n_channels
        n_vars = n_species + n_channels
        objective = -np.ones(n_vars, dtype=float)
        bounds = [(0.0, float(max(cap, 0.0))) for cap in alpha_cap]
        bounds.extend((self.config.exponent_floor, float(cap)) for cap in beta_cap)

        rows: list[np.ndarray] = []
        rhs: list[float] = []
        nu = network.nu
        nu_minus = network.nu_minus
        for channel_id in range(n_channels):
            reactant = nu_minus[channel_id]
            changed = np.flatnonzero(nu[channel_id] != 0.0)
            terms = self._linear_catalysis_terms(network, channel_id, alpha_cap)
            for species_id in changed:
                base_row = np.zeros(n_vars, dtype=float)
                base_row[:n_species] = reactant
                base_row[n_species + channel_id] = 1.0
                base_row[int(species_id)] -= 1.0
                rows.append(base_row)
                rhs.append(0.0)

                for term in terms:
                    for branch in term.lp_branches:
                        row = np.array(base_row, dtype=float, copy=True)
                        for species, coefficient in branch.species_coefficients:
                            row[int(species)] += float(coefficient)
                        rows.append(row)
                        rhs.append(-float(branch.constant_exponent))

        result = linprog(
            objective,
            A_ub=np.vstack(rows) if rows else None,
            b_ub=np.asarray(rhs, dtype=float) if rhs else None,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return alpha_cap.copy(), beta_cap.copy(), {
                "lp_used": False,
                "lp_status": str(result.message),
            }
        values = np.asarray(result.x, dtype=float)
        return values[:n_species], values[n_species:], {
            "lp_used": True,
            "lp_status": "optimal",
            "lp_objective": float(-result.fun),
        }

    def _effective_linear_catalysis_zeta(
        self,
        network: PDMPNetwork,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        nu_minus = network.nu_minus
        alpha_values = np.asarray(alpha, dtype=float)
        base = np.asarray(beta, dtype=float) + nu_minus @ alpha_values
        zeta = np.array(base, dtype=float, copy=True)
        for channel_id in range(network.n_channels):
            terms = self._linear_catalysis_terms(network, channel_id, alpha_values)
            for term in terms:
                term_exponent = min(branch.value(alpha_values) for branch in term.branches)
                value = base[int(channel_id)] + float(term_exponent)
                if value > zeta[int(channel_id)]:
                    zeta[int(channel_id)] = value
        return zeta

    def _linear_catalysis_terms(
        self,
        network: PDMPNetwork,
        channel_id: int,
        alpha_reference: np.ndarray,
    ) -> list["_CatalyticScalingTerm"]:
        if not isinstance(network, ReactionNetworkData):
            return []
        if network.get_channel_block_name(channel_id) == "INFLOW":
            return []
        catalysts = network.get_channel_catalysts(channel_id)
        if catalysts.size == 0:
            return []
        terms: list[_CatalyticScalingTerm] = []
        for catalyst_sid in catalysts:
            strength = network.get_catalytic_strength(channel_id, int(catalyst_sid))
            if strength <= 0.0:
                continue
            strength_exponent = np.log(max(float(strength), self.config.min_rate)) / np.log(self.config.N0)
            strength_exponent = max(float(strength_exponent), float(self.config.exponent_floor))
            if self._uses_saturating_catalysis_scaling(network, channel_id):
                branches = self._saturating_catalysis_branches(
                    network,
                    channel_id,
                    int(catalyst_sid),
                    strength_exponent,
                )
            else:
                branches = [
                    _CatalyticScalingBranch(
                        constant_exponent=float(strength_exponent),
                        species_coefficients=((int(catalyst_sid), 1.0),),
                    )
                ]
            lp_branch = min(branches, key=lambda branch: branch.value(alpha_reference))
            terms.append(
                _CatalyticScalingTerm(
                    catalyst_sid=int(catalyst_sid),
                    branches=tuple(branches),
                    lp_branches=(lp_branch,),
                )
            )
        return terms

    def _uses_saturating_catalysis_scaling(
        self,
        network: ReactionNetworkData,
        channel_id: int,
    ) -> bool:
        return (
            network.catalysis_mode == "substrate_saturating"
            and network.get_channel_block_name(channel_id) in {"LEFT_ADD", "RIGHT_ADD"}
        )

    def _saturating_catalysis_branches(
        self,
        network: ReactionNetworkData,
        channel_id: int,
        catalyst_sid: int,
        strength_exponent: float,
    ) -> list["_CatalyticScalingBranch"]:
        saturation_alpha_exponent = np.log(
            max(float(network.saturation_alpha), self.config.min_rate)
        ) / np.log(self.config.N0)
        branches = [
            _CatalyticScalingBranch(
                constant_exponent=float(strength_exponent - saturation_alpha_exponent),
                species_coefficients=((int(catalyst_sid), 1.0),),
            )
        ]
        for species_id in self._saturating_substrate_species(network, channel_id):
            branches.append(
                _CatalyticScalingBranch(
                    constant_exponent=float(strength_exponent),
                    species_coefficients=((int(species_id), 1.0),),
                )
            )
        return branches

    def _saturating_substrate_species(
        self,
        network: ReactionNetworkData,
        channel_id: int,
    ) -> tuple[int, ...]:
        block_name = network.get_channel_block_name(channel_id)
        local_id = network.get_channel_local_id(channel_id)
        if block_name == "LEFT_ADD":
            species = (
                int(network.left_add_monomer[local_id]),
                int(network.left_add_species[local_id]),
            )
        elif block_name == "RIGHT_ADD":
            species = (
                int(network.right_add_species[local_id]),
                int(network.right_add_monomer[local_id]),
            )
        else:
            species = ()
        return tuple(sorted(set(species)))


@dataclass(frozen=True, slots=True)
class _CatalyticScalingBranch:
    constant_exponent: float
    species_coefficients: tuple[tuple[int, float], ...]

    def value(self, alpha: np.ndarray) -> float:
        return float(
            self.constant_exponent
            + sum(
                float(coefficient) * float(alpha[int(species)])
                for species, coefficient in self.species_coefficients
            )
        )


@dataclass(frozen=True, slots=True)
class _CatalyticScalingTerm:
    catalyst_sid: int
    branches: tuple[_CatalyticScalingBranch, ...]
    lp_branches: tuple[_CatalyticScalingBranch, ...]


@dataclass(slots=True)
class FastSubnetwork:
    channels: np.ndarray
    changed_species: np.ndarray
    catalyst_species: np.ndarray
    surrounding_channels: np.ndarray
    delta_zeta: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _FastNetworkCandidateFeature:
    channels: tuple[int, ...]
    channels_array: np.ndarray
    changed_species: np.ndarray
    dependency_species: np.ndarray
    catalyst_species: np.ndarray
    touch_species: np.ndarray
    surrounding_channels: np.ndarray


@dataclass(slots=True)
class _FastNetworkCandidateCache:
    network_id: int
    n_channels: int
    n_species: int
    max_size: int
    candidates: list[tuple[int, ...]]
    features: list[_FastNetworkCandidateFeature]
    reaction_graph: list[set[int]]
    components: list[list[int]]


@dataclass(slots=True)
class FiniteMarkovConfig:
    """Configuration for finite Markov-process checks on fast subnetworks."""

    max_states: int = 512
    max_total_internal_count: int = 10_000
    min_transition_rate: float = 1e-12
    round_mode: str = "nearest"
    require_irreducible: bool = True
    count_single_state_as_averageable: bool = False

    def __post_init__(self) -> None:
        self.max_states = int(self.max_states)
        self.max_total_internal_count = int(self.max_total_internal_count)
        self.min_transition_rate = float(self.min_transition_rate)
        self.round_mode = str(self.round_mode).lower()
        self.require_irreducible = bool(self.require_irreducible)
        self.count_single_state_as_averageable = bool(self.count_single_state_as_averageable)
        if self.max_states <= 0:
            raise ValueError("max_states must be > 0")
        if self.max_total_internal_count < 0:
            raise ValueError("max_total_internal_count must be >= 0")
        if self.min_transition_rate < 0.0:
            raise ValueError("min_transition_rate must be >= 0")
        if self.round_mode not in {"nearest", "floor", "ceil"}:
            raise ValueError("round_mode must be 'nearest', 'floor', or 'ceil'")


@dataclass(slots=True)
class FiniteMarkovSubnetworkResult:
    """Finite-state CTMC analysis result for one selected fast subnetwork."""

    channels: np.ndarray
    changed_species: np.ndarray
    finite: bool
    averageable: bool
    reason: str
    state_count: int
    transition_count: int
    closed_class_count: int
    irreducible: bool
    stationary_distribution: np.ndarray | None = None
    stationary_mean: np.ndarray | None = None


@dataclass(slots=True)
class FastNetworkReport:
    """Whole-network fast-subnetwork report at one simulation event count."""

    event_count: int
    step_count: int
    simulation_time: float
    total_reaction_count: int
    candidate_subnetwork_count: int
    found_subnetwork_count: int
    found_subnetwork_reaction_count: int
    finite_markov_subnetwork_count: int
    finite_markov_reaction_count: int
    averageable_subnetwork_count: int
    averageable_reaction_count: int
    subnetwork_results: list[FiniteMarkovSubnetworkResult] = field(default_factory=list)


class FastSubnetworkSelector:
    """Algorithm-4-style fast subnetwork selector.

    This implements candidate construction, timescale-separation scoring, and
    greedy disjoint selection.  The averaging step is deliberately exposed as a
    later engine; selected subnetworks are reported in the partition result.
    """

    def __init__(self, *, threshold: float = 1.0, max_size: int = 3):
        self.threshold = float(threshold)
        self.max_size = int(max_size)
        self._candidate_cache: dict[tuple[int, int, int, int], _FastNetworkCandidateCache] = {}

    def select(self, network: PDMPNetwork, zeta: np.ndarray) -> list[FastSubnetwork]:
        cache = self._candidate_cache_for(network)
        scored = [self._score_candidate_feature(feature, zeta) for feature in cache.features]
        scored = [item for item in scored if item.delta_zeta >= self.threshold]
        scored.sort(key=lambda item: (item.delta_zeta, item.channels.size), reverse=True)
        selected: list[FastSubnetwork] = []
        used_species: set[int] = set()
        for item in scored:
            species = set(int(sid) for sid in item.changed_species)
            if species & used_species:
                continue
            selected.append(item)
            used_species.update(species)
        return selected

    def _candidate_cache_for(self, network: PDMPNetwork) -> _FastNetworkCandidateCache:
        key = (id(network), int(network.n_channels), int(network.n_species), int(self.max_size))
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        cached = _get_fast_network_candidate_cache(network, max_size=self.max_size)
        self._candidate_cache[key] = cached
        return cached

    def _candidate_channel_sets(self, network: PDMPNetwork) -> list[tuple[int, ...]]:
        return list(self._candidate_cache_for(network).candidates)

    def _score_candidate(
        self,
        network: PDMPNetwork,
        channels: np.ndarray,
        zeta: np.ndarray,
    ) -> FastSubnetwork:
        feature = _fast_network_candidate_feature_from_channels(
            network,
            tuple(int(channel_id) for channel_id in np.asarray(channels, dtype=np.int64)),
        )
        return self._score_candidate_feature(feature, zeta)

    def _score_candidate_feature(
        self,
        feature: _FastNetworkCandidateFeature,
        zeta: np.ndarray,
    ) -> FastSubnetwork:
        channels = feature.channels_array
        surrounding_a = feature.surrounding_channels
        inside = float(np.min(zeta[channels])) if channels.size else -np.inf
        outside = 0.0
        if surrounding_a.size:
            outside = max(outside, float(np.max(zeta[surrounding_a])))
        return FastSubnetwork(
            channels=np.asarray(channels, dtype=np.int64),
            changed_species=feature.changed_species,
            catalyst_species=feature.catalyst_species,
            surrounding_channels=surrounding_a,
            delta_zeta=float(inside - outside),
            metadata={"inside_min_zeta": inside, "outside_max_zeta": outside},
        )


class FiniteMarkovSubnetworkAnalyzer:
    """Finite Markov-process method for selected fast subnetworks.

    For a candidate fast subnetwork, species outside ``changed_species`` are
    frozen at the current simulation state.  The analyzer enumerates all
    reachable integer states of ``changed_species`` under the candidate
    reactions, builds the finite CTMC generator when enumeration closes, and
    accepts the subnetwork for averaging only when the reachable CTMC has a
    unique stationary distribution under the configured checks.
    """

    def __init__(self, config: FiniteMarkovConfig | None = None):
        self.config = config or FiniteMarkovConfig()

    def analyze(
        self,
        network: PDMPNetwork,
        state: SystemState,
        subnetwork: FastSubnetwork,
    ) -> FiniteMarkovSubnetworkResult:
        channels = np.asarray(subnetwork.channels, dtype=np.int64)
        changed_species = np.asarray(subnetwork.changed_species, dtype=np.int64)
        if channels.size == 0 or changed_species.size == 0:
            return FiniteMarkovSubnetworkResult(
                channels=channels,
                changed_species=changed_species,
                finite=True,
                averageable=False,
                reason="empty subnetwork",
                state_count=0,
                transition_count=0,
                closed_class_count=0,
                irreducible=False,
            )

        initial = self._initial_internal_state(state.x[changed_species])
        if np.any(initial < 0):
            return self._failed(channels, changed_species, "negative initial internal state")

        states, transitions, reason = self._enumerate_reachable_states(network, state, channels, changed_species, initial)
        finite = reason == "finite"
        if not finite:
            return FiniteMarkovSubnetworkResult(
                channels=channels,
                changed_species=changed_species,
                finite=False,
                averageable=False,
                reason=reason,
                state_count=len(states),
                transition_count=len(transitions),
                closed_class_count=0,
                irreducible=False,
            )

        graph = _directed_graph_from_transitions(len(states), transitions)
        components = _strongly_connected_components(graph)
        comp_id = _component_index(components, len(states))
        closed = _closed_component_ids(graph, comp_id)
        irreducible = len(components) == 1
        transition_count = len(transitions)
        averageable = bool(
            finite
            and (transition_count > 0 or self.config.count_single_state_as_averageable)
            and (
                irreducible
                if self.config.require_irreducible
                else len(closed) == 1
            )
        )

        stationary = None
        stationary_mean = None
        reason_out = "averageable" if averageable else "finite but not irreducible"
        if averageable:
            generator = _generator_from_transitions(len(states), transitions)
            stationary = _stationary_distribution(generator)
            state_matrix = np.asarray(states, dtype=float)
            stationary_mean = stationary @ state_matrix
        elif transition_count == 0:
            reason_out = "finite but no active internal transitions"

        return FiniteMarkovSubnetworkResult(
            channels=channels,
            changed_species=changed_species,
            finite=finite,
            averageable=averageable,
            reason=reason_out,
            state_count=len(states),
            transition_count=transition_count,
            closed_class_count=len(closed),
            irreducible=irreducible,
            stationary_distribution=stationary,
            stationary_mean=stationary_mean,
        )

    def _enumerate_reachable_states(
        self,
        network: PDMPNetwork,
        state: SystemState,
        channels: np.ndarray,
        changed_species: np.ndarray,
        initial: np.ndarray,
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, int, float, int]], str]:
        frozen_x = np.maximum(np.asarray(state.x, dtype=float), 0.0).copy()
        deltas = np.rint(network.nu[channels][:, changed_species]).astype(np.int64, copy=False)
        states: list[tuple[int, ...]] = [tuple(int(v) for v in initial)]
        state_to_index = {states[0]: 0}
        queue: deque[tuple[int, ...]] = deque([states[0]])
        transitions: list[tuple[int, int, float, int]] = []

        while queue:
            key = queue.popleft()
            source_index = state_to_index[key]
            full_x = frozen_x.copy()
            full_x[changed_species] = np.asarray(key, dtype=float)
            propensities = network.compute_propensities_for_channels(
                channels,
                SystemState(t=float(state.t), x=full_x),
            )
            active = np.flatnonzero(propensities > self.config.min_transition_rate)
            for local in active:
                target = np.asarray(key, dtype=np.int64) + deltas[int(local)]
                if np.any(target < 0):
                    return states, transitions, "negative target state"
                if int(np.sum(target)) > self.config.max_total_internal_count:
                    return states, transitions, "state-space total-count limit exceeded"
                target_key = tuple(int(v) for v in target)
                target_index = state_to_index.get(target_key)
                if target_index is None:
                    if len(states) >= self.config.max_states:
                        return states, transitions, "state-space count limit exceeded"
                    target_index = len(states)
                    state_to_index[target_key] = target_index
                    states.append(target_key)
                    queue.append(target_key)
                transitions.append((source_index, target_index, float(propensities[int(local)]), int(channels[int(local)])))
        return states, transitions, "finite"

    def _initial_internal_state(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if self.config.round_mode == "nearest":
            rounded = np.rint(x)
        elif self.config.round_mode == "floor":
            rounded = np.floor(x)
        else:
            rounded = np.ceil(x)
        return np.maximum(rounded, 0.0).astype(np.int64, copy=False)

    def _failed(
        self,
        channels: np.ndarray,
        changed_species: np.ndarray,
        reason: str,
    ) -> FiniteMarkovSubnetworkResult:
        return FiniteMarkovSubnetworkResult(
            channels=np.asarray(channels, dtype=np.int64),
            changed_species=np.asarray(changed_species, dtype=np.int64),
            finite=False,
            averageable=False,
            reason=str(reason),
            state_count=0,
            transition_count=0,
            closed_class_count=0,
            irreducible=False,
        )


def analyze_fast_network(
    network: PDMPNetwork,
    state: SystemState,
    *,
    selector: FastSubnetworkSelector,
    zeta: np.ndarray,
    finite_config: FiniteMarkovConfig | None = None,
) -> FastNetworkReport:
    """Run Algorithm-4 selection plus finite Markov checks for reporting."""

    cache = selector._candidate_cache_for(network)
    subnetworks = _select_scored_subnetworks(network, zeta, selector, cache.features)
    analyzer = FiniteMarkovSubnetworkAnalyzer(finite_config)
    results = [analyzer.analyze(network, state, item) for item in subnetworks]
    found_reactions = _unique_reaction_count(item.channels for item in subnetworks)
    finite_results = [item for item in results if item.finite]
    averageable_results = [item for item in results if item.averageable]
    return FastNetworkReport(
        event_count=int(state.event_count),
        step_count=int(state.step_count),
        simulation_time=float(state.t),
        total_reaction_count=int(network.n_channels),
        candidate_subnetwork_count=int(len(cache.candidates)),
        found_subnetwork_count=int(len(subnetworks)),
        found_subnetwork_reaction_count=int(found_reactions),
        finite_markov_subnetwork_count=int(len(finite_results)),
        finite_markov_reaction_count=int(_unique_reaction_count(item.channels for item in finite_results)),
        averageable_subnetwork_count=int(len(averageable_results)),
        averageable_reaction_count=int(_unique_reaction_count(item.channels for item in averageable_results)),
        subnetwork_results=results,
    )


def _finite_markov_candidate_channel_sets(network: PDMPNetwork, *, max_size: int) -> list[tuple[int, ...]]:
    """Return bounded candidates suitable for finite-state CTMC checks.

    The full Algorithm-4 combination search is combinatorial in large connected
    components.  For finite Markov averaging diagnostics we start from
    reversible reaction pairs, which are the dominant explicit-complex pattern:
    C + S <-> C:S.  Small isolated components are also included when their size
    is within ``max_size``.
    """

    return list(_get_fast_network_candidate_cache(network, max_size=max_size).candidates)


def _get_fast_network_candidate_cache(network: PDMPNetwork, *, max_size: int) -> _FastNetworkCandidateCache:
    key = (id(network), int(network.n_channels), int(network.n_species), int(max_size))
    cached = _FAST_NETWORK_CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached
    cached = _build_fast_network_candidate_cache(network, max_size=max_size)
    _FAST_NETWORK_CANDIDATE_CACHE[key] = cached
    return cached


def _build_fast_network_candidate_cache(network: PDMPNetwork, *, max_size: int) -> _FastNetworkCandidateCache:
    """Precompute structural candidate data for fast-subnetwork searches.

    This cache is intentionally state independent.  It stores candidate channel
    sets and their changed/dependency/touch/surrounding species data once per
    network structure.  Later reports only read the current ``zeta`` vector.
    """

    max_size_i = int(max_size)
    changed_by_channel, dependency_by_channel, touch_by_channel = _channel_structural_species(network)
    species_to_reactions = _species_to_touching_reactions(touch_by_channel, network.n_species)
    candidates: set[tuple[int, ...]] = set()
    reaction_by_key: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    nu_minus = np.asarray(network.nu_minus, dtype=float)
    nu_plus = np.asarray(network.nu_plus, dtype=float)
    for channel_id in range(network.n_channels):
        reactants = _stoich_tuple_key_from_row(nu_minus[int(channel_id)])
        products = _stoich_tuple_key_from_row(nu_plus[int(channel_id)])
        reaction_by_key.setdefault((reactants, products), []).append(int(channel_id))

    for (reactants, products), channels in reaction_by_key.items():
        reverse_channels = reaction_by_key.get((products, reactants), [])
        for channel_id in channels:
            for reverse_channel_id in reverse_channels:
                if int(channel_id) < int(reverse_channel_id):
                    candidates.add(tuple(sorted((int(channel_id), int(reverse_channel_id)))))

    graph = _reaction_species_graph_from_touch(touch_by_channel, network.n_species)
    components = _connected_components(graph, network.n_channels)
    for component in components:
        if len(component) <= max_size_i:
            candidates.add(tuple(sorted(int(channel_id) for channel_id in component)))
    ordered = sorted(candidates, key=lambda item: (len(item), item))
    features = [
        _fast_network_candidate_feature_from_precomputed(
            channels,
            changed_by_channel,
            dependency_by_channel,
            species_to_reactions,
        )
        for channels in ordered
    ]
    return _FastNetworkCandidateCache(
        network_id=id(network),
        n_channels=int(network.n_channels),
        n_species=int(network.n_species),
        max_size=max_size_i,
        candidates=ordered,
        features=features,
        reaction_graph=graph,
        components=components,
    )


def _select_scored_subnetworks(
    network: PDMPNetwork,
    zeta: np.ndarray,
    selector: FastSubnetworkSelector,
    candidates: list[tuple[int, ...]] | list[_FastNetworkCandidateFeature],
) -> list[FastSubnetwork]:
    if candidates and isinstance(candidates[0], _FastNetworkCandidateFeature):
        features = candidates
    else:
        features = [
            _fast_network_candidate_feature_from_channels(network, tuple(int(v) for v in candidate))
            for candidate in candidates
        ]
    scored = [selector._score_candidate_feature(feature, zeta) for feature in features]
    scored = [item for item in scored if item.delta_zeta >= selector.threshold]
    scored.sort(key=lambda item: (item.delta_zeta, item.channels.size), reverse=True)
    selected: list[FastSubnetwork] = []
    used_species: set[int] = set()
    for item in scored:
        species = set(int(sid) for sid in item.changed_species)
        if species & used_species:
            continue
        selected.append(item)
        used_species.update(species)
    return selected


def _unique_reaction_count(channel_sets) -> int:
    values: set[int] = set()
    for channels in channel_sets:
        values.update(int(channel_id) for channel_id in np.asarray(channels, dtype=np.int64))
    return len(values)


def _directed_graph_from_transitions(
    n_states: int,
    transitions: list[tuple[int, int, float, int]],
) -> list[set[int]]:
    graph = [set() for _ in range(int(n_states))]
    for source, target, rate, _channel_id in transitions:
        if float(rate) <= 0.0:
            continue
        graph[int(source)].add(int(target))
    return graph


def _strongly_connected_components(graph: list[set[int]]) -> list[list[int]]:
    n_nodes = len(graph)
    reverse = [set() for _ in range(n_nodes)]
    for source, targets in enumerate(graph):
        for target in targets:
            reverse[int(target)].add(int(source))

    seen: set[int] = set()
    order: list[int] = []

    def visit(node: int) -> None:
        seen.add(int(node))
        for nxt in graph[int(node)]:
            if int(nxt) not in seen:
                visit(int(nxt))
        order.append(int(node))

    for node in range(n_nodes):
        if node not in seen:
            visit(node)

    seen.clear()
    components: list[list[int]] = []

    def assign(node: int, component: list[int]) -> None:
        seen.add(int(node))
        component.append(int(node))
        for nxt in reverse[int(node)]:
            if int(nxt) not in seen:
                assign(int(nxt), component)

    for node in reversed(order):
        if int(node) in seen:
            continue
        component: list[int] = []
        assign(int(node), component)
        components.append(component)
    return components


def _component_index(components: list[list[int]], n_states: int) -> np.ndarray:
    comp_id = np.full(int(n_states), -1, dtype=np.int64)
    for index, component in enumerate(components):
        for state_id in component:
            comp_id[int(state_id)] = int(index)
    return comp_id


def _closed_component_ids(graph: list[set[int]], comp_id: np.ndarray) -> set[int]:
    closed = set(int(cid) for cid in np.unique(comp_id) if int(cid) >= 0)
    for source, targets in enumerate(graph):
        source_comp = int(comp_id[int(source)])
        for target in targets:
            target_comp = int(comp_id[int(target)])
            if source_comp != target_comp and source_comp in closed:
                closed.remove(source_comp)
    return closed


def _generator_from_transitions(
    n_states: int,
    transitions: list[tuple[int, int, float, int]],
) -> np.ndarray:
    generator = np.zeros((int(n_states), int(n_states)), dtype=float)
    for source, target, rate, _channel_id in transitions:
        value = max(float(rate), 0.0)
        if value <= 0.0 or int(source) == int(target):
            continue
        generator[int(source), int(target)] += value
    row_sums = np.sum(generator, axis=1)
    for row, total in enumerate(row_sums):
        generator[int(row), int(row)] = -float(total)
    return generator


def _stationary_distribution(generator: np.ndarray) -> np.ndarray:
    n_states = int(generator.shape[0])
    if n_states == 0:
        return np.empty(0, dtype=float)
    if n_states == 1:
        return np.ones(1, dtype=float)
    matrix = np.asarray(generator, dtype=float).T.copy()
    rhs = np.zeros(n_states, dtype=float)
    matrix[-1, :] = 1.0
    rhs[-1] = 1.0
    try:
        values = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        values, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.full(n_states, 1.0 / float(n_states), dtype=float)
    return values / total


def _species_tuple_key(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in values))


def _bounds_from_alpha(alpha: np.ndarray, N0: float, factor: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.power(float(N0), np.maximum(np.asarray(alpha, dtype=float), 0.0))
    lower = np.maximum(0.0, center / float(factor))
    upper = center * float(factor)
    discrete = np.asarray(alpha) <= 0.0
    lower[discrete] = -np.inf
    upper[discrete] = np.inf
    return lower, upper


def _bounds_from_algorithm3(
    alpha: np.ndarray,
    continuous_species_mask: np.ndarray,
    N0: float,
    mu: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha_values = np.asarray(alpha, dtype=float)
    continuous = np.asarray(continuous_species_mask, dtype=bool)
    lower = np.zeros(alpha_values.shape, dtype=float)
    upper = np.full(alpha_values.shape, float(N0) ** float(mu), dtype=float)
    if np.any(continuous):
        lower[continuous] = np.power(float(N0), alpha_values[continuous] - float(eta))
        upper[continuous] = np.power(float(N0), alpha_values[continuous] + float(eta))
    return lower, upper


def _rq_channels(
    network: PDMPNetwork,
    discrete_channels: np.ndarray,
    continuous_species: np.ndarray,
) -> np.ndarray:
    """Return Algorithm-2 RQ channels.

    RQ is the subset of currently discrete reactions whose stoichiometric jump
    changes at least one currently continuous species.  Firing such a reaction
    invalidates the current ODE segment, so the stepper accepts the jump at the
    event time and immediately reruns adaptation.
    """

    channels = np.asarray(discrete_channels, dtype=np.int64)
    species = np.asarray(continuous_species, dtype=np.int64)
    if channels.size == 0 or species.size == 0:
        return np.empty(0, dtype=np.int64)
    changed = np.asarray(network.nu[channels], dtype=float)[:, species] != 0.0
    return channels[np.any(changed, axis=1)].astype(np.int64, copy=False)


def _channel_structural_species(
    network: PDMPNetwork,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    nu = np.asarray(network.nu, dtype=float)
    nu_minus = np.asarray(network.nu_minus, dtype=float)
    changed_by_channel: list[np.ndarray] = []
    dependency_by_channel: list[np.ndarray] = []
    touch_by_channel: list[np.ndarray] = []
    for channel_id in range(network.n_channels):
        changed = np.flatnonzero(nu[int(channel_id)] != 0.0).astype(np.int64, copy=False)
        deps = np.flatnonzero(nu_minus[int(channel_id)] > 0.0).astype(np.int64, copy=False)
        touch = _unique_concat_arrays([changed, deps])
        changed_by_channel.append(changed)
        dependency_by_channel.append(deps)
        touch_by_channel.append(touch)
    return changed_by_channel, dependency_by_channel, touch_by_channel


def _species_to_touching_reactions(
    touch_by_channel: list[np.ndarray],
    n_species: int,
) -> list[np.ndarray]:
    reverse: list[set[int]] = [set() for _ in range(int(n_species))]
    for channel_id, species in enumerate(touch_by_channel):
        for sid in species:
            reverse[int(sid)].add(int(channel_id))
    return [np.asarray(sorted(values), dtype=np.int64) for values in reverse]


def _fast_network_candidate_feature_from_channels(
    network: PDMPNetwork,
    channels: tuple[int, ...],
) -> _FastNetworkCandidateFeature:
    changed_by_channel, dependency_by_channel, touch_by_channel = _channel_structural_species(network)
    species_to_reactions = _species_to_touching_reactions(touch_by_channel, network.n_species)
    return _fast_network_candidate_feature_from_precomputed(
        tuple(sorted(int(channel_id) for channel_id in channels)),
        changed_by_channel,
        dependency_by_channel,
        species_to_reactions,
    )


def _fast_network_candidate_feature_from_precomputed(
    channels: tuple[int, ...],
    changed_by_channel: list[np.ndarray],
    dependency_by_channel: list[np.ndarray],
    species_to_reactions: list[np.ndarray],
) -> _FastNetworkCandidateFeature:
    channels_array = np.asarray(channels, dtype=np.int64)
    changed = _unique_concat_arrays([changed_by_channel[int(channel_id)] for channel_id in channels_array])
    deps = _unique_concat_arrays([dependency_by_channel[int(channel_id)] for channel_id in channels_array])
    catalyst_like = np.setdiff1d(deps, changed, assume_unique=True).astype(np.int64, copy=False)
    touch_species = _unique_concat_arrays([changed, catalyst_like])
    surrounding = _unique_concat_arrays([species_to_reactions[int(sid)] for sid in touch_species])
    if surrounding.size:
        surrounding = np.setdiff1d(surrounding, channels_array, assume_unique=True).astype(np.int64, copy=False)
    return _FastNetworkCandidateFeature(
        channels=tuple(int(channel_id) for channel_id in channels_array),
        channels_array=channels_array,
        changed_species=changed,
        dependency_species=deps,
        catalyst_species=catalyst_like,
        touch_species=touch_species,
        surrounding_channels=surrounding,
    )


def _unique_concat_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(array, dtype=np.int64) for array in arrays if np.asarray(array).size]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    if len(arrays) == 1:
        return np.unique(arrays[0]).astype(np.int64, copy=False)
    return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)


def _stoich_tuple_key_from_row(row: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(row, dtype=float)
    species = np.flatnonzero(values > 0.0)
    if species.size == 0:
        return ()
    counts = np.rint(values[species]).astype(np.int64, copy=False)
    return tuple(int(sid) for sid in np.repeat(species, counts))


def _changed_species_for_channels(network: PDMPNetwork, channels: np.ndarray) -> np.ndarray:
    selected = np.asarray(channels, dtype=np.int64)
    if selected.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.any(network.nu[selected] != 0.0, axis=0)).astype(np.int64, copy=False)


def _dependency_species_for_channels(network: PDMPNetwork, channels: np.ndarray) -> np.ndarray:
    selected = np.asarray(channels, dtype=np.int64)
    if selected.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.any(network.nu_minus[selected] > 0.0, axis=0)).astype(np.int64, copy=False)


def _reaction_species_graph(network: PDMPNetwork) -> list[set[int]]:
    _changed, _deps, touch_by_channel = _channel_structural_species(network)
    return _reaction_species_graph_from_touch(touch_by_channel, network.n_species)


def _reaction_species_graph_from_touch(touch_by_channel: list[np.ndarray], n_species: int) -> list[set[int]]:
    graph = [set() for _ in range(len(touch_by_channel))]
    species_to_reactions: list[list[int]] = [[] for _ in range(int(n_species))]
    for channel_id, species in enumerate(touch_by_channel):
        for sid in species:
            species_to_reactions[int(sid)].append(int(channel_id))
    for reactions in species_to_reactions:
        for a in reactions:
            graph[a].update(int(b) for b in reactions if int(b) != int(a))
    return graph


def _connected_components(graph: list[set[int]], n_nodes: int) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for node in range(n_nodes):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        component = []
        while stack:
            current = stack.pop()
            component.append(int(current))
            for nxt in graph[current]:
                if int(nxt) in seen:
                    continue
                seen.add(int(nxt))
                stack.append(int(nxt))
        components.append(component)
    return components


def _validate_ids(values: np.ndarray, size: int, name: str) -> None:
    if np.any(values < 0) or np.any(values >= int(size)):
        raise IndexError(f"{name} contains out-of-range ids")
