from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from polymer_sim.core.elementary import ElementaryMassActionNetwork
from polymer_sim.core.network import ReactionNetworkData
from polymer_sim.core.state import SystemState
from polymer_sim.partition.pdmp import (
    FastNetworkReport,
    FastSubnetworkSelector,
    FiniteMarkovConfig,
    PDMPPartitionStrategy,
    analyze_fast_network,
)
from polymer_sim.recording.base import BaseRecorder, PathLike


class FastNetworkReportRecorder(BaseRecorder):
    """Recorder wrapper that writes finite-Markov fast-subnetwork reports.

    The wrapper delegates normal trajectory/summary recording to another
    recorder and adds a low-frequency analysis side effect.  It is intended for
    diagnostics: it does not feed results back into the stepper and therefore
    does not change the simulated stochastic process.
    """

    def __init__(
        self,
        base_recorder: BaseRecorder,
        *,
        network: ElementaryMassActionNetwork | ReactionNetworkData,
        partition_strategy: PDMPPartitionStrategy,
        output_path: PathLike,
        interval_events: int = 1000,
        selector: FastSubnetworkSelector | None = None,
        finite_config: FiniteMarkovConfig | None = None,
    ):
        self.base_recorder = base_recorder
        self.network = network
        self.partition_strategy = partition_strategy
        self.output_path = Path(output_path)
        self.interval_events = int(interval_events)
        self.selector = selector or _selector_from_strategy(partition_strategy)
        self.finite_config = finite_config or FiniteMarkovConfig()
        self.reports: list[FastNetworkReport] = []
        self._next_event = self.interval_events
        if self.interval_events <= 0:
            raise ValueError("interval_events must be > 0")

    def initialize(
        self,
        species_names: list[str],
        initial_state: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.base_recorder.initialize(species_names, initial_state, metadata)
        self.reports = []
        self._next_event = self.interval_events
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(_report_header(), encoding="utf-8")

    def record_step(
        self,
        *,
        time: float,
        state: np.ndarray,
        step_count: int,
        event_count: int,
        event_time: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.base_recorder.record_step(
            time=time,
            state=state,
            step_count=step_count,
            event_count=event_count,
            event_time=event_time,
            metadata=metadata,
        )
        if int(event_count) <= 0:
            return
        while self._next_event <= int(event_count):
            report_state = SystemState(t=float(time), x=np.asarray(state, dtype=float).copy())
            report_state.step_count = int(step_count)
            report_state.event_count = int(self._next_event)
            partition = self.partition_strategy.partition(self.network, report_state)
            report = analyze_fast_network(
                self.network,
                report_state,
                selector=self.selector,
                zeta=partition.zeta,
                finite_config=self.finite_config,
            )
            self.reports.append(report)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(_format_report(report))
            self._next_event += self.interval_events

    def finalize(self):
        record = self.base_recorder.finalize()
        run_metadata = getattr(record, "run_metadata", None)
        if isinstance(run_metadata, dict):
            run_metadata["fast_network_report_path"] = str(self.output_path)
            run_metadata["fast_network_report_interval_events"] = int(self.interval_events)
            run_metadata["fast_network_report_count"] = int(len(self.reports))
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, dict):
            metadata["fast_network_report_path"] = str(self.output_path)
            metadata["fast_network_report_interval_events"] = int(self.interval_events)
            metadata["fast_network_report_count"] = int(len(self.reports))
        return record


def _selector_from_strategy(strategy: PDMPPartitionStrategy) -> FastSubnetworkSelector:
    config = getattr(strategy, "config", None)
    threshold = float(getattr(config, "fast_subnetwork_threshold", 1.0))
    max_size = int(getattr(config, "fast_subnetwork_max_size", 3))
    return FastSubnetworkSelector(threshold=threshold, max_size=max_size)


def _report_header() -> str:
    return (
        "# fast network report\n"
        "# total_reaction_count: all elementary reaction channels in the current network\n"
        "# candidate_subnetwork_count: reversible-pair/small-component finite-Markov candidates considered\n"
        "# found_subnetwork_count/reactions: candidates passing Algorithm-4-style timescale scoring\n"
        "# finite_markov_*: selected subnetworks whose reachable internal CTMC enumeration completed\n"
        "# averageable_*: finite selected subnetworks with a unique stationary distribution under current checks\n"
        "event_count\tstep_count\tsimulation_time\ttotal_reaction_count\t"
        "candidate_subnetwork_count\tfound_subnetwork_count\tfound_subnetwork_reaction_count\t"
        "finite_markov_subnetwork_count\tfinite_markov_reaction_count\t"
        "averageable_subnetwork_count\taverageable_reaction_count\n"
    )


def _format_report(report: FastNetworkReport) -> str:
    line = (
        f"{int(report.event_count)}\t"
        f"{int(report.step_count)}\t"
        f"{float(report.simulation_time):.12g}\t"
        f"{int(report.total_reaction_count)}\t"
        f"{int(report.candidate_subnetwork_count)}\t"
        f"{int(report.found_subnetwork_count)}\t"
        f"{int(report.found_subnetwork_reaction_count)}\t"
        f"{int(report.finite_markov_subnetwork_count)}\t"
        f"{int(report.finite_markov_reaction_count)}\t"
        f"{int(report.averageable_subnetwork_count)}\t"
        f"{int(report.averageable_reaction_count)}\n"
    )
    detail_lines = []
    for index, item in enumerate(report.subnetwork_results):
        if not item.averageable:
            continue
        channels = ",".join(str(int(channel_id)) for channel_id in item.channels.tolist())
        species = ",".join(str(int(species_id)) for species_id in item.changed_species.tolist())
        detail_lines.append(
            "# averageable_subnetwork "
            f"event={int(report.event_count)} index={index} "
            f"channels={channels} changed_species={species} "
            f"states={int(item.state_count)} transitions={int(item.transition_count)}\n"
        )
    return line + "".join(detail_lines)
