import json
import shutil
from pathlib import Path

from polymer_sim import (
    ChannelBlock,
    ExperimentRunner,
    FixedPartitionStrategy,
    HybridStepper,
    ReactionNetworkData,
    SSAStepper,
    TrajectoryRecorder,
    build_reaction_rule_tables,
    generate_fixed_species_space,
)


def make_network():
    space = generate_fixed_species_space(["A", "B"], max_len=3, initial_counts={"A": 20, "B": 20})
    tables = build_reaction_rule_tables(space)
    return ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.002,
        k_poly_right=0.002,
        k_frag_left=0.02,
        k_frag_right=0.02,
    )


def test_ssa_runs():
    network = make_network()
    recorder = TrajectoryRecorder()
    result = ExperimentRunner().run_one(network, SSAStepper(), t_end=0.5, seed=1, recorder=recorder)
    assert result.state.t == 0.5
    assert result.summary.n_steps >= 1
    assert recorder.finalize().states.shape[1] == network.n_species


def test_hybrid_skeleton_runs():
    network = make_network()
    a = network.species_idx("A")
    b = network.species_idx("B")
    fast = network.channel_id(ChannelBlock.LEFT_ADD, int(network.left_add_local_id[a, b]))
    result = ExperimentRunner().run_one(
        network,
        HybridStepper(),
        t_end=0.5,
        seed=2,
        dt=0.05,
        partition_strategy=FixedPartitionStrategy([fast]),
    )
    assert result.state.t >= 0.5
    assert result.summary.n_steps >= 1


def test_runner_can_write_timing_report():
    network = make_network()
    output_dir = Path("tests_runtime_timing_report")
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        result = ExperimentRunner().run_one(
            network,
            SSAStepper(),
            t_end=0.1,
            seed=3,
            timing_report=True,
            timing_report_dir=output_dir,
            timing_report_interval_events=1,
            timing_report_name="timing_test",
        )
        paths = result.summary.metadata["timing_report_paths"]
        json_path = output_dir / "timing_test.json"
        plot_path = output_dir / "timing_test_events.png"
        simulation_clock_plot_path = output_dir / "timing_test_simulation_clock.png"
        assert paths["json"] == str(json_path)
        assert paths["event_plot"] == str(plot_path)
        assert paths["simulation_clock_plot"] == str(simulation_clock_plot_path)
        assert json_path.exists()
        assert plot_path.exists()
        assert simulation_clock_plot_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["seed"] == 3
        assert payload["stepper"] == "SSAStepper"
        assert payload["runner_setup_wall_seconds"] >= 0.0
        assert payload["simulation_loop_wall_seconds"] >= 0.0
        assert payload["step_wall_seconds"] >= 0.0
        assert payload["simulation_clock_interval"] == 0.01
        assert "simulation_clock_samples" in payload
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
