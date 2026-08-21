import json
import shutil
from pathlib import Path

import numpy as np

from polymer_sim import (
    ChannelBlock,
    CLEStepper,
    ExperimentRunner,
    FixedPartitionStrategy,
    HybridStepper,
    ReactionNetworkData,
    SSAStepper,
    TrajectoryRecorder,
    build_reaction_rule_tables,
    generate_fixed_species_space,
    has_trajectory_sidecar,
    load_trajectory_record,
    sample_trajectory_states_from_path,
    save_trajectory_record,
    trajectory_dt_statistics,
    trajectory_sidecar_dir,
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


def test_runner_lightweight_step_mode_counts():
    network = make_network()
    ssa_result = ExperimentRunner().run_one(network, SSAStepper(), t_end=0.5, seed=11)
    assert ssa_result.summary.metadata["ssa_steps"] == ssa_result.summary.n_events
    assert ssa_result.summary.metadata["con_steps"] == 0

    cle_result = ExperimentRunner().run_one(network, CLEStepper(), t_end=0.1, seed=12, dt=0.05)
    assert cle_result.summary.metadata["ssa_steps"] == 0
    assert cle_result.summary.metadata["con_steps"] == cle_result.summary.n_steps


def test_runner_stops_on_catalytic_overflow_guard():
    network = make_network()
    a = network.species_idx("A")
    b = network.species_idx("B")
    catalyst = network.species_idx("AA")
    channel = network.channel_id(ChannelBlock.LEFT_ADD, int(network.left_add_local_id[a, b]))
    network.set_catalytic_strength(channel, catalyst_sid=catalyst, strength=1e200, mirror_reverse=False)
    x0 = network.x0.copy()
    x0[a] = 1e10
    x0[b] = 1e10
    x0[catalyst] = 1e100

    result = ExperimentRunner().run_one(network, SSAStepper(), t_end=0.5, seed=17, x0=x0)

    assert result.summary.metadata["stop_reason"] == "numerical_guard_catalysis_overflow_risk"
    guard = result.summary.metadata["numerical_guard"]
    assert guard["guard"] == "catalytic_propensity_multiply"
    assert guard["threshold_fraction_of_float_max"] == 0.5
    assert guard["n_risky_entries"] >= 1
    assert result.summary.final_time == 0.0


def test_trajectory_dt_statistics_reads_intervals_without_states():
    network = make_network()
    recorder = TrajectoryRecorder()
    ExperimentRunner().run_one(network, SSAStepper(), t_end=0.5, seed=13, recorder=recorder)
    record = recorder.finalize()
    path = Path("tests") / "_trajectory_dt_statistics_tmp.npz"
    sidecar = trajectory_sidecar_dir(path)
    plot_path = Path("tests") / "_trajectory_dt_statistics_tmp.png"
    path.unlink(missing_ok=True)
    shutil.rmtree(sidecar, ignore_errors=True)
    plot_path.unlink(missing_ok=True)

    try:
        save_trajectory_record(path, record)
        stats = trajectory_dt_statistics(path, bins=5, plot_path=plot_path, x_log=True)

        assert stats.count == record.times.size - 1
        assert stats.histogram_counts.sum() == stats.count
        assert np.all(stats.histogram_edges > 0.0)
        assert stats.plot_path == str(plot_path)
        assert plot_path.exists()
        assert "accepted_step_intervals" in np.load(path, allow_pickle=False).files
    finally:
        path.unlink(missing_ok=True)
        shutil.rmtree(sidecar, ignore_errors=True)
        plot_path.unlink(missing_ok=True)


def test_trajectory_sidecar_mmap_and_sampling():
    network = make_network()
    recorder = TrajectoryRecorder()
    ExperimentRunner().run_one(network, SSAStepper(), t_end=0.5, seed=14, recorder=recorder)
    record = recorder.finalize()
    path = Path("tests") / "_trajectory_sidecar_tmp.npz"
    sidecar = trajectory_sidecar_dir(path)
    shutil.rmtree(sidecar, ignore_errors=True)
    path.unlink(missing_ok=True)

    try:
        save_trajectory_record(path, record)

        assert path.exists()
        assert has_trajectory_sidecar(path)
        assert (sidecar / "times.npy").exists()
        assert (sidecar / "states.npy").exists()
        assert (sidecar / "species_names.json").exists()
        assert (sidecar / "metadata.json").exists()

        mmap_record = load_trajectory_record(path)
        assert isinstance(mmap_record.times, np.memmap)
        assert isinstance(mmap_record.states, np.memmap)
        assert np.allclose(mmap_record.states[-1], record.states[-1])

        eager_record = load_trajectory_record(path, mmap=False)
        assert not isinstance(eager_record.states, np.memmap)
        assert np.allclose(eager_record.states[-1], record.states[-1])

        sample_points = np.asarray([0.0, float(record.times[-1])], dtype=float)
        sampled, species_names, info = sample_trajectory_states_from_path(path, sample_points, mmap=True)
        assert info["storage"] == "sidecar"
        assert info["mmap"] is True
        assert species_names == record.species_names
        assert sampled.shape == (2, network.n_species)
        assert np.allclose(sampled[0], record.states[0])
        assert np.allclose(sampled[-1], record.states[-1])
    finally:
        path.unlink(missing_ok=True)
        shutil.rmtree(sidecar, ignore_errors=True)


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
        dt_cle_metrics_plot_path = output_dir / "timing_test_dt_cle_metrics.png"
        assert paths["json"] == str(json_path)
        assert paths["event_plot"] == str(plot_path)
        assert paths["simulation_clock_plot"] == str(simulation_clock_plot_path)
        assert paths["dt_cle_metrics_plot"] == str(dt_cle_metrics_plot_path)
        assert json_path.exists()
        assert plot_path.exists()
        assert simulation_clock_plot_path.exists()
        assert dt_cle_metrics_plot_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert result.summary.metadata["stepper_name"] == "SSAStepper"
        assert result.summary.metadata["stepper_info"]["name"] == "SSAStepper"
        assert payload["seed"] == 3
        assert payload["stepper"] == "SSAStepper"
        assert payload["metadata"]["stepper_name"] == "SSAStepper"
        assert payload["metadata"]["stepper_info"]["name"] == "SSAStepper"
        assert payload["runner_setup_wall_seconds"] >= 0.0
        assert payload["simulation_loop_wall_seconds"] >= 0.0
        assert payload["step_wall_seconds"] >= 0.0
        assert "restriction_wall_seconds" not in payload
        assert payload["simulation_clock_interval"] == 0.01
        assert "simulation_clock_samples" in payload
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
