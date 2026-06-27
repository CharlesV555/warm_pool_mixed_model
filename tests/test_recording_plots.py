import matplotlib

matplotlib.use("Agg")

import numpy as np

from polymer_sim import (
    BlendedHybridConfig,
    BlendedHybridStepper,
    ChannelBlock,
    ExperimentRunner,
    ReactionNetworkData,
    SSAStepper,
    TrajectoryRecorder,
    animate_reaction_network_state_tree,
    assign_paper_minimal_catalysis,
    build_reaction_rule_tables,
    build_restriction,
    build_n3_wh_network,
    generate_fixed_species_space,
    plot_channel_propensity_time_series,
    plot_reaction_interval_bar,
    plot_reaction_interval_wave,
    plot_reaction_frequency_over_time,
    plot_reaction_network_state_tree,
    plot_reaction_trigger_frequency,
    plot_species_with_outflow,
    plot_time_series,
)


def test_trajectory_metadata_contains_trigger_counts_and_outflow():
    network = build_n3_wh_network(initial_counts={"0": 10.0, "1": 10.0}, k_right_add=0.1, k_nonfood_outflow=0.8)
    assign_paper_minimal_catalysis(network, strength=1.0)
    recorder = TrajectoryRecorder()
    result = ExperimentRunner().run_one(
        network,
        SSAStepper(),
        t_end=0.5,
        seed=10,
        recorder=recorder,
        restriction=build_restriction(network, food_count=10.0),
    )
    record = recorder.finalize()
    assert result.summary.n_events >= 0
    assert "channel_trigger_counts" in record.run_metadata
    assert len(record.run_metadata["channel_trigger_counts"]) == network.n_channels
    assert "channel_event_times" in record.run_metadata
    assert "channel_event_ids" in record.run_metadata
    assert len(record.run_metadata["channel_event_times"]) == len(record.run_metadata["channel_event_ids"])
    assert "reaction_intervals" in record.run_metadata
    assert "reaction_interval_times" in record.run_metadata
    assert len(record.run_metadata["reaction_intervals"]) == len(record.run_metadata["channel_event_ids"])
    assert len(record.run_metadata["reaction_interval_times"]) == len(record.run_metadata["reaction_intervals"])
    assert all(interval >= 0.0 for interval in record.run_metadata["reaction_intervals"])
    assert "tracked_outflow" in record.run_metadata
    assert set(record.run_metadata["tracked_outflow"]["species_names"]) == {
        name for name in network.species_names if name not in {"0", "1"}
    }


def test_plot_functions_run_on_record():
    network = build_n3_wh_network(initial_counts={"0": 10.0, "1": 10.0}, k_right_add=0.1, k_nonfood_outflow=0.8)
    assign_paper_minimal_catalysis(network, strength=1.0)
    recorder = TrajectoryRecorder()
    ExperimentRunner().run_one(
        network,
        SSAStepper(),
        t_end=0.5,
        seed=11,
        recorder=recorder,
        restriction=build_restriction(network, food_count=10.0),
    )
    record = recorder.finalize()
    fig1, ax1 = plot_reaction_trigger_frequency(record, species_names=["000"])
    fig1b, ax1b = plot_reaction_trigger_frequency(record, species_names=["000"], legand="species_names")
    fig1c, ax1c = plot_reaction_trigger_frequency(record, species_names=["000"], legand=None)
    fig2, axes2 = plot_species_with_outflow(
        record,
        species_indices=[
            record.species_names.index("000"),
            record.species_names.index("111"),
        ],
    )
    fig3, ax3 = plot_channel_propensity_time_series(
        record,
        network,
        block_type=ChannelBlock.RIGHT_ADD,
        top_n=4,
    )
    fig4, ax4 = plot_reaction_frequency_over_time(record, n_bins=100, top_n=4)
    fig5, ax5 = plot_reaction_interval_bar(record, time_range=(0.0, 0.5))
    fig6, ax6 = plot_reaction_interval_wave(
        record,
        time_windows=[
            {"label": "0-0.2", "start": 0.0, "end": 0.2},
            {"label": "0.2-end", "start": 0.2, "end": None},
        ],
    )
    fig7, ax7 = plot_reaction_network_state_tree(record, time_points=(0.0, 0.5), label_alpha=0.6)
    fig8, ax8, anim8 = animate_reaction_network_state_tree(
        record,
        dt=0.1,
        time_range=(0.0, 0.5),
        label_alpha=0.6,
    )
    fig9, ax9 = plot_time_series(record, species_indices=[0, 1], time_range=(0.1, 0.3))
    assert fig1 is not None and ax1 is not None
    assert len(ax1.collections) >= 1
    assert fig1b is not None and ax1b is not None
    assert ax1b.get_legend() is not None
    assert any("000" in text.get_text() for text in ax1b.get_legend().get_texts())
    assert fig1c is not None and ax1c is not None
    assert ax1c.get_legend() is None
    assert fig2 is not None and axes2 is not None
    assert fig3 is not None and ax3 is not None
    assert fig4 is not None and ax4 is not None
    assert fig5 is not None and ax5 is not None
    assert fig6 is not None and ax6 is not None
    assert fig7 is not None and ax7 is not None
    assert fig8 is not None and ax8 is not None and anim8 is not None
    assert fig9 is not None and ax9 is not None
    for line in ax9.lines:
        xdata = line.get_xdata()
        assert xdata.min() >= 0.1
        assert xdata.max() <= 0.3


def test_reaction_trigger_frequency_stacks_continuous_counts():
    space = generate_fixed_species_space(
        ["A"],
        max_len=2,
        initial_counts={"AA": 100.0},
    )
    tables = build_reaction_rule_tables(space)
    network = ReactionNetworkData.from_species_space(
        space,
        tables,
        k_poly_left=0.0,
        k_poly_right=0.0,
        k_frag_left=0.0,
        k_frag_right=0.0,
        k_outflow=2.0,
        outflow_species_ids=[space.idx("AA")],
    )
    recorder = TrajectoryRecorder()
    ExperimentRunner().run_one(
        network,
        BlendedHybridStepper(BlendedHybridConfig(i1=-2.0, i2=-1.0, dt_cle=0.01, dt_macro=0.01)),
        t_end=0.03,
        seed=12,
        recorder=recorder,
        max_steps=10,
    )
    record = recorder.finalize()
    outflow_channel = network.channel_id(
        ChannelBlock.OUTFLOW,
        int(network.outflow_local_id_by_source[network.species_idx("AA")]),
    )
    discrete_counts = np.asarray(record.run_metadata["channel_trigger_counts"], dtype=float)
    continuous_counts = np.asarray(record.run_metadata["channel_continuous_trigger_counts"], dtype=float)

    assert discrete_counts[outflow_channel] == 0.0
    assert continuous_counts[outflow_channel] > 0.0
    fig, ax = plot_reaction_trigger_frequency(record, top_n=1)
    assert fig is not None and ax is not None
