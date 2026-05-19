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
