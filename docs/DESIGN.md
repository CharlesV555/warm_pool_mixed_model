# 架构说明

更新时间：2026-08-08。

本文总结 `polymer_sim` 与 `examples` 中批量测试文件的内容和职能。

## 总体流程

普通 polymer-rule 模拟流程：

```text
SpeciesSpace
  -> ReactionRuleTables
      -> ReactionNetworkData
          -> SystemState
              -> Stepper.step(...)
                  -> ExperimentRunner.run_one(...)
                      -> Recorder / Timing / Trajectory
```

PDMP/strict paper-style 流程多一层 elementary 展开：

```text
ReactionNetworkData
  -> ElementaryMassActionNetwork
      -> PDMP partition strategy
          -> PDMPStepper / strict_2018 comparison
```

`ReactionNetworkData` 是 polymer-rule 热路径。`ElementaryMassActionNetwork`
是 elementary zero/first/second-order mass-action SRN 视图，主要给 PDMP 和严格论文算法用。

## species 与 rule tables

`polymer_sim/model/species.py` 定义 `SpeciesSpace`。

核心字段：

- `species_names`
- `name_to_idx`
- `x0`
- `lengths`
- `n_monomers`
- `alphabet`
- `max_len`

固定约定：`sid < n_monomers` 表示 monomer。

`polymer_sim/model/rules.py` 生成 terminal polymer 规则表：

- `left_join[m, sid]`
- `right_join[sid, m]`
- `split_left_monomer[sid]`
- `split_left_rest[sid]`
- `split_right_rest[sid]`
- `split_right_monomer[sid]`
- `can_split[sid]`

当前只表示 terminal monomer addition 和 terminal monomer cleavage，不生成内部 cleavage channel。

## ReactionNetworkData

文件：`polymer_sim/core/network.py`。

职能：

- 从 species/rule table 构造 block-contiguous channel；
- 保存 block-local reactant/product/rate/catalysis arrays；
- 提供 channel semantic API；
- O(1) 更新 state；
- scalar/full/subset propensity 计算；
- dependency index 与 local propensity update；
- linear 和 substrate-saturating catalysis；
- sparse catalyst cache。

当前 channel block：

- `LEFT_ADD`
- `RIGHT_ADD`
- `LEFT_SPLIT`
- `RIGHT_SPLIT`
- `OUTFLOW`
- `INFLOW`

关键热路径方法：

- `compute_all_propensities(state)`
- `compute_propensities_for_channels(channel_ids, state)`
- `affected_channels_for_species(species_ids)`
- `update_propensities_for_species(propensities, state, species_ids)`
- `_compute_block_base_propensities(...)`
- `_apply_block_catalysis(...)`
- `_sparse_linear_catalytic_factors(...)`
- `_sparse_substrate_saturating_factors(...)`

当前 profile 表明 `_apply_block_catalysis` 和 `_sparse_substrate_saturating_factors`
是主要瓶颈。

## ElementaryMassActionNetwork

文件：`polymer_sim/core/elementary.py`。

主要对象：

- `ElementaryExpansionConfig`
- `ElementaryMassActionNetwork`
- `build_elementary_mass_action_network(...)`

职能：

- 把 `ReactionNetworkData` 展开成 elementary mass-action reactions；
- 保存 `nu_minus`、`nu_plus`、`nu`、`nu_csr`；
- 保存 rate constants 和 reaction labels；
- 建立 source polymer channel 到 elementary channel 的映射；
- 预计算 reaction order、reactant1/reactant2、homo second-order；
- 提供和 `ReactionNetworkData` 类似的 dependency API。

该层适合 PDMP scaling/partition，不应替代普通 polymer-rule SSA/CLE 热路径。

## simulation/stepper.py

这是当前算法开发中心。

共享对象：

- `StepperContext`：network、rng、partition strategy、blending strategy、wall deadline。
- `StepResult`：advanced time、event flag、channel id、propensity sum、tau、details。
- `OptimizedNRMConfig`
- `PDMPConfig`
- `BlendedHybridConfig`

主要 stepper：

- `SSAStepper`：direct SSA，本地 propensity cache + cumulative sampling。
- `OptimizedNRMStepper`：heap NRM，scheduled time/version/stale pop。
- `PDMPStepper`：adaptive PDMP，支持 scan/heap/Gillespie discrete event locator。
- `CLEStepper`：Euler-Maruyama CLE，自适应负值 retry。
- `HybridStepper`：fixed fast/slow split。
- `BlendedHybridStepper`：beta-based jump/CLE splitting。
- `NRMBlendedHybridStepper`：blended hybrid 的 NRM discrete event 版本。

`BlendedHybridStepper` 内部重点：

- beta lookup cache：`_ChannelBetaLookup`；
- beta value cache：`_channel_beta_cache`、`_species_beta_cache`；
- rounded observed state cache：`_observed_propensity_cache`；
- beta reuse hint：把 changed species、beta affected channel、catalyst affected channel 传给 observed propensity 局部更新；
- adaptive CLE dt；
- optional CLE sparsity sampler。

## partition/pdmp.py

PDMP partition 专用模块。

核心对象：

- `PDMPPartitionResult`
- `PDMPPartitionStrategy`
- `FixedPDMPPartitionStrategy`
- `ScalingPDMPConfig`
- `ScalingPDMPPartitionStrategy`
- `LinearCatalysisScalingPDMPPartitionStrategy`
- `FiniteMarkovScalingPDMPPartitionStrategy`
- `FastSubnetworkSelector`
- `FiniteMarkovSubnetworkAnalyzer`

职能：

- 根据 copy-number/rate scale 给 species/channel 分类；
- 生成 continuous/discrete channels 和 species；
- 生成 bounds 和 reaction quotient channels；
- 识别 fast subnetworks；
- 给 PDMP stepper 提供 adaptation 依据。

## experiment/runner.py

`ExperimentRunner.run_one(...)` 负责整体调度。

职能：

- 初始化 RNG 和 `SystemState`；
- 初始化 recorder，写入 seed、stepper metadata、channel labels；
- 循环调用 `stepper.step(...)`；
- 检查 `t_end`、`max_steps`、`max_runtime_seconds`、`no_progress`；
- 把 wall deadline 放进 `StepperContext`；
- 每步后应用 restriction；
- restriction 后 invalidates stepper cache；
- 记录 event metadata、continuous channel increments；
- 生成 timing report；
- finalize 后合并 stepper summary metadata，例如 CLE sparsity sampling。

stop reason：

- `reached_t_end`
- `max_steps`
- `max_runtime_seconds`
- `no_progress`

## recording

`polymer_sim/recording/` 包含：

- `trajectory.py`：`TrajectoryRecorder`、`.npz` 保存/读取。
- `summary.py`：`SummaryRecorder`、batch summary、dt comparison。
- `timing.py`：`RunTimingReport`、event/simulation-clock/dt CLE metric 图。
- `plot_single_run.py`：species、reaction frequency、interval、state tree 等图。
- `plot_generation.py`：批量 trajectory 绘图。
- `distribution_comparison.py`：批量分布比较。
- `fast_network_report.py`：fast subnetwork 诊断。
- `cle_sparsity_sampler.py`：当前未跟踪诊断模块，采样 CLE amounts 和 stoichiometry 稀疏性。

## examples 根目录

单次运行和旧示例：

- `minimal_run.py`：最小运行。
- `hs2014_formal_example.py`：WH/HS2014 formal RAF。
- `without_catalyst.py`：无催化 baseline。
- `catalyst_run.py`：随机催化 polymer。
- `time_comparison.py`：较早 timing。
- `plot.ipynb`、`plot2.ipynb`：分析 notebook。

中期批量/方法示例：

- `blended_hybrid_minimal.py`：blended smoke。
- `multiple_run_core.py`：多次运行核心实现。
- `multiple_run.py`：多次运行入口和 paired SSA/blended 入口。
- `nrm_vs_ssa_multiple_run.py`：NRM vs SSA。
- `compute_strategy.py`：CPU 并行策略。
- `fast_dimerization.py`：elementary fast dimerization benchmark。
- `oscillator.py`：手工构造 oscillator-like 催化网络。

场景目录：

- `MM_catalysis/`：substrate-saturating cross-catalysis。
- `linear_catalysis/`：linear cross-catalysis SSA/PDMP 版本。

## examples/compare 批量测试框架

`examples/compare/common.py` 是核心。

定义：

- `NetworkSpec`
- `RunSettings`
- `NETWORK_SPECS`
- `METHOD_ORDER`
- `build_network(...)`
- `prepare_network_for_method(...)`
- `make_stepper(...)`
- `run_method(...)`
- `run_method_profiled(...)`
- `run_comparison(...)`
- `write_tables(...)`
- `write_simulation_summary_tables(...)`
- `write_profile_report(...)`

method registry：

- `gillespie_ssa`
- `optimized_nrm`
- `gillespie_cle_hybrid`
- `nrm_cle_hybrid`
- `gillespie_pdmp_lp`
- `nrm_pdmp_lp`
- `strict_2018_pdmp`

重要 network registry：

- `fast_dimerization`
- `toggle_switch`
- `repressilator`
- `polymer_len5_00000_catalyzes_0`
- `polymer_food_dimer_inhibition_len3`
- `polymer_len10_two_stage_1_catalysis`
- `polymer_a_len5_a5_catalyzes_a_constant_food`
- `polymer_a_len6_a6_catalyzes_a_constant_food`
- `polymer_a_len8_a8_catalyzes_a_constant_food`
- `linear_cross_len3`
- `linear_cross_len4`
- `linear_cross_len5`

薄 wrapper：

- `gillespie_ssa.py`
- `optimized_nrm.py`
- `gillespie_cle_hybrid.py`
- `nrm_cle_hybrid.py`
- `gillespie_pdmp_lp.py`
- `nrm_pdmp_lp.py`
- `strict_2018_pdmp.py`

批量文件：

- `batch_compare.py`：固定 wall-clock cProfile 对比。
- `batch_compare_wall_times.py`：多 wall-clock budget sweep。
- `experiment_matrix.py`：矩阵式配置/结果框架。
- `batch_compare_matrix.py`：`experiment_matrix.py` CLI 入口。
- `a_polymer_update_matrix.py`：A-only len5/6/8 具体矩阵。
- `beta_hybrid_update_matrix.py`：四种 beta-hybrid local/global 更新矩阵，当前性能优化最相关。

`experiment_matrix.py` 输出：

- `test_config.csv`
- `test_config.xlsx`
- `test_result.csv`
- `test_result.xlsx`
- `test_result_long.csv`
- `test_result_long.xlsx`
- `profiles/*.prof`
- `profiles/*_top40.txt`
- trajectories。

## food supply mode

compare suite 支持：

- `explicit_inflow`：food 是 formal inflow channel。
- `constant`：runner restriction 每步恢复 food count。
- `upper_limit`：只 cap，不 replenish。
- `none`：无 food helper。

这会影响 channel 数、restriction、stepper cache invalidation 和性能。

## 优化边界

优先考虑：

- 降低 `compute_propensities_for_channels` 的 selected-channel 开销；
- 降低 substrate-saturating sparse factor 的重复计算；
- 改善 observed propensity local update 的 affected-channel 选择；
- 减少 CLE sparse slicing/matmul 开销；
- 保持 benchmark 输出和 profile 可复现。

谨慎修改：

- channel id 布局；
- catalysis 公式；
- food restriction 时机；
- low-count rounding 和 negative CLE handling；
- stepper cache invalidation。
