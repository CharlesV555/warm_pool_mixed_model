# 项目长期状态记录

更新时间：2026-08-08。

用途：给跨电脑继续开发的 Codex 读取。新会话优先读本文件，再读
`docs/DESIGN.md` 和 `docs/BENCHMARK.md`。

## 当前主线

当前需要做的是算法优化，核心入口是：

- `polymer_sim/simulation/stepper.py`

项目已经不是 README 里早期的最小 SSA/CLE 骨架。目前主线已经包含：

- Gillespie SSA；
- optimized NRM；
- Gillespie CLE hybrid；
- NRM CLE hybrid；
- PDMP；
- strict/paper-style 2018 PDMP 对比；
- beta-hybrid local/global 更新矩阵；
- cProfile、timing report、experiment matrix 批量测试框架。

目前最该关注 `stepper.py` 中这些对象：

- `BlendedHybridStepper`
- `NRMBlendedHybridStepper`
- `SSAStepper`
- `OptimizedNRMStepper`
- `PDMPStepper`

当前性能瓶颈更接近：

- beta 计算和 beta cache；
- rounded observed state 的 propensity cache；
- local propensity update；
- CLE increment 与自适应 dt；
- `ReactionNetworkData.compute_propensities_for_channels(...)`；
- `ReactionNetworkData._apply_block_catalysis(...)`；
- `ReactionNetworkData._sparse_substrate_saturating_factors(...)`。

不要把旧文档中“PDMP 未实现”“hybrid 只是占位”的描述当作当前事实。

## 工作区状态

写入本文档时，工作区有未提交修改和未跟踪生成文件。不要随手回退。

已修改的 tracked 文件：

- `examples/blended_hybrid_minimal.py`
- `examples/compare/experiment_matrix.ipynb`
- `examples/plot2.ipynb`
- `polymer_sim/__init__.py`
- `polymer_sim/core/elementary.py`
- `polymer_sim/core/network.py`
- `polymer_sim/experiment/runner.py`
- `polymer_sim/recording/__init__.py`
- `polymer_sim/simulation/stepper.py`
- `tests/test_blended_hybrid_stepper.py`

重要未跟踪文件和目录：

- `examples/compare/outputs/`
- `timing_reports/`
- `blended.prof`
- `blended_cprofile_out.txt`
- `cross_catalysis_SSA_linear.prof`
- `cross_catalysis_PDMP_SSA_linear_out.txt`
- `beta_hybrid_update_matrix.py.lprof`
- `cle_sparsity.png`
- `polymer_sim/recording/cle_sparsity_sampler.py`

`beta_hybrid_update_matrix.py.lprof` 是二进制 line-profiler dump，不是普通文本。

## 最近提交脉络

最近 git log 显示的开发方向：

- `13619ce`：继续优化 beta-hybrid 性能。
- `b46f3ee`：优化 beta-hybrid 前备份。
- `f55e9c4`：PDMP 引入 numba。
- `c3921fe`：优化 PDMP 底层和 elementary network propensity。
- `76f8ef5`：加入严格按论文思路的 2018 PDMP 实现。
- `2348ccf`：修复 batch wall-time 传入 stepper，长步也能被墙钟截断。
- `968683c`：加入 6 种算法批量对比。
- `75bed91`：PDMP 引入；examples 分成 SRN 模拟和饱和催化线性模拟两类；未来优化方向是线性模拟底层和 partition。
- `9da7435`：废除 restriction 保持 inflow 恒定的旧做法，改成 hill/capped inflow；加入 NRM 与 hybrid NRM；timing.py 增加离散/连续 propensity 比例和 dt 内离散反应概率诊断。

## polymer_sim 结构记忆

`polymer_sim` 是固定 species 空间、整数 id、NumPy 数组和 block-contiguous channel id 的模拟框架。

- `polymer_sim/core/`
  - `network.py`：主热路径，`ReactionNetworkData`。
  - `elementary.py`：把 polymer-rule network 展开成 elementary mass-action SRN，主要给 PDMP/strict 2018 用。
  - `state.py`：轻量 `SystemState`。
  - `enums.py`：`ChannelBlock`。
  - `kernels/numba_kernels.py`：可选 numba kernel，用于 elementary network 局部 cache 刷新。
- `polymer_sim/model/`
  - `species.py`：固定全空间 species 生成。
  - `rules.py`：terminal join/split 查表。
  - `catalysis.py`：催化剂分配、清除、强度设置。
  - `wills_henderson.py`：HS2014/Wills-Henderson 参考系统。
- `polymer_sim/simulation/`
  - `stepper.py`：所有算法 stepper，当前优化中心。
  - `restriction.py`：food replenish、upper limit、food supply mode。
  - `propensity.py`：旧兼容包装。
- `polymer_sim/partition/`
  - `strategies.py`：普通 fixed partition 和 blending interface。
  - `pdmp.py`：PDMP scaling、LP/finite Markov partition、fast-subnetwork 分析。
- `polymer_sim/experiment/`
  - `runner.py`：`ExperimentRunner.run_one(...)`，墙钟停止、restriction、recorder、timing report。
- `polymer_sim/recording/`
  - trajectory、summary、timing、plot、distribution comparison、fast network report；
  - trajectory 持久化当前决策：`save_trajectory_record(...)` 默认只写 sidecar 目录
    `trajectory_name/{times.npy, states.npy, species_names.json, metadata.json}`；
    不默认写 `.npz`。只有显式传 `write_npz=True` 才生成 legacy `.npz`。
    读取/分析入口应先检查 sidecar，sidecar 不存在时再 fallback 到 `.npz`。
    判断轨迹是否存在应使用 `trajectory_storage_exists(...)`，不要只用 `Path.exists()` 检查 `.npz` 路径。
  - 当前新增未跟踪 `cle_sparsity_sampler.py`，只做 CLE 稀疏性采样和画图，不改模拟状态。
- `polymer_sim/analysis/`
  - RAF/maxRAF/irrRAF 静态分析。

## stepper.py 当前重点

`SSAStepper`：

- Gillespie direct SSA；
- `use_local_propensity_updates=True` 时维护 propensity cache；
- reaction 后通过 `network.update_propensities_for_species(...)` 局部刷新；
- 抽样仍是 cumulative sum + search。

`OptimizedNRMStepper`：

- heap-based Next Reaction Method；
- 维护每个 channel 的 scheduled time、version 和 stale heap entry；
- reaction 后用 dependency graph 重排受影响 channel。

`PDMPStepper`：

- adaptive PDMP；
- discrete event locator 支持 scan NRM、heap NRM、Gillespie；
- 支持 local propensity update、partition adaptation、wall deadline；
- elementary network 路径可用 numba refresh。

`BlendedHybridStepper`：

- 当前 beta-hybrid 优化核心；
- 计算 channel beta，并按 beta 把 reaction 分成 jump 与 CLE contribution；
- `beta_fully_compute` 是全量 beta；
- `beta_compute_by_state_difference` 是 rounded state 差异驱动的局部 beta；
- `strict_int_for_CLE=True` 时 observed propensity 使用 rounded-state cache 和局部更新；
- 可能 fallback 到 full recompute；
- 支持 adaptive CLE dt、低拷贝数 rounding、负值 clipping、CLE sparsity sampling。

`NRMBlendedHybridStepper`：

- 继承 blended hybrid；
- mixed/pure discrete 部分用 split NRM event schedule。

修改 `BlendedHybridStepper` 时优先看这些函数：

- `_mixed_step`
- `_pure_cle_step`
- `_pure_ssa_step`
- `_cle_increment`
- `_adaptive_cle_increment`
- `_channel_betas`
- `_channel_betas_by_state_difference`
- `_update_channel_beta_cache_for_changed_species`
- `_propensities_for_observed_cached`
- `_observed_propensity_affected_channels_from_beta_hint`
- `_recompute_observed_propensity_cache`

## examples 新旧程度与目的

旧的稳定单次运行示例：

- `examples/minimal_run.py`：最小 API smoke。
- `examples/hs2014_formal_example.py`：HS2014/Wills-Henderson formal RAF 示例。
- `examples/without_catalyst.py`：无催化 polymer baseline。
- `examples/catalyst_run.py`：随机催化 polymer network。
- `examples/time_comparison.py`：较早的 timing/comparison 示例。
- `examples/plot.ipynb`、`examples/plot2.ipynb`：探索性绘图 notebook。

中期方法示例：

- `examples/blended_hybrid_minimal.py`：短 blended-hybrid smoke run。
- `examples/multiple_run.py`、`examples/multiple_run_core.py`：通用多次运行与 paired SSA/blended 流程。
- `examples/compute_strategy.py`：serial/thread/process CPU worker 策略。
- `examples/nrm_vs_ssa_multiple_run.py`：NRM vs SSA batch。
- `examples/fast_dimerization.py`：elementary benchmark network。
- `examples/oscillator.py`：构造的 catalytic oscillator-style network。

催化场景目录：

- `examples/MM_catalysis/`：当前 substrate-saturating/MM-style cross-catalysis 示例。
- `examples/linear_catalysis/`：linear-catalysis cross-catalysis 的 SSA/PDMP 变体，和 `cross_catalysis_SSA_linear.prof` 等 profile 相关。

当前最新/最重要批量对比目录：

- `examples/compare/common.py`：中心注册表，定义 network、method、settings、stepper factory、profile 和表格输出。
- `examples/compare/gillespie_ssa.py`、`optimized_nrm.py`、`gillespie_cle_hybrid.py`、`nrm_cle_hybrid.py`、`gillespie_pdmp_lp.py`、`nrm_pdmp_lp.py`、`strict_2018_pdmp.py`：单方法薄 wrapper。
- `examples/compare/batch_compare.py`：固定 wall-clock 的 cProfile 对比。
- `examples/compare/batch_compare_wall_times.py`：多个 wall-clock budget 的 sweep。
- `examples/compare/experiment_matrix.py`：当前 spreadsheet-style benchmark matrix，输出 `test_config.*`、`test_result.*`、`test_result_long.*`。
- `examples/compare/batch_compare_matrix.py`：`experiment_matrix.py` 的薄 CLI 入口。
- `examples/compare/a_polymer_update_matrix.py`：A-only len5/6/8，比较 SSA 和 blended local/global 更新。
- `examples/compare/beta_hybrid_update_matrix.py`：当前最相关性能测试，跑四种 beta-hybrid 组合：global/global、global/local、local/global、local/local。

现有 compare 输出：

- `examples/compare/outputs/20260808_114150/`：A-polymer len5/6/8，含 SSA、blended 变体和 cProfile。
- `examples/compare/outputs/20260808_113550/`、`20260808_114753/`、`20260808_123846/`、`20260808_124002/`、`20260808_124115/`、`20260808_124231/`：len10 beta-hybrid 压力测试。

## tests 新旧程度与目的

基础/较早测试：

- `tests/test_species.py`：species 顺序与初始 count。
- `tests/test_rules_and_channels.py`：join/split 表和 channel block 连续布局。
- `tests/test_run.py`：SSA、hybrid skeleton、timing report smoke。
- `tests/test_catalysis_assignment.py`：催化剂分配、reverse mirroring、强度缩放。
- `tests/test_hs2014_formal_example.py`：WH/HS2014 species、reaction、RAF、outflow。
- `tests/test_restriction.py`：food replenish、upper limit、food supply mode、runtime stop。
- `tests/test_recording_plots.py`：trajectory metadata 和 plotting。
- `tests/test_recording_summary.py`：batch summary、dt comparison、stepper metadata。
- `tests/test_updates_and_propensity.py`：state update、vectorized block propensity、sparse catalysis cache、local propensity update、substrate-saturating 公式。

当前算法测试：

- `tests/test_blended_hybrid_stepper.py`：当前最重要。覆盖 beta 分支、自适应 CLE dt、NRM blended、inflow/outflow、低 count rounding、beta local update、observed propensity cache reuse、changed catalyst channel、runner compatibility、CLE sparsity metadata。
- `tests/test_optimized_nrm.py`：optimized NRM heap/dependency 行为。
- `tests/test_pdmp.py`：elementary expansion、scaling partition、finite Markov fast subnetworks、PDMP runner、deadline、discrete locator、dependency CSR、numba refresh。
- `tests/test_compare_strict_2018_pdmp.py`：strict 2018 compare 注册、skip policy、constant/explicit food、finite Markov partition。

## 修改风险

- `ReactionNetworkData` 同时有 dense catalyst block 和 sparse catalyst cache，改催化相关逻辑必须确认 dependency indices 和 sparse cache 一起刷新。
- `BlendedHybridStepper` 里 beta cache、observed propensity cache、reuse hint、adaptive dt 相互耦合，不能只看单个函数。
- `strict_int_for_CLE=True` 并不保证每步都是 local update；affected channel 太多会 fallback full recompute。
- 修改 channel id layout 会影响 trajectory metadata、plotting 和大量测试。
- 修改 catalysis formula 会影响科学语义，必须同时更新测试。
- 文件中部分中文注释是 mojibake，除非专门清理编码，不要顺手改。

## 下一步建议

1. 改 `stepper.py` 后先跑：
   `python -m pytest tests/test_blended_hybrid_stepper.py`
2. 改 `network.py` propensity/catalysis 后再跑：
   `python -m pytest tests/test_updates_and_propensity.py tests/test_pdmp.py`
3. 性能验证优先用：
   `python examples/compare/beta_hybrid_update_matrix.py`
4. 新结果与 `docs/BENCHMARK.md` 对照，特别看：
   `simulation_final_time`、`n_steps`、`n_events`、`python_memory_peak_mb`、`process_rss_delta_mb`、`profiles/*_top40.txt`。
