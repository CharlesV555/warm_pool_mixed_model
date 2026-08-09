# 性能测试记录

更新时间：2026-08-08。

本文只整理当前工作区已有结果，没有重新跑 benchmark。

## 已知结果来源

现有性能/诊断文件：

- `examples/compare/outputs/20260808_114150/experiment_matrix/`
- `examples/compare/outputs/20260808_113550/experiment_matrix/`
- `examples/compare/outputs/20260808_114753/experiment_matrix/`
- `examples/compare/outputs/20260808_123846/experiment_matrix/`
- `examples/compare/outputs/20260808_124002/experiment_matrix/`
- `examples/compare/outputs/20260808_124115/experiment_matrix/`
- `examples/compare/outputs/20260808_124231/experiment_matrix/`
- `blended_cprofile_out.txt`
- `cross_catalysis_PDMP_SSA_linear_out.txt`
- `timing_reports/`
- `cle_sparsity.png`

`*.prof` 是 cProfile 原始文件；`*_top40.txt` 是可读 profile 摘要。

## A-polymer 矩阵：2026-08-08 11:41-11:42

来源：

- `examples/compare/outputs/20260808_114150/experiment_matrix/test_result_long.csv`

条件：

- 每个 cell 约 60 秒 wall-clock；
- food mode：constant；
- network：A-only len5、len6、len8；
- 全部 stop reason：`max_runtime_seconds`。

| config | network | method | sim time | steps | events | wall s | peak MB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| blended_global_beta_global_propensity | len5 A | gillespie_cle_hybrid | 0.0206457 | 11108 | 69 | 60.139 | 7.40 |
| blended_local_beta_global_propensity | len5 A | gillespie_cle_hybrid | 0.0206468 | 11121 | 69 | 60.151 | 7.41 |
| blended_local_beta_local_propensity | len5 A | gillespie_cle_hybrid | 0.0207679 | 12993 | 80 | 60.172 | 8.47 |
| ssa | len5 A | gillespie_ssa | 0.0143385 | 14454 | 14454 | 60.153 | 10.36 |
| blended_global_beta_global_propensity | len6 A | gillespie_cle_hybrid | 0.0364580 | 11498 | 282 | 60.131 | 8.33 |
| blended_local_beta_global_propensity | len6 A | gillespie_cle_hybrid | 0.0364559 | 11450 | 282 | 60.158 | 8.31 |
| blended_local_beta_local_propensity | len6 A | gillespie_cle_hybrid | 0.0365519 | 13338 | 334 | 60.164 | 9.48 |
| ssa | len6 A | gillespie_ssa | 0.0279086 | 14497 | 14497 | 60.155 | 11.28 |
| blended_global_beta_global_propensity | len8 A | gillespie_cle_hybrid | 0.101222 | 11722 | 429 | 60.157 | 9.02 |
| blended_local_beta_global_propensity | len8 A | gillespie_cle_hybrid | 0.101222 | 11741 | 429 | 60.150 | 9.05 |
| blended_local_beta_local_propensity | len8 A | gillespie_cle_hybrid | 0.100401 | 13291 | 517 | 60.179 | 10.11 |
| ssa | len8 A | gillespie_ssa | 0.0549916 | 14558 | 14558 | 60.179 | 12.00 |

初步解释：

- A-only constant-food 网络中，blended hybrid 在相同 wall-clock 下推进的 simulation time 明显多于 SSA。
- local beta 对 global propensity 路径帮助不明显。
- local observed propensity 增加了 step/event 数和内存，但这轮没有明显提升 simulation time。

## len10 two-stage beta-hybrid 压力测试

来源：

- `20260808_113550`
- `20260808_114753`
- `20260808_123846`
- `20260808_124002`
- `20260808_124115`
- `20260808_124231`

共同条件：

- network：`polymer_len10_two_stage_1_catalysis`
- species：2046
- channels：10216
- food mode：constant restriction
- stop reason：`max_runtime_seconds`

| timestamp | config | sim time | steps | events | wall s | peak MB | RSS delta MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260808_113550 | beta_hybrid_local_beta_local_propensity | 0.0349354 | 26962 | 616 | 62.778 | 2394.27 | 2081.06 |
| 20260808_114753 | beta_hybrid_global_beta_global_propensity | 0.0349613 | 19919 | 617 | 62.491 | 1939.46 | 1710.48 |
| 20260808_123846 | beta_hybrid_local_beta_local_propensity | 0.0349354 | 26357 | 616 | 62.799 | 2355.22 | 2046.46 |
| 20260808_124002 | beta_hybrid_local_beta_global_propensity | 0.0349613 | 19374 | 617 | 62.706 | 1904.28 | 1682.24 |
| 20260808_124115 | beta_hybrid_global_beta_local_propensity | 0.0349354 | 26363 | 616 | 62.962 | 2355.61 | 2046.98 |
| 20260808_124231 | beta_hybrid_global_beta_global_propensity | 0.0349613 | 19334 | 617 | 62.662 | 1901.70 | 1679.95 |

初步解释：

- 四种 beta/propensity 组合推进到几乎相同的 simulation time。
- local observed propensity 模式 step 更多、内存更高，peak 约 2.35-2.39 GB。
- global observed propensity 模式 peak 约 1.90-1.94 GB。
- 在这个 len10 压力测试里，local observed propensity cache 暂时不是明确收益点。

## cProfile：blended hybrid，2026-08-05

来源：

- `blended_cprofile_out.txt`

摘要：

- `blended.prof`
- 50,894,566 calls
- 66.690 seconds

主要项目热点：

- `stepper.py:3144(step)`：58.424 s cumulative。
- `stepper.py:3268(_mixed_step)`：56.294 s。
- `stepper.py:3656(_propensities_for_observed_cached)`：50.324 s。
- `network.py:742(compute_propensities_for_channels)`：47.332 s。
- `network.py:1184(_apply_block_catalysis)`：41.542 s。
- `network.py:1310(_sparse_substrate_saturating_factors)`：35.346 s。
- NumPy reductions，如 `sum`、`any`、ufunc reduce，也占比较高。

解释：

- 最大瓶颈不是 beta 本身，而是 blended stepper 调用的 observed propensity 与 catalytic propensity。
- `_sparse_substrate_saturating_factors` 是 network 层最明显瓶颈。
- profile 中也有 matplotlib text layout 成本，优化模拟时应优先看项目函数。

## cProfile：A-polymer len8 blended local/local

来源：

- `examples/compare/outputs/20260808_114150/experiment_matrix/profiles/blended_local_beta_local_propensity__polymer_a_len8_a8_catalyzes_a_constant_food__gillespie_cle_hybrid_top40.txt`

摘要：

- 11,610,519 calls
- 60.395 seconds

主要热点：

- `experiment_matrix.py:312(_run_matrix_cell_body)`：60.405 s。
- `runner.py:30(run_one)`：60.153 s。
- `stepper.py:3212(step)`：54.691 s。
- `stepper.py:3553(_adaptive_cle_increment)`：49.561 s。
- `stepper.py:3230(_pure_cle_step)`：49.335 s。
- `stepper.py:3480(_cle_increment)`：48.267 s。
- `stepper.py:3998(_propensities_for_observed_cached)`：18.124 s。
- `stepper.py:4141(_recompute_observed_propensity_cache)`：17.556 s。
- `network.py:746(compute_all_propensities)`：16.898 s。
- `network.py:1201(_apply_block_catalysis)`：11.419 s。

额外现象：

- SciPy sparse slicing/matrix 操作进入 top40，说明 sparse stoichiometry 路径如果访问方式不好，也会成为瓶颈。

## cProfile：A-polymer len8 SSA

来源：

- `examples/compare/outputs/20260808_114150/experiment_matrix/profiles/ssa__polymer_a_len8_a8_catalyzes_a_constant_food__gillespie_ssa_top40.txt`

摘要：

- 9,723,345 calls
- 60.517 seconds

主要热点：

- `runner.py:30(run_one)`：60.155 s。
- `stepper.py:297(step)`：53.867 s。
- `stepper.py:328(_step_from_channels)`：36.084 s。
- `stepper.py:405(_update_cached_propensities)`：32.548 s。
- `network.py:815(update_propensities_for_species)`：32.420 s。
- `network.py:1201(_apply_block_catalysis)`：32.266 s。
- `network.py:759(compute_propensities_for_channels)`：29.550 s。
- `network.py:746(compute_all_propensities)`：17.489 s。
- `network.py:1328(_sparse_substrate_saturating_factors)`：10.584 s。
- `trajectory.py:108(record_step)`：3.624 s。

解释：

- SSA local update 已经启用，但受影响 catalytic propensity 重算仍很贵。
- recorder 有成本，但不是第一瓶颈。

## cProfile：cross-catalysis linear，2026-08-05

来源：

- `cross_catalysis_PDMP_SSA_linear_out.txt`

摘要：

- `cross_catalysis_SSA_linear.prof`
- 12,589,339 calls
- 11.637 seconds

主要热点：

- `stepper.py:3134(step)`：5.801 s。
- `stepper.py:3255(_mixed_step)`：5.397 s。
- `stepper.py:3593(_propensities_for_x)`：4.165 s。
- `network.py:729(compute_all_propensities)`：4.129 s。
- `stepper.py:3422(_adaptive_cle_increment)`：2.729 s。
- `stepper.py:3382(_cle_increment)`：2.662 s。
- `network.py:1184(_apply_block_catalysis)`：2.465 s。
- `network.py:1154(_compute_block_base_propensities)`：1.279 s。

解释：

- 即使是较小的 linear-catalysis profile，full propensity recompute 和 block catalysis 仍是核心成本。
- 该结果早于 2026-08-08 beta-hybrid matrix，只作为辅助证据。

## timing_reports

`timing_reports/` 中有大量 2026 年 7 月到 8 月的 JSON/PNG：

- `fast_dimerization_pdmp*.json`：较早 PDMP timing。
- `fast_dimerization_strict_2018_pdmp*.json/png`：2026-07-31 strict 2018 PDMP。
- `repressilator_strict_2018_pdmp*.json/png`：2026-07-31 repressilator strict 2018。
- `timing_seed_123_063` 到 `timing_seed_123_068`：2026-08-05 到 2026-08-08 近期 timing。
- `timing_seed_124` 到 `timing_seed_124_006`：2026-08-05 timing。

PNG 命名：

- `*_events.png`：event count vs wall time。
- `*_simulation_clock.png`：simulation time vs wall time。
- `*_dt_cle_metrics.png`：dt/CLE 相关诊断。

## 当前优化判断

根据现有结果：

1. 单独优化 beta 计算，收益可能有限。
2. 更大瓶颈是 `stepper.py` 调用的 propensity 重算和 `network.py` 催化因子。
3. local observed propensity cache 在 len10 上暂未显示 wall-clock 优势。
4. `_sparse_substrate_saturating_factors` 的 selected-channel 路径是重点。
5. CLE sparse stoichiometry 要小心 SciPy sparse slicing 开销。
6. recorder 成本存在，但优先级低于 propensity/catalysis。

## 建议 benchmark 协议

改 `stepper.py` 后：

1. 先跑：
   `python -m pytest tests/test_blended_hybrid_stepper.py`
2. 短 benchmark：
   `python examples/compare/beta_hybrid_update_matrix.py --networks polymer_a_len5_a5_catalyzes_a_constant_food --wall-seconds 10 --workers 1`
3. len10 压力测试：
   `python examples/compare/beta_hybrid_update_matrix.py --networks polymer_len10_two_stage_1_catalysis --configs beta_hybrid_global_beta_global_propensity --wall-seconds 60 --workers 1`
4. 对比 `test_result_long.csv`：
   `simulation_final_time`、`n_steps`、`n_events`、`python_memory_peak_mb`、`process_rss_delta_mb`。
5. 对比 `profiles/*_top40.txt`：
   `stepper.py`、`network.py`、`scipy.sparse`、`trajectory.py`。
