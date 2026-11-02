# E004-1.4：八小时主动反例反馈实验

18 题×六组×单种子×两轮，共 108 条轨迹、最多 234 次新 LLM 调用；最多 756 次七后端交叉求解。独立入口，不修改历史 runtime 或共享模块全局设置。

实验记录：[E004-1.4_record.md](../../docs/experiments/E004_feedback/E004-1.4_record.md)。算法基础：[E004-1.3_plan.md](../../docs/experiments/E004_feedback/E004-1.3_plan.md)。本地测试不等于服务器四卡验收或效果验证。

## 服务器启动

要求已有完整 E004-1.2 结果目录（包含 manifest、cases.public.json、initial、evaluation），四张可加载 Qwen3-32B 的 GPU 和至少 24 个分配给该作业的 CPU 核；沿用原服务器模型、格式约束和七求解器依赖，不自动安装或下载。

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc
git fetch origin codex/e004-1.4-witness-8h
git worktree add --detach ../qihc-E004-1.4-git origin/codex/e004-1.4-witness-8h
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.4-git
bash experiments/feedback_witness_v1/00_check_cpu.sh
# 在单个四卡作业内运行；同一命令在原截止时间内续跑
bash experiments/feedback_witness_v1/01_run_4gpu.sh
```

若服务器 E004-1.2 的运行目录不是默认路径，启动前设置：

```bash
export QIHC_PILOT_SOURCE=/实际路径/E004-1.2_feedback_pilot
```

可覆盖 `QIHC_BASE`、`QIHC_ENV_PREFIX`、`QIHC_MODEL`、`QIHC_FORMAT_DEPS`、`QIHC_SOLVER_DEPS`、`QIHC_WITNESS_RUN`。默认结果目录为 `$QIHC_BASE/results/E004-1.4_feedback_witness_8h`，不会续写 E004-1.2。

## 状态、报告与结果包

```bash
cat /inspire/hdd/global_user/czxs25210022/why/results/E004-1.4_feedback_witness_8h/status.json
bash experiments/feedback_witness_v1/02_report.sh
bash experiments/feedback_witness_v1/03_pack.sh
```

正常退出或可处理的异常后自动生成 `report.md`、`summary.json`、`cross_matrix.json`，并打包为结果目录旁的 `E004-1.4_feedback_witness_8h_analysis.tar.gz`。强制断电/硬超时可能来不及打包，可事后执行 `03_pack.sh`；这只导出已有结果，不重跑实验。

`clock.json` 固定原始截止时间，续跑不重置。监督器含准备、加载、反馈、离线评估和打包共八小时上限；7.5 小时停止计算。约定请求 180 秒、补丁请求 300 秒，另有 15 秒回收窗口，不自动重试。GPU 卡死会停止新请求并记录失败，不承诺故障下 108 条都有成功结果。

## 实现细节与研究边界

- 共享约定由原文单独生成一次，是带原文引文的可执行 LLM 假设，绝非 gold。补丁提示只获得它的引文、状态和要求数量；完整假设只供证据计算/验收，避免直接给各组一个可复制的答案模型。
- 两个原子族是 `objective` 与包含全部线性/逻辑要求的 `constraints`。声明替换后返回完整模型，先在副本编译和验收，再提交。支持通用 `forall_rows`；有效初稿的变量域和映射不可更改，无效初稿可从原文重建。
- 覆盖检查采用假设因子逐项覆盖，可能拒绝不同表达的等价约束；有限上下文检查也不能证明语义。该保守验收各组一致，拒绝及原候选均保存。独立离线核验允许正比例目标、忽略因子 ID，并以反例/枚举或结构恒等区分正确、错误、未判定。
- P/R 共用 p-bit 基础状态，M/X 使用其自身探针状态，A 直接检查全部二次变量对并加入固定随机上下文。所有组最终搜索均为 p-bit，随后冻结模型给七个求解器，隔离反馈和最终求解贡献。
- 反馈保留同上下文的四状态及目标/惩罚对比；只展示最多八个非零惩罚贡献，其余数量和贡献和明确保留。最多两张完整证据卡，超过 token 配额整卡省略并记数量。
- 隐藏标准模型只进入离线 evaluator 子进程；反馈请求和探针任务不读取 evaluator 数据。已观察的 18 题是开发诊断集，不能据此作正式未见性能结论。
- 无效初稿、格式/超时失败和语义未判定均保留。`all_stages_processed` 只表示处理终结，绝非全部成功；真实评分、正确性、失败和阶段成本分别报告。
- 单次 30 秒语义核验不保证全量确认，数据绑定含义未单独获得证明时明确未单独认证；不能以格式通过、约定通过或一个最优解代替完整建模准确性。

## 本地验证

```bash
python -m experiments.feedback_witness_v1.budget
python -m pytest tests/test_feedback_witness_v1.py -q
```

测试需要 numpy、jsonschema、pytest、lm-format-enforcer 0.11.3、interegular 0.3.3、SCIP、HiGHS、OR-Tools。模拟 LLM 的两轮流程验证调用数、失败分母和续跑；真实 CPU 子进程验证七后端与五种探针来源。真实 Qwen3-32B/四卡仍在服务器启动时检查。
