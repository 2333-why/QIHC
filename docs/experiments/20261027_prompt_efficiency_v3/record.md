# 实验 03：约束 ID 修复与通用语义复核

- 实验 ID：`20261027_prompt_efficiency_v3`
- 状态：实现及本地验证已完成；服务器真实模型验收和正式实验均待运行。
- 日期：2026-10-27，Asia/Shanghai。
- 前一轮：[v2 部分结果](../20261027_prompt_efficiency_v2/analysis_20261027_partial_001.md)。新算法、新目录；不覆盖旧结果。
- 分支：`llm-cvrp`；代码发布提交将在完成验证后登记；服务器 `manifest.json.code_commit` 保存实际完整提交。
- 服务器结果目录：`/inspire/hdd/global_user/czxs25210022/why/results/prompt_efficiency_v3`。

## 目的与设置

| 项目 | 固定设置 |
|---|---|
| 问题 | 修复分步约束 ID 冲突；判断类型接口之外的语义复核是否有用 |
| 数据 | `results/co_comparison_v1`，预期 NLCO 240 + 合成 60；版本以源 cases SHA-256 固定 |
| 本地预检数据哈希 | `96ee820e4a0a07d10636d9816f3c5dccf57ef7352aa53c286f5949efc0b54752`；服务器另核对源 manifest，不假定一致 |
| 划分 | 每类最长原文共 8 题验收；正式 300 题，含验收题；已用于诊断，不是未见测试集 |
| 模型 | 源 manifest 指定，预期已下载 Qwen3-32B；无适配器，无后训练，thinking=false |
| 辅助 | 各组共享原题及标准模型的变量名称/顺序；不提供标准目标、约束或答案 |
| 环境 | `$BASE/envs/qihc-cu128`；Transformers 5.13.0、torch 2.9.1+cu128，LMFE 0.11.3、interegular 0.3.3；服务器实际值由预检输出记录 |
| 资源 | 单 worker、4 H100、80 CPU；2 个双卡副本 × batch 2；32 CPU 搜索进程 |
| 生成预算 | 各组输出总计 ≤16384 tokens，≤8 调用；flat 单步 ≤8192，staged ≤6144，语义复核 ≤8192 且只用剩余额度；每次生成 ≤900 秒 |
| 搜索预算 | p-bit 种子 0/1/2、16 链、每链 8192 候选评分；CP-SAT 单线程 5 秒 |
| 开关 | 无求解反馈、无后训练、固定能量策略；本轮只研究建模 |

## 对照与统计

| 比较 | 要回答的问题 |
|---|---|
| legacy_flat → typed_flat | 完整类型接口的效率/质量变化，保留原版一次建模 |
| typed_staged → canonical_staged | 相同初始提示与预算，统一编号是否解决组装失败 |
| typed_flat → semantic_flat | 一次建模后进行通用语义复核是否改善原题解 |
| canonical_staged → semantic_staged | 修复编号后，语义复核是否还有增益 |
| semantic_flat → semantic_staged | 两种初始生成方式在同一复核机制下的表现 |

每题先平均种子再平均题目；失败纳入主表分母。gap 仅限原题可行且有参考值的子集，列明样本数，不一律视为证明最优。报告同题配对可行率差、token 差、可行同种子 gap 差及描述性 bootstrap；不作未经校正的显著性宣称。

语义复核同样可能犯错：另外对每个复核请求自己的初始模型使用相同后端/种子/预算评分，记录改善、退步、保持不变及子集数量。初始模型诊断不向 GPU 回传，额外搜索时间单列。实际 tokens/FLOPs/时延不相等；这里只统一生成上限和各后端内部搜索预算。

## 实现与证据

- `canonical_builder.py` 按需求序号、规则类型及位置为 ID/prefix 命名，保存原名映射；系数、RHS、算子、逻辑表达式不变。
- `semantic_review.py` 读取原题和预测数学的通用诊断，让模型补全变量含义、量词与目标检查，并提交 2–6 个不同赋值的预期值。程序执行这些自检；没有标准答案反馈。
- `review_workflow.py` 保留初始输出；复核失败/预算不足回退初始可执行模型，原始复核输出和失败成本保留。成功复核仍标为 `semantic_correctness=unverified`。
- 新实验仅从 `experiments/prompt_efficiency_v3/` 进入；历史 v1/v2 实验文件未改。共享 runner 仅增加可选实验验收钩子，默认仍是原验收行为。
- 实际失败证据：v2 汇总中的重名错误；v1 原始归档中的错误目标与缺失独立量词。**没有 v2 逐题原始归档，不把 v1 原因冒充 v2 已证实原因。**

## 启动、续跑与结果

完整发布提交和 worktree 命令将在验证后补齐。进入独立固定提交 worktree 后：

```bash
# 联网 CPU：仅准备，无常驻计算，不重新下载模型
bash experiments/prompt_efficiency_v3/00_prepare_cpu.sh
# 四卡任务：启动、断点续跑均为这一条命令
bash experiments/prompt_efficiency_v3/01_run_4gpu.sh
# 报告自动更新，也可手动汇总
bash experiments/prompt_efficiency_v3/02_report.sh
BASE=/inspire/hdd/global_user/czxs25210022/why
cat "$BASE/results/prompt_efficiency_v3/status.json"
cat "$BASE/results/prompt_efficiency_v3/gate/report.md"
cat "$BASE/results/prompt_efficiency_v3/formal/report.md"
tar -czf "$BASE/prompt_efficiency_v3_analysis_$(date +%Y%m%dT%H%M%S).tar.gz" -C "$BASE/results" prompt_efficiency_v3
```

默认目录独立，CPU/GPU 自动启用项目解释器与依赖。不要在正在运行的 v2 worktree 执行 pull/switch。检查点锁及源/配置/响应哈希防止混跑；改变配置须新 run 目录。真实负载仍可能被平台回收，不承诺利用率阈值；恢复保留已发布子步骤和评分。

## 时间线与验证

- 2026-10-27：基于 v2 部分汇总启动独立 v3；确认本地只存在 v1 原始归档。
- 2026-10-27：完成统一编号、语义复核、回退、独立 CPU 归因和只读旧结果诊断。
- 2026-10-27：253 项相关测试通过（88.19 秒）；最终数学差异标签调整后，9 项 v3 测试再次通过（4.16 秒）。覆盖真实 v1 失败、ID 冲突、两后端 CPU 搜索、6 组报告、断点重放、旧结果只读诊断。
- 2026-10-27：实际 Qwen3-32B tokenizer 的 49 项类型/复核 schema 探针通过；1800 首轮提示全部通过长度检查，最大 14328 tokens；原版 legacy_flat 最大 10949。未载入 32B 权重。
- 2026-10-27：Ruff F 检查、三个 shell 脚本分别 bash -n、git diff --check 均通过。共享旧默认流程经相关回归测试；旧 v1/v2 实验源码未改。
- 预检源指纹：`ec70029916a9b816b258c7a70936f32851c6a0edf765554f50c64b8fb9a5e3f4`。详细证据：[validation.json](validation.json)。

## 结果与结论

| 证据 | 状态 |
|---|---|
| ID 修复、原数学不变、引用映射 | 本地回归通过 |
| 旧运行兼容、断点重放、真实 CPU 两后端 | 本地回归通过 |
| Qwen3-32B tokenizer/schema 与 1800 首轮输入 | 49 项探针及 1800 首轮长度检查通过；未载入权重 |
| 四卡真实模型验收、正式结果、语义复核收益 | 待返回，未宣称改善 |

分析与边界详见 [failure_analysis.md](failure_analysis.md)；运行与预算详见 [实验 README](../../../experiments/prompt_efficiency_v3/README.md)。
