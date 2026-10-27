# 通用语义复核与确定性约束编号 v3

实验记录：[record.md](../../docs/experiments/20261027_prompt_efficiency_v3/record.md)。历史证据和边界：[failure_analysis.md](../../docs/experiments/20261027_prompt_efficiency_v3/failure_analysis.md)。

## 六组配对实验

| 组 | 处理方式 | 关键比较 |
|---|---|---|
| legacy_flat | 原版一次建模，保留原提示、宽 schema 和编译重试 | 与 typed_flat 比较完整类型接口 |
| typed_flat | v2 类型接口，一次建模 | 原样保留，不把语义复核混入基线 |
| typed_staged | v2 分步建模，保留原 ID 行为 | 与 canonical_staged 隔离组装修复 |
| canonical_staged | 相同分步提示和预算；程序统一 ID、映射需求证据 | 检查编译失败是否减少 |
| semantic_flat | typed_flat 成功后，追加一次原题语义复核 | 对比 typed_flat；另评估自己的初始模型 |
| semantic_staged | canonical_staged 成功后，同样追加语义复核 | 对比 canonical_staged；另评估自己的初始模型 |

统一编号只改变约束名称及对应证据目标，不合并、删除约束，不改系数、方向或 RHS。原始 ID、所属需求、位置与新 ID 均保存；重复别名按一对多记录。重复 JSON 键、错误引用、越界索引和非法数学仍会失败。

复核读取完整原题、初始模型、通用数学诊断：单变量强制等式、局部目标赋值、引用疑点、需求覆盖。模型需重新检查变量含义、量词、数值和目标，并返回修正模型及 2–6 个不同赋值的预期目标/可行性。程序执行赋值检查；不自洽、输出截断或不能编译时回退初始可执行模型，保留全部失败输出与成本。可以增删有原文依据的约束，但不会根据 gold 或求解分数选择修改。

这是基于原题的自检；错误模型和错误预期仍可能相互自洽。引文/覆盖问题继续单独审计，不混同于数学合法性。接受复核不等于语义正确。未使用题型名称选公式，未增加后训练或求解反馈。

## 预算与验收

- 数据复用 `results/co_comparison_v1`：预期 NLCO 240 + 合成 60。固定数据哈希；当前集合已经用于诊断，不能宣称未见测试集。全部组共享来自标准模型的变量名称/顺序，标准目标、约束和答案仅 CPU 评分使用。
- 复用已下载模型，预期 Qwen3-32B，以源 manifest 为准。累计输出 16384 tokens、8 次调用，包括编译重试及语义复核；初始阶段提示和单步上限与对应基线一致。复核仅用剩余预算，最多 8192 tokens，一次尝试。未留足预算会记录 `skipped_budget`，不扩大额度。
- 每组 CP-SAT 单线程最多 5 秒；p-bit 种子 0/1/2、16 链、每链最多 8192 次候选评分。两求解器预算不同，不宣称等成本优势。
- 验收：8 题 × 6 组；每组至少一个模型编译并进入两个后端搜索；两个复核组还须各至少一次自检通过，否则熔断，不启动正式 300 × 6 组。
- 正式阶段至多 1800 个建模请求、5400 次 p-bit 和 1800 次 CP-SAT 主评测。复核组另做初始模型诊断评分，最多追加 600 次 CP-SAT 和 1800 次 p-bit；初始模型与最终模型完全一致时复用评分。额外诊断耗时单列，不向 LLM 回传。
- 四卡单 worker：两个常驻双卡副本，各 batch 2；32 个 CPU 搜索进程。联网 CPU 仅准备，GPU 任务内同时运行建模、搜索和报告。继承断点、锁、哈希验证、ETA 和失败熔断。真实负载不能保证平台不回收；不制造虚假占用。

## 运行

在独立固定提交的 worktree 中，联网 CPU 运行一次：

```bash
bash experiments/prompt_efficiency_v3/00_prepare_cpu.sh
```

复用原环境、格式依赖、数据和权重；缺少格式依赖时仅从国内镜像下载两个固定小包。真实 tokenizer 检查完成后，联网 CPU 可退出。在 1 worker、4 H100、80 CPU 的任务中运行一次：

```bash
bash experiments/prompt_efficiency_v3/01_run_4gpu.sh
```

同一固定提交、同一配置重复该命令即续跑；不要套 `torchrun`。结果独立写到 `/inspire/hdd/global_user/czxs25210022/why/results/prompt_efficiency_v3`，旧 v2 worktree 和结果不修改。自定义 `QIHC_PROMPT_V3_RUN` 时准备与启动保持一致。

报告自动更新，也可执行：

```bash
bash experiments/prompt_efficiency_v3/02_report.sh
cat /inspire/hdd/global_user/czxs25210022/why/results/prompt_efficiency_v3/status.json
cat /inspire/hdd/global_user/czxs25210022/why/results/prompt_efficiency_v3/formal/report.md
```

汇总包含编译率、原题可行率、可行子集参考 gap、调用/token/时间成本、按题配对差值、纠错前后同一初始模型的改善/退步，以及逐解失败分类。未完成不算完成；无可行 gap 样本不填零。自检收益只在有初始可执行模型的子集上计算，并报告该子集题数；主表仍包含全部失败。

## 对旧 typed_flat 做只读诊断

在新 worktree、联网 CPU 上执行（不需要 GPU，不调用模型或求解器）：

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
PYTHONPATH="$PWD" "$BASE/envs/qihc-cu128/bin/python" -m experiments.prompt_efficiency_v3.diagnose_existing --run "$BASE/results/prompt_efficiency_v2" --output "$BASE/results/prompt_v2_diagnosis_$(date +%Y%m%dT%H%M%S)"
```

输出必须位于旧结果目录之外，且不能覆盖已有诊断。`report.md` 给出逐题/后端/种子的归因，`diagnosis.json` 保存生成模型的诊断与违反的原题约束。运行中的旧实验按当前已发布评分读取，因此只是快照。把该诊断目录与 v2 原始归档一起返回，才能确定具体错误公式与修复比例。
