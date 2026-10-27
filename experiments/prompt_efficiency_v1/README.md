# 提示词效率实验 v1

目标：减少提示与重复上下文负担，同时检验原题约束满足与解质量，不能仅凭 token 下降宣称更有效。
模型沿用已下载权重；不做后训练或搜索反馈修订。用 5 组固定提示，避免一次改变推理模式、格式约束和求解算法。

| 组 | 提示方式 | 配对作用 |
|---|---|---|
| legacy_flat | 当前 v6 一次建模原文 | 原版基线 |
| concise_flat | 精简指令，保留同一输出协议和源数据 | 对 legacy_flat 检验精简提示 |
| legacy_staged | 当前 v6 分步提示原文 | 原版分步基线 |
| concise_staged | 每步只提供该步语法；约束步骤只携带当前需求及必要状态 | 对 legacy_staged 检验减少重复上下文；对 concise_flat 检验分步流程 |
| semantic_staged | 精简分步 + 通用语义检查清单 | 对 concise_staged 检验量词、单位、目标局部赋值与遗漏自查 |

清单没有任何题型公式或完整题目答案示例。原题、数值、引用编号保持完整。控制组与改进组均有相同变量名称／顺序辅助（有标准模型时从其变量列表提取）；不是无辅助建模。
关闭 thinking、确定性生成、相同 schema 约束、每逻辑请求 16384 输出 token 上限、一次局部格式／数学纠错。
分步最多 8 次调用，单次最多 6144；flat 单次最多 8192。各组实际调用和输入成本不同，报告完整成本，不声称等 FLOPs。

## 数据与后端

默认读取 `results/co_comparison_v1` 已准备的 300 题：NLCO 240 题 + 合成题 60 道。
数据沿用原始固定版本，不联网下载新数据。不把此前已用于调试的题称为从未见过的测试集。
本轮要求每题有标准数学模型，用于独立可行性评分；仅有参考答案的数据会在准备阶段拒绝，需另建适配实验。
每组产生一个数学模型后，用同一模型分别运行 CP-SAT 和 p-bit：

- CP-SAT：每题 1 次，单线程，5 秒时间上限。
- p-bit：种子 0、1、2，16 条链，每链 8192 次候选评分预算，固定能量／更新策略、关闭干预。
- 不进行求解反馈修改模型；这样首先测提示词对建模与初始求解的影响。
- 先跑每类原文最长题共 8 道验收，每组至少一个模型成功编译且两种后端均实际搜索，才进入正式 300 题。
- 验收不要求最优或全部原题可行；语义质量如实计分，不因低分停掉实验。

正式阶段为 1500 个建模请求、最多 4500 次 p-bit 搜索和 1500 次 CP-SAT 调用。验收记录另存。

## 运行

建议独立固定提交的 worktree。联网 CPU 只运行准备与检查，原环境和权重继续复用：

```bash
bash experiments/prompt_efficiency_v1/00_prepare_cpu.sh
```

1 worker、4 H100、80 CPU 核任务执行：

```bash
bash experiments/prompt_efficiency_v1/01_run_4gpu.sh
```

内部两个双卡模型副本，每副本 batch 2；32 个 CPU 搜索进程与 GPU 并行。模型跨验收和正式阶段常驻。
相同固定 checkout 下重复 GPU 命令即可续跑。新配置另设 `QIHC_PROMPT_RUN`，CPU/GPU 必须一致。
不套 torchrun 或每卡重复启动。默认独立输出 `results/prompt_efficiency_v1`；准备时只读旧目录的数据与模型配置，运行结果不写入旧目录。

每 30 秒输出进度和当前阶段的生成 ETA（至少新增 10 个请求后估计，不含后续阶段及 CPU 收尾）。
GPU 子进程异常时主日志回显该进程日志末尾；子步骤写盘，可定位失败步骤。
有效计算优先并行，但不能保证平台不回收：两副本无待生成请求且 CPU 未完成超过 15 分钟时保存状态退出；不使用空算或显存填充。

## 查看、汇总与打包

CPU 可随时运行，报告会在 GPU 任务中自动更新：

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
PYTHONPATH="$PWD" "$BASE/envs/qihc-cu128/bin/python" -m experiments.prompt_efficiency_v1.run status
bash experiments/prompt_efficiency_v1/02_report.sh
cat "$BASE/results/prompt_efficiency_v1/formal/report.md"
tar -czf "$BASE/prompt_efficiency_v1_analysis.tar.gz" -C "$BASE/results" prompt_efficiency_v1
```

`manifest.json` 固定代码指纹、数据、模型元数据与设置；`record.md` 记录入口；`status.json` 为当前状态。
`record.md` 自动追加启动、阶段切换、验收、失败或完成时间线；`prompt_examples/` 导出五组真实首轮提示。
`gate/` 和 `formal/` 内各有 requests、responses、builder_steps、scores、report.md、summary.json。
按题先平均种子，再平均题目。失败计入完成题分母；未完成单列。配对 gap 仅比较相同种子下双方可行且有参考值的题。
报告每组输入／输出 token、调用数、分摊生成耗时和搜索耗时；分摊时间不是单请求独占 GPU 的延迟。
配对 bootstrap 区间仅作描述，未作多重比较显著性声明。可行率不能证明整个模型数学等价。

实验记录：[本轮文档](../../docs/experiments/E001_prompt/E001-1.1_record.md)。
