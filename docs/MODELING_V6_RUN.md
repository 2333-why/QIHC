# Modeling v6：分步建模与局部纠错实验

本版增强 Agent 的建模与修订流程，不修改模型权重，不启动 SFT/DPO。
独立入口 `tools/runtime/modeling_v6/`，复用固定的 v5 数学编译器和原文引用接口；不修改旧 runtime、数据或结果。
不按题型识别并套用解法。当前范围仍为有界整数/布尔变量、线性约束、二次目标及已支持的逻辑表达式；不宣称任意优化问题均可处理。

## CPU 准备与四卡启动

在新代码的独立 checkout 内，联网 CPU 执行：

```bash
bash tools/runtime/modeling_v6/00_prepare_format.sh
```

已装 v5 的小型格式依赖时直接复用，不升级环境或下载模型。没有时只安装固定版本 `lm-format-enforcer==0.11.3` 和 `interegular==0.3.3`，默认清华源。
CPU 用本地真实 tokenizer 检查五种输出 schema 与两类初始提示长度，不加载模型权重。

若旧版本准备脚本出现 `cannot import name 'PreTrainedTokenizerBase'`，紧接着提示
`transformers is not installed`，这是 LM Format Enforcer 0.11.3 的旧 Transformers 适配器误报。
修复版通过 `token_bridge.py` 将同一个 TokenEnforcer 核心接到公开的 `prefix_allowed_tokens_fn` 接口，
不再导入旧适配器，不修改已安装的 Transformers/Torch，也不禁用格式约束。
准备脚本会打印实际使用的项目 Python 路径和依赖版本；不要在 `/etc/inspire` 环境用裸 `pip` 重装 Transformers。
CPU 准备失败后保留的 `.format-v6-*` 临时目录不影响再次执行准备脚本，不需要重新下载模型。
兼容回归包含 Transformers 5.13.0 的真实 tokenizer、五种 schema 过滤与随机初始化小模型的批量生成；
这验证的是接口连接，不是 Qwen 权重或 H100 实验效果。

1 worker、4 H100、80 核 CPU 的任务启动命令：

```bash
bash tools/runtime/modeling_v6/01_run_4gpu.sh
```

自动启用原环境，运行真实模型验收、正式实验和报告。不要套 torchrun。
默认 32 个 CPU 搜索进程、2 个双卡模型副本、每个副本 batch 上限 2；相同 schema 的请求优先合批。
联网 CPU 不承担正式搜索。重复同一固定 checkout 中的启动命令可续跑，包含已保存的建模子步骤。
不要与旧任务共占同一组 GPU；脚本不停止旧进程，不用无效负载占卡，不能承诺平台绝不回收。

默认读取 `why/results/co_comparison_v1` 的已封存数据与模型位置。
可用 `QIHC_DISTRIBUTION_SOURCE` 指定另一份同格式来源，或用 `QIHC_DISTRIBUTION_RUN` 指定新的独立输出目录。
正式实验复制来源的全部实例；原来源为 300 题时即 300 题，不在脚本中写死题量。

## 新的建模与纠错过程

1. 计划：定义变量，逐条列出原文需求、类型、证据和预期标量约束数；最多 64 条需求。
2. 目标：单独生成目标函数，引用原文数据，检查方向、单位和局部取值含义。
3. 约束：需求最多分为 4 批，逐批生成约束；表格和数组可通过通用 `forall_rows`、`sum_rows`、source 引用展开。
4. 编译：确定性地合并模型并检查维度、索引、ID 和数值范围。预期约束数保留模型原预测，不用实际展开数覆盖它来美化审计。
5. 搜索与反馈：按实验组提供最佳解、分布或约束干预证据。标准模型的系数、约束、答案不进入反馈。
6. 修订：仅修改选定目标、变量边界或需求对应的约束组；未指定部分保持不变。允许补充漏掉的需求。

若把取值域误写成固定赋值，可附证据说明并纠正需求类型，移除错误数学约束，但不能悄悄删除原文需求记录。
局部修改是原子操作，编译失败不会污染已接受状态；整体失败不会把半份模型标成成功。
结构与维度检查无法证明语义正确，需求清单仍可能漏项，模型也可能选择错误数据源。

## 11 组对照

| 组 | 初始建模 | 修订方式 | 搜索证据 / 策略 |
|---|---|---|---|
| flat_self_review | 一次完整生成 | 完整重写 | 自查，无搜索证据 / 固定 |
| flat_selected | 一次完整生成 | 完整重写 | 分布与干预 / 联合选择 |
| staged_self_review | 分步 | 局部修改 | 自查，无搜索证据 / 固定 |
| staged_best | 分步 | 局部修改 | 最佳解 / 固定 |
| staged_distribution | 分步 | 局部修改 | 最佳解与分布 / 固定 |
| staged_probe_blind | 分步 | 局部修改 | 执行干预但隐藏干预结果 / 固定 |
| staged_intervention | 分步 | 局部修改 | 展示干预结果 / 固定 |
| staged_full_review | 同一分步初始模型 | 完整重写 | 同 staged_intervention |
| staged_selected | 分步 | 局部修改 | 分布与干预 / 联合选择 |
| staged_metropolis | 同一分步初始模型 | 局部修改 | Metropolis，分布与干预 / 联合选择 |
| staged_cp_sat | 同一分步初始模型 | 局部修改 | CP-SAT 最佳解反馈，单独计时参照 |

关键配对自动报告：

- 分步建模收益：staged_self_review 对 flat_self_review 的初始阶段；修订阶段则包含建模和修订方式的共同差异。
- 完整流程收益：staged_selected 对 flat_selected；这是组合改进，不能只归因于某一个模块。
- 局部修订收益：staged_intervention 对 staged_full_review；相同初始模型、求解策略和反馈证据。
- 搜索反馈、分布信息：staged_best 对 staged_self_review；staged_distribution 对 staged_best。
- 干预搜索与干预信息分别比较：staged_probe_blind 对 staged_distribution；staged_intervention 对 staged_probe_blind。
- 策略联合选择、采样后端：staged_selected 对 staged_intervention；staged_selected 对 staged_metropolis。

两类初始模型各生成一次，各自在所属对照组之间共享。
每次逻辑初始建模/反馈修订的总输出上限相同，均为 16,384 tokens，最多一次局部错误修复。
分步成功路径为 2～6 次生成；完整生成通常 1 次，失败可再修复 1 次；局部修订通常 1 次。
分步方式增加输入 tokens 和调用次数，因此不是等 FLOPs/等调用实验。报告记录输入/输出 tokens、实际生成次数和分摊推理时间；修订阶段的配对成本包含初始阶段。
完整生成单次上限 8,192，分步单次上限 6,144；所有错误重试消耗同一个总预算，不额外扩容。

默认每组每阶段每种子预算为 8,192×16 条链的候选评分上限，包括探测；正式采样 3 个种子，验收 1 个种子。
CP-SAT 每阶段一次、5 秒上限，仅作为单独时间预算参照，不能声称同能耗或同候选次数。
保留基准提供的变量名/顺序辅助，所有前端相同；这并非完全无布局辅助的自然语言建模。

## 验收、报告与断点

真实 GPU 验收覆盖 flat_selected、staged_selected、staged_full_review 三条路径；所有初始/修订结果必须真实生成、数学合法并实际搜索。
原题可行率、引用和覆盖检查分别报告，验收通过不代表语义正确或高解质量。
持续执行失败触发熔断。成功/失败子步骤及其提示、响应、哈希、token 和耗时保存在 `builder_steps/`，已完成子步骤在续跑时不重新生成。
被强制中断且尚未写出检查点的当前生成步骤需要重做；单请求 OOM 保留未完成状态并记录额外耗时。

输出全部在 `why/results/`：

| 路径 | 内容 |
|---|---|
| distribution_feedback_modeling_v6_gate/gate_acceptance.json | 验收检查完成后写出 |
| distribution_feedback_modeling_v6_gate/job_logs/ | 验收进程日志 |
| distribution_feedback_modeling_v6/summary.md、summary.json | 各数据集、各组结果与审计 |
| distribution_feedback_modeling_v6/builder_steps/ | 子步骤检查点、局部修复和调用记录 |
| distribution_feedback_modeling_v6/job_status.json、health.json | 活跃队列、错误与熔断状态 |
| distribution_feedback_modeling_v6_reports/report.md | 实例配对比较与条件 ETA |
| distribution_feedback_modeling_v6_reports/paired_statistics.json | Bootstrap 区间、Holm 校正、成本差 |

按实例平均种子，失败实例计入分母。只有标准数学模型的题报告独立可行率；仅有参考答案的题报告答案一致率，两者不混为“正确率”。
gap 对比只使用相同种子下双方都可行的结果，并与可行率共同展示。
ETA 按真实逻辑请求分摊时间估计，区分完整/分步初始建模以及修订类型；早期数据不足时显示待校准。
本轮不自动生成 SFT 正例或 DPO 偏好标签；低能量、成功执行、一次可行解均不是正确建模证明。

查看状态（模型验收阶段把目录改为 `_gate`）：

```bash
/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128/bin/python tools/runtime/modeling_v6/experiment.py status --run /inspire/hdd/global_user/czxs25210022/why/results/distribution_feedback_modeling_v6
```

## 参考设计

- [OptiMUS-0.3](https://arxiv.org/html/2407.19633v3)：分解建模与阶段纠错。
- [LEAN-LLM-OPT](https://arxiv.org/html/2601.09635v1)：工具承担机械数据处理，借鉴其通用工作流，不复现题型分类模板。
- [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer)：结构约束解码；数学与语义检查另行进行。

本实现是独立方法改进与对照，不冒充上述论文的原版复现，也没有本地 H100/Qwen 效果结论。
