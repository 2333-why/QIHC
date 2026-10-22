# 在不改动作业的前提下，自动统计关键对照和估算 ETA

新增工具位于 `tools/analysis/distribution_feedback/`，**不在旧 CO 或新分布反馈实验的源码哈希覆盖目录内**。不修改两套运行器、实验 manifest、请求/响应、检查点或旧 summary；不重启、不迁移、不重跑实验。可以分析已运行或已完成的 `distribution_feedback_v1`。

## 运行

建议在 CPU 节点建立独立的分析 worktree，不动正在运行的代码目录。下面的 worktree 创建命令只需执行一次：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && git fetch origin && git worktree add --detach ../qihc-report origin/llm-cvrp
```

一次生成当前快照：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-report && bash tools/analysis/distribution_feedback/04_report.sh
```

每 60 秒自动更新，全部实例完成后退出；Ctrl+C 只停止分析，不影响实验：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-report && QIHC_DISTRIBUTION_REPORT_INTERVAL=60 bash tools/analysis/distribution_feedback/04_report.sh
```

默认读取 `why/results/distribution_feedback_v1/`，写入独立目录 `why/results/distribution_feedback_v1_reports/`：

| 文件 | 内容 |
|---|---|
| `report.md` | 关键对照表、覆盖率、剩余时间和解释限制 |
| `paired_statistics.json` | 分数据集及汇总的所有配对指标、置信区间、检验、实例 ID |
| `eta.json` | CPU/GPU 已完成/剩余工作、校准样本、条件 ETA、待恢复状态 |
| `snapshot.json` | 分析时间、实验 manifest/source 哈希、分析器源码哈希 |

输出目录禁止放进实验目录，CLI 使用输出目录自己的锁避免两个报告进程交叉覆盖。报告采用原子写入；缓存已读检查点，减少每次刷新的解析开销。它仍会读共享文件系统并使用少量 CPU，不是零资源开销；请勿在 GPU 上分配新模型来运行报告。

自定义源/输出使用 `QIHC_DISTRIBUTION_RUN`、`QIHC_DISTRIBUTION_REPORT`。CPU 并发数和 GPU 副本数必须与实际作业一致，例如 `QIHC_DISTRIBUTION_CPU_WORKERS=8 QIHC_DISTRIBUTION_REPLICAS=2`；这里的 2 是两个模型副本，**不是两张卡**。默认校准近期 3600 秒，可用 `QIHC_DISTRIBUTION_ETA_WINDOW` 修改。

## 关键对照及配对方法

默认自动报告 11 对：

| 标识 | 实验组 − 对照组 | 解释 |
|---|---|---|
| `search_feedback` | best − self_review | 搜索证据相对纯文本自审 |
| `distribution_information` | distribution − best | 分布反馈增益 |
| `intervention_search` | probe_blind − distribution | 实际干预搜索的影响 |
| **`intervention_information`** | **intervention − probe_blind（⑤ − ④）** | 同样干预下，向 LLM 披露干预结果的增益 |
| `energy_selection` | energy − intervention | 能量表示选择 |
| `update_selection` | block − intervention | 更新方式选择 |
| `joint_vs_energy` | selected − energy | 联合选择相对只选能量 |
| `joint_vs_update` | selected − block | 联合选择相对只选更新 |
| **`selection_vs_random`** | **selected − random_policy（⑧ − ⑨）** | 同等 pilot 后按排名选策略相对随机选择 |
| `framework_on_metropolis` | metropolis_selected − metropolis_best | 框架是否也帮助 Metropolis |
| `pbit_vs_metropolis` | pbit_selected − metropolis_selected | 共同框架下更新核差异 |

仅报告当前 manifest 包含的组。分初始 `initial`、修订后 `revised` 两个阶段；初始阶段的策略对比共享同一初始模型，修订后衡量完整闭环。

1. 每一对只使用双方该阶段**所有预定种子都已保存**的实例，先平均种子再配对；不把多个种子当独立样本。无效模型/unsupported 是已完成失败，保留在分母中。
2. 显示配对/全部实例数、覆盖率、待完成数。无需等某实例的其他 10 组全部结束，就能报告⑤/④或⑧/⑨，但部分覆盖可能偏向较容易实例，不能视为最终结论。
3. 独立可行率和公开答案数值一致率分开统计；汇总为 `ALL_gold_verified` 与 `ALL_answer_only`，按实例等权，不混成“正确率”。
4. gap 仅比较**双方在同一个种子下都可行**且有独立参考值的结果，再平均到实例。报告共同可行种子数、可用实例数。不能拿两边不同的成功实例集合直接比较 gap。
5. 同时报各阶段记录的搜索秒数与输出 token 配对差。它们不包含未记录的解析/排队等开销，不应称为完整端到端能耗。
6. 所有 Δ 都是“实验 − 对照”：可行率越正越好；gap、时间、token 越负越好。Markdown 将率和 gap 换算为百分点 pp，JSON 保持原比例。

统计量包含配对均值差、实例 bootstrap 95% 区间（默认 2000 次）、胜/平/负、双侧配对 sign test，以及 Holm 校正 p 值。校正家族是当前快照输出的全部“数据集/汇总 × 对照 × 阶段 × 指标”检验，允许依赖但较保守。bootstrap 区间本身不是同时置信区间；sign test 检验胜负方向，不是均值相等。

少于两个配对实例不给置信区间和 p 值；无样本返回 null 而非 0。持续刷新属于探索性观察，Holm 不校正跨快照反复查看的行为；正式显著性结论应以预先定义的完整最终结果为准。

## ETA 的口径

### CPU

根据完整任务图计算尚缺的逐种子搜索：所有实例 × 所有组 × 初始/修订 × 各自种子数（CP-SAT 每阶段一次）。不只数当前队列。

从真实检查点 `seconds` 校准：优先近期同数据集/组/阶段，再退到同数据集/后端、同后端、全采样任务；不足时尝试历史样本。每个池至少 3 个有效时长。无采样器时长时显示待校准；CP-SAT 可退回配置的求解时间限额，并明确标注它不是实测。已知同一阶段模型无效的未完成种子不虚增求解工作，但解析/写入仍有未记录开销。

同一实例的种子任务不会并行调度，因此还计算最长剩余实例的串行工作量，不会把“只剩一个实例”错误地按 8 个进程加速。

### GPU

预期请求是每例 1 次共享初始建模 + 每组 1 次修订。计入尚未由 CPU 生成的未来请求，避免队列暂空时显示“快完成”。从旧作业复制的回答只算已完成，不用于推理速度校准。

按 `replica + batch_id` 去重：一个 batch 的 `batch_seconds` 只算一次，不乘回答条数。只用全部回答均已保存、元数据一致、成功生成的 batch；正在写入的半批不用于测速。批大小变化和拆批后的小 batch 按实际大小计入。失败重试开销单列。

初始建模与修订分别校准，不用短初始回答直接推断长修订回答；至少需要 3 个完整 batch 和对应类型的足够样本。按请求哈希分片估算每个副本未来工作量，取最长副本作为 GPU 连续工作时间估计。

另给“近期队列真实吞吐”估计：按近期完成 batch 的时间跨度计算，可能包含 CPU 等待、重新启动等实际停顿。样本太少或时间跨度小于 60 秒不显示该估计。

### 整体

输出 CPU/GPU 完全重叠与完全串行两种情景，**不是统计置信区间、保证的上下界或固定完成时刻**。早期简单实例、后续更大模型表示、长输出、平台排队/回收均会使估计变化。CPU/GPU 报告停止时显示 `resume_required`；不会假设它们会自动恢复。无心跳只能说状态过期，不能由此断言进程死亡。

状态值：`warming_up`（待校准）、`conditional_estimate`、`resume_required`、`awaiting_finalization`、`complete`。所有搜索/回答完成但最终文件未汇总时，不会误报完整实验结束。

## 验证

```bash
python -m pytest tests/test_distribution_reporting.py -q
```

覆盖关键对照、种子聚合、无效模型分母、共同可行种子、最大化目标的 gap 方向、bootstrap 复现、Holm、未来任务计数、batch 去重、复用回答排除、样本不足、停止/最终化状态、读缓存更新、输入完整性、实验目录逐字节不变，以及两套实验源码哈希保持不变。

2026-10-22 验证：新增报告测试 23 项；连同分布反馈、旧 CO、公共数据适配与通用能量相关回归，共 **205 passed**。真实 CPU 搜索检查点 → 独立 CLI 报告的集成测试通过，Shell 语法与静态检查通过。此验证不等同于远端 H100 的真实速度校准；ETA 必须读取服务器运行记录后才有意义。
