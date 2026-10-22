# 分布诊断、约束干预与通用能量策略实验 v1

这是独立的新实验，**不改动正在运行的 CO 实验、结果目录、脚本或源码哈希**。新增代码仅位于 `qihc/generic/distribution/` 和 `tools/benchmarks/distribution/`；旧实验对顶层 Python 文件的校验不包含它们。为进一步隔离，请使用单独 Git worktree，不在旧作业工作目录切换分支。

本轮是方法实现与无后训练对照实验，不是已取得质量提升的结果。模型权重、环境、现有审计后的数据均复用，无需下载或重跑原来的准备流程。

## 1. 实现内容与适用范围

| 模块 | 本轮实现 | 边界 |
|---|---|---|
| 分布诊断 | 固定窗口快照、逐约束违反率、共违反率、变量经验熵、目标分位数、状态变化率 | 相关的有限退火样本，不是独立平衡采样，也不是语义正确性证明 |
| 约束干预 | 选基准窗口违反率最高的最多 2 条约束；惩罚乘以 0.25/4；同起点、同种子、固定温度 0.5；比较违反率和目标变化 | 不将敏感性解释为约束错误、冗余或无解证明；无约束槽位仍消耗对照预算 |
| 能量表示 | 原始平方残差、按整数公因子约简后的平方残差、约简后的线性残差 | 不改原目标、变量域或约束；不执行模型生成的代码 |
| 更新核 | 单比特 Gibbs；由约束/目标因子支持集导出的双比特精确热浴更新；对称 Metropolis 提议 | 按通用因子结构，不按背包/图问题名称写特殊邻域；双比特策略保留单比特更新 |
| 自动选择 | 等额 pilot 比较可行样本比例、原始违反量和目标，选择表示/更新方式 | 是当前实例的预算内搜索选择，**不是已训练的 LLM 策略头**；不读 gold |
| LLM 反馈 | 相同初始模型；不同证据反馈后返回完整模型及原文 grounding | 原文引述只表示可追溯，不证明没有遗漏约束 |
| 调度 | 独立 CPU 多进程评分；独立四卡双副本批量推理；每阶段每种子原子保存 | 不伪造 GPU 负载；不能保证达到平台未知的回收阈值 |

统一接受现有 `bounded-integer-v2`：有限有界整数/二进制变量、线性约束、Boolean 逻辑、线性/二次目标。仍受现有 IR 的变量数、系数和表达式大小限制；不声称支持任意连续、无界或黑盒优化。

### 能量正确性说明

对线性约束，用 `gcd(|a_1|,...,|a_n|,|rhs|)`（至少为 1）约简残差。因原变量与系数均为整数，非零约简残差至少为 1；逻辑残差是 0/1。目标除以其全变量域保守绝对上界，因此归一化目标位于 [-1,1]。

常规窗口每条惩罚权重为 3。若原问题存在可行解，则三种表示的**全局最小点**均可行，并在可行域上保持原目标排序，因为一次违反的惩罚超过目标的最大可能差 2。这不代表有限时间采样必能找到全局最小点。

干预窗口临时降低权重时不使用上述保证，所有返回候选仍按原始约束检查。有限整数的比特编码可能存在重复表示，因此位空间 Gibbs 分布也不能直接称为原整数状态上的均匀先验 Boltzmann 分布。

## 2. 对照组（默认全部 12 组）

| 组名 | 返回 LLM 的证据 | 实际干预 | 策略 |
|---|---|---:|---|
| `pbit_self_review` | 不提供搜索证据，自审文本 | 否 | 固定 |
| `pbit_best` | 最佳访问解 | 否 | 固定 |
| `pbit_distribution` | 最佳解 + 分布诊断 | 否 | 固定 |
| `pbit_probe_blind` | 最佳解 + 分布诊断，屏蔽干预结果 | 是 | 固定 |
| `pbit_intervention` | 最佳解 + 分布 + 干预 | 是 | 固定 |
| `pbit_energy` | 同上 | 是 | 只选择能量表示 |
| `pbit_block` | 同上 | 是 | 只选择更新方式 |
| `pbit_selected` | 同上 | 是 | 联合选择 |
| `pbit_random_policy` | 同上 | 是 | 相同策略 pilot 后随机选择，不利用排名 |
| `metropolis_best` | 最佳访问解 | 否 | 固定 |
| `metropolis_selected` | 最佳解 + 分布 + 干预 | 是 | 联合选择，Metropolis 核 |
| `cp_sat_feedback` | CP-SAT 求解状态与候选 | 不适用 | 独立计时参考 |

建议首先阅读以下配对比较，而不是只看“完整版 vs 最弱组”：

- `distribution` vs `best`：相同搜索过程，只改变可见的分布信息。
- `intervention` vs `probe_blind`：相同实际干预和候选，只改变干预信息是否可见。
- `probe_blind` vs `distribution`：搜索干预本身的影响，不归因为 LLM 理解提升。
- `selected` vs `random_policy`：相同 pilot 组合与开销，检查学习式排名选择是否优于随机选策略。
- `energy` / `block` / `selected`：分别检验表示、更新、二者组合。
- `metropolis_selected` vs `pbit_selected`：共同框架下的采样核差异。

软件 p-bit 实现就是这里的 Gibbs 热浴更新；`method='gibbs'` 是严格同核别名，已有逐项等价测试。**不能把两个名字当成不同算法并宣称 p-bit 胜过同核 Gibbs。**本轮也未实现并行回火或 CP-SAT 多样化解池，因此不足以证明优势是 p-bit 独有；这是后续因果归因实验的边界。此前延期的论文原版复现不在本轮中冒充已运行。

## 3. 预算与公平性

默认复用原 CO 队列的全部实例：如果源队列是此前的 300 例，即 NLCO 240 + 合成 60，本轮仍是同样 300 例；实际数量以 manifest 为准。不按求解成功与否筛样本。

| 项目 | 默认 |
|---|---:|
| 每组 LLM 调用 | 2 次：共享初始建模 + 本组一次修订 |
| 每个搜索阶段的种子数 | 3（0、1、2） |
| 每种子的链数 | 16 |
| 每链候选评分预算 | 8192 |
| 每种子预算分配 | 基准约 1/4；4 个成对探测槽约 1/4；6 个策略 pilot 约 1/4；剩余用于最终搜索 |
| 每实例每采样组两阶段总评分上限 | 8192 × 16 × 3 × 2 = 786432 |
| CP-SAT | 每阶段 1 次、单线程、5 秒求解限额；另报实际调用时间 |
| CPU 并行 | 8 个实例级进程，每次任务至多计算一个搜索种子 |
| GPU 并行 | 2 个模型副本，各分配 2 卡；每副本 batch=4 |

固定组用相同槽位做 sham 干预与固定策略 pilot；所有组都允许访问过的候选进入返回解档案。计费包含全部初始化、探测、pilot 和最终搜索；当前状态的能量缓存复用。单比特 Gibbs/Metropolis 每次新增评分 1 个状态，双比特 Gibbs 评分 3 个备选，Metropolis 只评分实际提出的 1 个状态，**不浪费对照计算来凑计数**。固定变量模型可提前结束，未用额度显式记录。

CP-SAT 的时间限额与采样评分预算不是等价预算；它是单独报告的强求解器参考，不能据此下“等算力击败 CP-SAT”的结论。报告所有采样组的实际时间、候选评分数、LLM token 数及摊分推理耗时。相同调用次数不代表相同 token/时间，最终应结合质量—成本关系判断。

LLM 反馈固定取 seed=0，不选择最好种子；独立评测按实例先平均种子。初始回答可从已知源码版本、相同模型身份与相同提示词的旧队列**只读复制**，不复用旧搜索结果或修订回答；旧源码身份不匹配时只复制数据并重新排队初始建模。共享初始推理在每组成本账上归属一次，物理执行不重复。搜索检查点绑定实际模型回答的 SHA256，防止回答改变后混用旧搜索。

自动汇总包含修订前/后的独立可行率、有可行解实例的 signed gap、各阶段成本、相对 `pbit_best` 的实例配对差与探索性 bootstrap 区间。无效模型和 unsupported 不从已完成实例的分母中删除。只有最终全部组完成的实例进入汇总，并明确显示全数据集覆盖率，不能把部分汇总当完整结果。

原数学模型下的可行率、外部 gold 下的可行率、公开答案数值一致率是不同指标。最后一种不证明解的可行性；输出解在 gold 上可行也不等于整个生成模型语义等价。对于没有证明最优的公开参考，只称“相对参考值 gap”。

## 4. 不打扰现有作业的部署与运行

CPU 端只做一次：获取新代码并建立独立 worktree。**不要在旧作业目录切换分支或停止旧作业。**

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && git fetch origin && git worktree add --detach ../qihc-distribution origin/llm-cvrp
```

联网 CPU 计算节点一行启动（本脚本不再下载任何模型或数据；需旧队列已有 manifest）：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-distribution && bash tools/benchmarks/distribution/01_cpu.sh
```

CPU 显示 `New CPU queue ready` 后，在**另一份 GPU 资源分配**中，配置 1 worker / 4 H100，一行启动；不用 torchrun：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-distribution && bash tools/benchmarks/distribution/02_gpu.sh
```

如果只有正在运行旧实验的四张卡，等待旧实验释放后再启动新 GPU 作业，不要两个作业同时占同一批卡。新 CPU 评分也消耗计算资源；要完全避免争用，请另申请 CPU 计算节点，不在登录节点跑搜索。

两个脚本均直接使用 `/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128/bin/python`，不要求手工激活 conda。新结果默认写入 `results/distribution_feedback_v1`；旧目录 `results/co_comparison_v1` 不变。

两条原命令均可续跑。保持 worktree 的版本不变；源码、预算、数据或模型身份改变会拒绝混跑。若另做一组预算实验，设置新的 `QIHC_DISTRIBUTION_RUN`，CPU/GPU 使用相同值。

推理批大小与显卡分片可能影响浮点数值结果；每条回答记录 batch、replica 和 token 开销，发布结果时应披露混合执行配置，不能宣称跨批大小逐字复现。

常用设置：`QIHC_BASE`、`QIHC_ENV_PREFIX`、`QIHC_DISTRIBUTION_SOURCE`（旧队列）、`QIHC_DISTRIBUTION_RUN`（新队列）、`QIHC_DISTRIBUTION_CPU_WORKERS`、`QIHC_DISTRIBUTION_BATCH`、`QIHC_DISTRIBUTION_REPLICAS`。预算覆盖为 `QIHC_DISTRIBUTION_BUDGET`、`QIHC_DISTRIBUTION_CHAINS`、`QIHC_DISTRIBUTION_SEEDS`、`QIHC_DISTRIBUTION_SOLVER_SECONDS`，均须新建结果目录。

### 进度、结果、报错

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-distribution && bash tools/benchmarks/distribution/03_status.sh
```

- 汇总：`results/distribution_feedback_v1/summary.md` 与 `summary.json`。
- 完整记录：`final/`；逐种子检查点：`progress/`；请求/响应：`requests/`、`responses/`。
- CPU 错误证据：`errors/`；历史错误不会因后续成功而自动删除。
- 启动日志：`logs/distribution-cpu-*.log`、`logs/distribution-gpu-*.log`。
- OOM 会按现有批推理逻辑拆分 batch；单请求仍 OOM 会报错并保留待处理请求，不伪装为模型质量失败。
- GPU 无待推理任务约 120 秒会主动保存状态并退出；CPU 继续搜索产生新请求后，可重提同一 GPU 命令。退出不等于完整实验结束，以 summary 的 `status=complete` 为准。

不能保证平台不回收资源。采用真实批量推理、CPU 并行、检查点续跑及空闲释放，不做无意义矩阵乘法、显存占位或虚报利用率。可能仍需多次 GPU 分配，尤其 CPU 搜索长于推理时。

## 5. 验证与尚未验证的部分

启动 CPU 脚本会运行一个临时目录下的完整工程 smoke：真实 Gibbs/Metropolis/CP-SAT，**模拟 LLM 回答**，检查全部 12 组、汇总与续跑。它不是科学评测结果，不验证 32B 模型生成质量或四卡显存。

本地还应执行：

```bash
python -m pytest tests/test_distribution_feedback.py tests/test_comparison.py tests/test_parallel_inference.py tests/test_bounded_energy.py tests/test_nlco.py tests/test_public_simulation.py tests/test_public_benchmarks.py -q
python -m pytest tests/test_generic_energy.py tests/test_generic_reuse.py -q
python tools/benchmarks/distribution/smoke.py --subprocess-cpu
```

覆盖能量最小点保持原目标与可行性、整数/逻辑约束、精确预算、同核 Gibbs 等价、低温双比特移动、sham 零差异、主动干预盲对照、随机策略对照、无 gold 泄露、错误模型独立验错、只读复制、断点续跑、分片队列和旧源码哈希不变。

2026-10-22 本地验证：上述两组分别 **108 passed** 与 **74 passed**，合计 **182 passed**；12 组真实 CPU 求解/模拟 LLM 的端到端 smoke（含子进程池与恢复）通过；三个 Shell 入口语法检查通过。测试使用已有隔离的 OR-Tools 依赖，不改动用户主环境。

尚未在此开发机执行远端四卡 H100 / 全量真实模型实验，也未证明新版提高质量、语义泛化或能耗。后训练数据导出、SFT/DPO、并行回火、CP-SAT 多样化解池和真正 p-bit 硬件测量不在本轮实现范围。
