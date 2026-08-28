# QIHC 下一阶段实验方案：LLM 引导的 p-bit 大邻域搜索

## 1. 结论先行

主实验建议选择 **自然语言约束的容量车辆路径问题（NL-CVRP）**，并将方法定义为：

> LLM 解析自然语言约束、选择有希望的候选邻域；p-bit 在受限 QUBO 上完成组合重排；确定性验证器保证硬约束；求解反馈再用于候选更新和 LLM 后训练。

这比继续以 BBH 子集选择作为主实验更合适。BBH 可以保留为“组合推理”辅证，但不应承担离散优化与硬件加速的核心结论。

建议方法名暂用 **QIHC-LNS**，避免在尚未证明目标函数对应严格变分界或稳态分布之前，把整个闭环直接称为“统一变分自由能”。

### 1.1 一页式方案总结

QIHC-LNS 的核心不是让 LLM 直接求出最终解，而是建立一个有明确信任边界的异构闭环：

```text
自然语言需求 + 数值实例
        ↓
LLM 解析约束、选择候选邻域、输出置信度
        ↓
统一 Constraint IR + 格式/语义检查
        ↓
模板编译为局部稀疏 QUBO
        ↓
p-bit 生成多个低能量组合样本
        ↓
确定性验证器检查硬约束和真实目标
        ↓
接受更优可行解，否则保留当前最好解
        ↓
将违约、改进量和采样统计反馈给 LLM
        ↓
事件触发迭代，并用于 SFT/DPO/GRPO 后训练
```

| 层次 | 核心职责 | 输出 | 可信性要求 |
|---|---|---|---|
| LLM 语义层 | 理解不同表述、区分硬约束与软偏好、选择候选邻域 | Constraint IR、候选集合、置信度 | 可以有误，不直接决定最终可行性 |
| 约束编译层 | 将通用约束原语编译成局部 QUBO | 稀疏 `h, J` 和解码信息 | 模板化、可测试、可与枚举对齐 |
| p-bit 搜索层 | 在受限二值空间内概率采样和组合修复 | 多个低能量样本 | 比较固定预算下的 gap、TTT 和样本多样性 |
| 确定性验证层 | 检查全部硬约束并计算真实任务成本 | 可行性、违约列表、真实目标值 | 最终可信边界，不接受未验证解 |
| 反馈与训练层 | 决定何时刷新，并用求解反馈改进候选策略 | 下一轮状态或训练偏好对 | 无改进时回退，保证 do-no-harm |

主任务采用 NL-CVRP；Max-Cut 用于 p-bit 后端标度，BBH 用于组合推理辅证，TSP 用于最小 QUBO 与硬件映射验证。后续若扩展到排程、选址或资源分配，只替换问题变量、约束模板和验证器，保留同一闭环接口。

## 2. 为什么选择 NL-CVRP

### 2.1 任务同时需要语义和组合优化

- LLM 擅长：从自然语言中识别容量、优先级、同车、先取后送、时间窗等约束；判断哪些客户或边值得重新组合；生成多样候选。
- p-bit 擅长：在二值变量构成的受限邻域内并行采样，搜索低能量组合。
- 验证器擅长：确定性检查每个客户恰好访问一次、容量、时窗和顺序等硬约束。

三者的职责互补，任何一个单独组件都不能替代完整系统。

### 2.2 有标准数据、强基线和硬件映射依据

- CVRPLIB 提供公开实例与已知最好解，可计算最优差距。
- RoutBench 提供 1,000 个由 24 类属性构成的 VRP 变体，实例包含自然语言描述、数值数据和验证代码，适合检验语义泛化。
- HGS-CVRP 是必须加入的强经典基线。
- 近期 Ising/p-bit 硬件已在 TSP 上验证，近期 QUBO 分解工作也已在 CVRP 上验证，因此路由问题与硬件路线相容。

### 2.3 不选其他问题作为主实验的原因

| 问题 | 优点 | 主要缺点 | 定位 |
|---|---|---|---|
| Max-Cut | QUBO 最自然、现有代码可直接运行 | 几乎没有 LLM 必要性，也缺少硬约束 | 后端标度与硬件基准 |
| BBH 子集选择 | 与现有 VCI 代码衔接快 | 不是典型工业离散优化，正确性与 Ising 能量容易失配 | 辅助组合推理实验 |
| TSP | 文献和硬件基准多 | 语义层作用弱，全量映射需约 `N^2` 比特 | CVRP 前置 smoke test |
| 作业车间调度 | 工业意义强、约束丰富 | QUBO 编码和可靠解码更复杂，第一阶段风险较高 | 后续扩展任务 |
| NL-CVRP | 语义、硬约束、标准评价和硬件映射兼具 | 必须采用分解，不能直接全量 QUBO | **主实验** |

## 3. 方法：不要让 LLM 直接输出最终解

### 3.1 状态与接口

第 `t` 轮维护一个当前可行解 `S_t`。LLM 接收：

- 问题的自然语言约束；
- 当前路线摘要，而不是完整冗长轨迹；
- 当前违约类型、瓶颈车辆、边代价和历史改进；
- p-bit 上一轮的样本能量、可行比例和样本多样性。

LLM 输出结构化对象：

```json
{
  "destroy_customers": [3, 8, 11, 19],
  "candidate_insertions": {
    "3": [[0, 4], [1, 7]],
    "8": [[0, 5], [2, 3]]
  },
  "constraint_weights": {
    "priority": 0.8,
    "same_vehicle": 1.0
  },
  "confidence": 0.74
}
```

LLM 的研究对象是 **候选邻域策略**，不是最终路线字符串。这样输出容易校验、可做监督学习，也能与传统候选选择方法公平比较。

### 3.2 跨问题约束理解：Constraint IR

LLM 可以作为不同离散优化问题的统一语义入口，但“理解约束”不等于“保证约束正确”。系统必须将自然语言先转换为统一的约束中间表示 `Constraint IR`，再由程序编译和验证。

例如输入：

> 每辆车载重不超过 100；客户 A 和 B 必须由同一辆车服务；客户 C 必须在 D 之前访问；VIP 客户尽量在上午送达。

LLM 应输出类似：

```json
{
  "hard_constraints": [
    {"type": "capacity", "limit": 100, "scope": "per_vehicle"},
    {"type": "same_resource", "entities": ["A", "B"]},
    {"type": "precedence", "before": "C", "after": "D"}
  ],
  "soft_constraints": [
    {
      "type": "preferred_time",
      "entity": "VIP",
      "interval": ["08:00", "12:00"],
      "weight": 0.7
    }
  ]
}
```

通用约束原语至少包括：

| 约束原语 | 路由示例 | 排程/分配示例 | 后端处理 |
|---|---|---|---|
| `exactly_k` / `exactly_one` | 客户恰好访问一次 | 任务恰好分配给一台机器 | one-hot/cardinality QUBO 模板 |
| `capacity` | 车辆载重上限 | 机器或资源容量 | QUBO 惩罚、辅助变量或解码修复 |
| `mutual_exclusion` | 两客户不能同车 | 两任务不能同时执行 | 二次耦合模板 |
| `same_resource` | 两客户必须同车 | 两任务必须同机 | 等价/一致性模板 |
| `precedence` | 先访问 C 再访问 D | 工序 A 先于 B | 顺序变量或验证器 |
| `time_window` | 在时间窗内服务 | 在期限前完成 | 离散时间模板或验证器 |
| `coverage` | 每个需求点被服务 | 每项需求至少被一个设施覆盖 | 覆盖模板 |
| `connectivity` | 路线不能产生子环 | 所选网络必须连通 | 分解、割平面或验证反馈 |
| `implication` | 选择 A 则必须选择 B | 启用设备 A 则启用配套资源 | 逻辑 QUBO 模板 |
| `soft_preference` | VIP 尽量早送达 | 高优先级任务尽量先执行 | 线性场或二次语义代价 |

约束处理分三级：

1. **已有原语**：使用经过单元测试的固定模板直接编译。
2. **原语组合**：LLM 生成组合关系，程序执行类型检查、变量检查和冲突检查后编译。
3. **未知复杂约束**：先生成确定性 `constraint_checker`，仅用于筛选样本和反馈，不直接让 LLM 自由生成 QUBO；验证充分后再沉淀为新模板。

必须设置三层检查：

- **格式检查**：JSON Schema、变量是否存在、类型和数值范围是否合法；
- **语义检查**：硬/软约束分类、方向和参数是否与原文一致，可通过反向自然语言解释或双模型一致性检查实现；
- **确定性验证**：最终解逐条执行约束检查，LLM 和 p-bit 均无权绕过。

跨问题能力应通过“约束解析准确率、硬软分类准确率、参数抽取准确率、结构化输出合法率、未见约束组合可行率”进行评价，而不能只凭案例展示宣称理解任意问题。

### 3.3 受限 QUBO 与 p-bit 修复

从当前解中移除 `r` 个客户，只对候选插入位置建立变量：

\[
x_{i,p}=1 \quad \Longleftrightarrow \quad \text{客户 } i \text{ 被插入候选位置 } p.
\]

优化目标建议写成可审计的加权和：

\[
E(x)=\Delta d(x)
+\lambda_{\text{assign}} C_{\text{assign}}(x)
+\lambda_{\text{cap}} C_{\text{cap}}(x)
+\lambda_{\text{sem}} C_{\text{sem}}(x;q_t).
\]

其中：

- `Δd`：相对当前路线的距离变化；
- `C_assign`：每个被移除客户恰好插入一次；
- `C_cap`：车辆容量等可二次化的约束惩罚；
- `C_sem`：由 LLM 解析得到的语义偏好或约束代价。

不应只依靠一个极大惩罚系数保证所有硬约束。p-bit 采样后必须经过确定性可行解码或 repair；若无可行样本，则回退到 `S_t`，从而建立 **do-no-harm** 保证。

### 3.4 控制 p-bit 规模

第一阶段每个子问题控制在 64–256 个二值变量：

- `r = 4, 8, 12, 16` 个待重排客户；
- 每个客户保留 `k = 4, 8, 16` 个候选插入位置；
- 比较候选覆盖率与 p-bit 规模的 Pareto 前沿。

不要直接使用位置编码把整个 `N` 城市问题映射成 `N^2` 个变量。主张的可扩展性应来自“LLM/规则缩小候选空间 + p-bit 解子问题”，而不是全量 QUBO。

## 4. 闭环与后训练

### 4.1 推理时闭环

1. 解析：LLM 将自然语言描述转成规范约束和候选邻域。
2. 编码：GPU 构建稀疏受限 QUBO，并校准惩罚系数。
3. 搜索：p-bit 运行 Gibbs、PT 或后续 FPGA 后端，返回多个低能量样本。
4. 验证：确定性验证器计算可行性、距离和语义代价。
5. 更新：只在以下条件满足时刷新 LLM 候选：连续 `m` 轮无改进、候选覆盖率代理下降、或反馈置信度超过阈值。
6. 回退：本轮没有更好的可行解时保留 `S_t`。

“刷新律”和“信任律”应落到可计算门控上，而不是只作为概念表述。

### 4.2 后训练路线

训练目标仍然是候选选择策略：

- **SFT**：用 HGS/OR-Tools 轨迹生成 `(状态 → 高收益 destroy set / candidate edges)` 数据。
- **偏好优化**：同一状态下，将 p-bit 改进更大、可行率更高、计算量更低的候选标为 preferred。
- **GRPO/DPO 或轻量策略梯度**：

\[
R = \alpha\,\frac{d(S_t)-d(S_{t+1})}{d(S_t)}
+\beta\,\mathbb{1}[S_{t+1}\text{ feasible}]
-\gamma\,N_{\text{viol}}
-\delta\,\text{pbit\_steps}
-\xi\,\text{LLM tokens}.
\]

建议先做 LoRA，不要一开始全参数训练。先证明“后训练提高候选覆盖率，继而提高固定预算下的最终最优差距”，再讨论更大的模型。

## 5. 实验设计

### 5.1 两条数据轨道

**Track A：标准 CVRP，隔离优化能力**

- CVRPLIB 的 A/B/E/P/X 小中型实例；
- 先用 20–100 客户，后扩展到 200+；
- 不加入自然语言软目标；比较不同候选选择器和 p-bit 后端。

**Track B：NL-CVRP，验证语义泛化**

- RoutBench 的 25/50/100 节点实例；
- 先选容量、优先级、同车三类约束，再扩展时间窗与取送；
- 按约束组合切分训练/测试，必须包含 unseen constraint composition。

Track A 回答“p-bit 是否真的改善搜索”，Track B 回答“LLM 是否真的提供语义泛化”。两条结论不能混在一张总准确率表中。

### 5.2 必须比较的基线

在相同 wall-clock 或相同目标评估次数下比较：

1. HGS-CVRP；
2. OR-Tools / Gurobi（小规模最优值或下界）；
3. 经典 ALNS；
4. LLM-only best-of-N；
5. p-bit-only，全候选或规则候选；
6. random-neighborhood + p-bit；
7. kNN/handcrafted-neighborhood + p-bit；
8. LLM-neighborhood + SA/PT；
9. **LLM-neighborhood + p-bit（QIHC-LNS）**；
10. post-trained LLM-neighborhood + p-bit。

必须加入第 7 和第 8 项，否则无法区分收益来自 LLM、候选空间缩小，还是 PT/退火本身。

### 5.3 主指标

- **Feasible rate**：有效路线比例；
- **Optimality gap**：`(cost - BKS) / BKS`；
- **Time-to-target (TTT/TTS)**：达到指定 gap 的时间或采样次数；
- **Candidate recall**：BKS/高质量解中的关键边有多少进入候选集；
- **Improvement per p-bit update**：单位随机更新带来的成本下降；
- **Generalization**：未见规模、未见约束组合上的性能；
- **System metrics**：端到端延迟、LLM token、GPU 时间、p-bit 更新数；硬件阶段再加能耗。

不要把“准确率”作为 CVRP 主指标，也不要用 Ising 能量代替真实路线代价。

### 5.4 消融

- 去掉 LLM，只用随机/kNN 候选；
- 去掉 p-bit，改用 greedy/SA/PT repair；
- 单候选 vs 多样候选；
- 无反馈、每轮反馈、事件触发反馈；
- 无信任门、固定阈值门、自适应置信门；
- 无后训练、SFT、SFT + preference/RL；
- p-bit 变量数、量化位宽、同步/异步更新、温度梯度；
- 固定大惩罚与自适应惩罚系数。

每个配置至少使用 20 个随机种子或足够多的独立实例，报告均值、置信区间和配对显著性检验。所有方法固定时间预算，并单独报告 LLM 首次调用开销与摊销后开销。

## 6. 三个可证伪假设

- **H1 候选假设**：在相同候选数量下，LLM 的关键边/高收益邻域 recall 高于 kNN、随机和手工启发式。
- **H2 后端假设**：在相同候选集与时间预算下，p-bit 后端的最终 gap 或 TTT 优于 SA/Gibbs-only，并在噪声或多峰实例上优势更明显。
- **H3 闭环假设**：p-bit 反馈训练提高候选 recall，且该提升能传导为 unseen constraint composition 上的最终 gap 改善。

若 H1 不成立，LLM 不应被宣称为必要组件；若 H2 不成立，研究应转向 p-bit 采样/硬件机制；若只有可行率上升而 gap 变差，应重新校准能量与真实目标，而不是只汇报可行率。

## 7. 12 周最小可行计划

| 周期 | 交付物 | 通过标准 |
|---|---|---|
| 第 1–2 周 | CVRP 数据读取、验证器、HGS/OR-Tools 基线 | 可复现 BKS/gap，单元测试覆盖约束 |
| 第 3–4 周 | LNS 子问题与 QUBO 编码 | 小规模可与暴力枚举逐例对齐 |
| 第 5–6 周 | p-bit repair 接入现有 backend | 固定候选下与 SA/PT 公平对比 |
| 第 7–8 周 | LLM 结构化候选生成 | JSON 解析成功率 >99%，候选 recall 曲线完整 |
| 第 9–10 周 | 完整闭环和消融 | H1/H2 可被统计检验，do-no-harm 生效 |
| 第 11–12 周 | LoRA SFT + preference pilot | 在未见实例上候选 recall 和 gap 同时改善 |

硬件迁移应放在算法接口和标度结论稳定之后。第一阶段只需记录稀疏耦合数量、位宽、并行更新冲突和所需 p-bit 数，为 FPGA/CMOS 映射提供约束。

## 8. 对现有仓库与答辩方案的直接修改建议

### 8.1 现有代码

保留：

- `qihc.ising` 的 Gibbs/PT/SQA/SA 后端；
- `PBitBackend` 抽象；
- 反馈门控、刷新和信任机制的实验脚手架；
- BBH 与 Max-Cut 作为辅证和回归测试。

新增：

- `qihc/problems/cvrp/instance.py`：实例与自然语言约束；
- `qihc/problems/cvrp/verifier.py`：确定性可行性和真实成本；
- `qihc/problems/cvrp/neighborhood.py`：destroy/repair 候选；
- `qihc/problems/cvrp/qubo.py`：受限子问题编码与小规模精确校验；
- `qihc/orchestrator/cvrp_scheduler.py`：do-no-harm 闭环；
- `experiments/run_cvrp_lns.py` 与统一预算基准脚本。

### 8.2 答辩叙事

把当前“统一自由能提升质量与效率”的结论式表述改为可验证问题：

> 研究语义候选压缩、概率组合修复与反馈学习之间的协同条件，揭示候选覆盖率、p-bit 规模、可行率、最优差距和能效之间的关系。

结果页必须补充：

- 数据集、实例数、随机种子和置信区间；
- 与 HGS/OR-Tools/ALNS 的真实 gap；
- 同预算消融；
- 当前负结果：低 Ising 能量不必然对应更高任务正确率；
- 硬件结论与软件仿真结论分开陈述。

## 9. 参考资源

- ARS / RoutBench: https://arxiv.org/abs/2502.15359
- RoutBench code and data: https://github.com/Ahalikai/ARS-Routbench
- CVRPLIB: https://galgos.inf.puc-rio.br/cvrplib/
- HGS-CVRP: https://github.com/vidalt/HGS-CVRP
- Combinatorial Reasoning: https://arxiv.org/abs/2407.00071
- End-to-end LLM CO solver and feasibility-aware post-training: https://arxiv.org/abs/2509.16865
- G-LNS: https://arxiv.org/abs/2602.08253
- QUBO search-space decomposition for CVRP: https://doi.org/10.1038/s41598-026-57443-z
- p-bit/Ising TSP hardware example: https://doi.org/10.1038/s44335-026-00060-w
- Parallel p-bit update and hardware-cost analysis: https://doi.org/10.1038/s41598-026-47285-0
