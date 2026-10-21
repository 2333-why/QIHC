# 统一语义建模与因子能量仿真：公开测试接入

## 范围与状态

本轮不按背包、分配、路径等类别分派算法；代码中没有公开题号或问题类别的搜索分支。
统一支持 `bounded-integer-v2`：有界整数/二元变量、整数系数线性约束、二次目标、二元逻辑。
最多256个原始变量、512条源约束，边界/系数绝对值≤10000；这些是当前软件输入安全限制，不是 FPGA 限制。
连续变量、未能证明的有限界、一般非线性、题面缺数据必须明确拒绝，不能改题补数。

已完成代码、测试、实际公开题目的准备阶段。**尚未运行本机不具备的32B GPU推理，未获得公开数据集端到端成绩。**
默认从六个已校验数据集各按索引等距取40题，共240题；选择不看答案、建模成功率或搜索结果。
`QIHC_PUBLIC_PER_DATASET=0` 可准备全部1961条；全量题目含重复，不代表1961个独立问题。
这批数据只作外部测试，不进入SFT/DPO；旧训练adapter可以单独评测，但旧格式adapter不保证适配新版输出。

## 核心改变

1. LLM输出统一模型和题面依据：每个变量、目标、约束都有原文引用及推导解释。
   验证器检查引用确实存在、字段完整；**不能证明模型语义正确，更不能发现所有遗漏**。
2. 新仿真后端直接计算原始约束能量：
   `E = signed_objective / objective_bound + sum(lambda_j * violation_j^2)`。
   等式用绝对残差，不等式用正部残差，逻辑用真假违反量；不生成松弛比特、不生成稠密QUBO矩阵。
3. 有界整数仍编码为比特；p-bit按完整能量差执行单点热浴更新，不再划分128位硬件块。
   该能量可以是高阶函数，这是仿真用的软件因子能量后端，不宣称可直接映射二体耦合硬件。
   有界编码可能让同一整数有多个表示，因此比特态的Gibbs分布不等于整数解的均匀基准分布。
4. 内层反馈按所有原始约束的违反率调惩罚；归档按原始可行性、原始目标选解。
   外层反馈将固定第一个种子的搜索诊断送回LLM复核，禁止删除约束来制造可行性。
   另保留不看搜索结果的LLM自我复核分支，二者都多调用一次LLM。

浮点仿真与有限步随机搜索不提供全局最优保证。直接算原始能量减少辅助比特，但当前实现并未缓存所有局部因子，
不能预先断言运行更快；每次更新仍需评估两个候选状态，需统计实际耗时与能量评估次数。
此前QUBO/分块后端保持不变，可另做编码开销对比，当前公开流程默认使用因子能量路径。

## 实验对照与指标

同一个LLM模型输出，共享预算和种子，运行：p-bit+反馈、p-bit无反馈、Metropolis+反馈、随机扫描。
默认每组3个种子、16条链、每链4096次单比特更新；全部组都记能量评估次数和耗时。
额外CP-SAT最多5秒、单CPU线程，只求解相同的预测模型，用来区分搜索质量与建模偏差；不是语义oracle。
CP-SAT答案不返回LLM，只有p-bit诊断进入搜索反馈分支。

| 指标 | 含义 |
|---|---|
| searched/invalid/unsupported | 全部入选题目的覆盖情况，失败题不丢弃 |
| predicted_model_feasible_rate_all_cases | 在LLM预测的约束下的可行率，不是原题正确率 |
| reference_agreement_rate_all_cases | 搜索结果满足预测模型且目标值接近发布答案；非数值答案暂不评分并单独计数 |
| cp_sat_optimal_predicted_models | CP-SAT证明预测模型最优的数量，不是证明原题建模正确 |
| generation tokens/seconds、search seconds | 推理和搜索成本；多轮比较应累计前一轮全部成本 |

目标匹配容差固定为绝对/相对1e-4；这只是项目统一试测约定，不冒充各原论文官方判分脚本。
整体目标匹配率以全部题目为分母，非数值/不支持/无效题均不会贡献成功，单独保留状态。
最终论文需独立参考模型或经审核的解验证器、题型覆盖/重复分组和数据集分项汇总。
不能用被测LLM自己生成的约束证明它理解正确，也不能以“回答对了目标值”替代完整建模正确性。

## 运行安排（CPU/GPU分离）

使用新版独立代码目录，保留旧实验checkout。新增代码会改变旧源指纹，不能绕过旧校验续跑。
已下载模型和环境直接复用，不重新运行第一步。以下均从仓库根目录执行。

```bash
# 联网CPU：首次下载公开数据（已有校验文件会跳过）
bash tools/benchmarks/prepare_public_cpu.sh
# CPU：准备240题；新输出目录，重复prepare不会覆盖
bash tools/benchmarks/run_public_simulation.sh prepare
# GPU任务：只推理，单worker启动一次，多卡以device_map分布模型，不能套torchrun
bash tools/benchmarks/run_public_simulation.sh infer base
# CPU计算节点：搜索、参考求解、生成两个后续复核请求
bash tools/benchmarks/run_public_simulation.sh score base
# 两个GPU任务可分别执行，保持同一个原始模型/adapter
bash tools/benchmarks/run_public_simulation.sh infer feedback
bash tools/benchmarks/run_public_simulation.sh infer self_review
# CPU计算节点评估复核后的模型
bash tools/benchmarks/run_public_simulation.sh score feedback
bash tools/benchmarks/run_public_simulation.sh score self_review
```

默认根目录 `/inspire/hdd/global_user/czxs25210022/why`，复用 `envs/qihc-cu128` 和 `models/Qwen3-32B`。
默认结果 `results/public_simulation_v1/`；使用 `QIHC_PUBLIC_RUN` 指定独立实验目录。
`QIHC_PUBLIC_MODEL`、`QIHC_PUBLIC_ADAPTER` 可选择基座或后训练模型（每个模型必须用不同结果目录）。
wrapper直接调用环境Python，不依赖交互终端激活。GPU不执行长CPU搜索，避免此前低GPU利用率回收问题；仍需遵守平台策略。

结果文件：`base.scores.summary.json`、`feedback.scores.summary.json`、`self_review.scores.summary.json`。
对应`.scores.jsonl`保留逐题状态、解、约束轨迹、每个种子的结果。目录中的`data/selection.json`固定测试选择。
`answers.DO_NOT_PROMPT.jsonl`仅评分进程读取，推理请求和外层反馈均不含答案字段。
每题原子保存，推理/评分同命令可续跑；参数、输入或源码变动拒绝混用结果。
模型配置内容及权重文件元数据纳入续跑检查，权重内容SHA256仍由原下载准备清单负责。

不允许将这240题的复核结果直接当成后训练数据；若需要训练，应另建来源独立的训练集，做跨来源、近重复及模板泄漏检查。
