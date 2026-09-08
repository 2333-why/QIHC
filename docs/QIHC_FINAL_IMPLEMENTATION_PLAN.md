# QIHC 当前确认实施方案

状态：已确认，作为当前代码实现、正式实验和后续汇报的唯一基线。
日期：2026-09-08

## 1. 核心目标

QIHC 联合大语言模型和 p-bit 概率网络求解带自然语言约束的组合优化问题。LLM 负责约束理解、候选解与候选搜索域；确定性编译器负责能量构造；p-bit 是核心组合搜索器；p-bit 的精英解、负样本和动作后验在线校准 LLM 候选 logits，并形成后训练数据。

主闭环：

`LLM(约束+候选) -> 多表示能量编译 -> p-bit多链采样 -> 严格验证 -> 精英/反例反馈 -> LLM`。

LLM 不直接生成 QUBO/Ising 系数，最终解也不由 LLM 直接决定。

## 2. 双线输入输出

### 2.1 约束线

自然语言与实例首先生成 CPP 约束 IR，每个原子包含类型、作用域、参数、硬/软属性、权重、来源文字和置信度。CPP 依次通过 schema、实体/类型、测试、精确反例和可编码性验证；失败时以反例自动修复。

验证后的约束按代价模型选择 QUBO、p-dit 或 MFC 表示。当前代码执行 QUBO 与 p-dit+MFC 两条后端，并记录每个约束的表示选择；HUBO 是后续硬件支持高阶耦合后的扩展接口，不作为当前已实现结果声称。

### 2.2 组合优化线

LLM 输入问题、验证约束、当前解和历史 p-bit 反馈，输出候选完整解、destroy 变量、候选取值/路径、候选边、动作 logits 与置信度。候选域与当前解、几何启发式和随机探索取并集，因此 LLM 只能排序和压缩搜索，不能永久排除动作。

CVRP 是首个完整实现：动作是客户到车辆路径的重新分配，局部解码后按最小插入代价恢复路径。相同接口可迁移到调度中的工序-机器、图着色中的节点-颜色和背包中的物品-选择动作。

## 3. 能量与 p-bit 搜索

当前统一能量为：

`E_t(x) = E_obj(x) + sum_k lambda_k phi_k(x) - alpha_t sum_a z_t(a) x_a`。

第一项是组合优化目标，第二项是验证约束的确定性编译，第三项是有界的 LLM/p-bit 候选先验。语义先验是软偏置，不能代替硬约束。

p-bit 使用多链退火采样。每条链保留其历史最低能状态，随后执行确定性解码与严格 verifier。只有可行且目标严格改进的解才替换 incumbent，实现 do-no-harm 接受规则。

## 4. p-bit 到 LLM 的反馈

反馈不使用全部样本的简单平均。采样结果被分成精英可行集、普通可行集和不可行/较差集。对动作 `a` 计算精英后验 `q+(a)` 与负后验 `q-(a)`，在线更新为：

`Delta z_a = eta * A_t * [q+(a) - kappa q-(a) - pi_LLM(a)]`。

更新带衰减和裁剪。有效精英样本不足时不做强更新。校准后的 logits 同时进入下一轮 LLM 提示和 QUBO 软偏置。正式实验包含关闭反馈的对照。

## 5. 后训练

联合轨迹保存 CPP、验证反例、候选提案、p-bit 样本结果、候选召回与压缩率、最终改进和 logits 更新。

训练顺序为：

1. SFT：学习正确 CPP 和由 p-bit 结果筛选的优质候选提案；
2. 在 SFT 适配器基础上执行 DPO，在同实例、同预算下偏好约束正确且 p-bit 改进更大的提案；
3. 合并上一阶段适配器后执行 GRPO，使用记录式可验证奖励训练结构合法性与优质提案生成。

硬约束可行性优先于目标值。当前 GRPO 是基于已记录 verifier/p-bit 轨迹的离线可验证奖励，不宣称训练期间实时调用 p-bit。

## 6. 实验范围

主任务为 NL-CVRP。基线包括 greedy、random+p-bit、KNN+p-bit、LLM+p-bit、LLM+p-bit 无反馈，以及 HGS/OR-Tools。主要指标为约束精确匹配/A4通过率、最终可行率、BKS gap、目标改进、time-to-target、候选召回、候选压缩率和 p-bit 时间。

多表示消融比较 QUBO 与 p-dit+MFC。后训练比较 base、SFT、DPO 和 GRPO。完成主任务后，再用图着色或调度验证变量-候选值接口的迁移性。

## 7. 当前实现映射

- CPP 与验证：`qihc/s2e/cpp.py`、`validator.py`、`repair.py`；
- 多表示选择：`qihc/s2e/compiler.py`；
- 候选生成与安全扩展：`qihc/problems/cvrp/llm_selector.py`、`neighborhood.py`；
- QUBO 与语义软偏置：`qihc/problems/cvrp/qubo.py`；
- p-bit 求解闭环：`qihc/problems/cvrp/scheduler.py`；
- 精英 logits 反馈：`qihc/problems/cvrp/logit_feedback.py`；
- p-dit+MFC 后端：`qihc/s2e/hybrid_sampler.py`；
- 联合训练数据：`experiments/prepare_joint_training_data.py`；
- 四卡正式入口：`scripts/s2e/gpu_run_formal.sh`。

## 8. 结论边界

当前实现验证的是 LLM 候选先验能否提高 p-bit 局部搜索质量，以及约束自动生成能否通过验证后稳定进入能量模型。它不预设 p-bit 在所有规模上优于成熟求解器，也不声称任意新约束都能一次生成正确；这些必须由等预算实验决定。
