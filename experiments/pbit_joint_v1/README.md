# E005-1.13：分别改进建模和p-bit控制，并测组合收益

本轮采用已有加权SFT和分类SFT，重点利用p-bit的多链、条件采样和可钳制接口。相关操作也能在其他采样器中实现，不主张它们为p-bit独有。旧实验保持原代码和结果；本轮统一在自然语言输入的780题上端到端比较，不再把正确模型控制轨道的84%与建模轨道的41%混比。

## 比较矩阵

|建模器|固定降温 schedule|规则控制 rule|后训练控制 learned|
|---|---|---|---|
|历史建模SFT base|主基线 B|控制规则对照|只改控制 C|
|等步数普通SFT plain|额外标准答案训练对照|辅助对照|辅助组合|
|p-bit反例加权SFT pbit|只改建模 M|无控制训练的组合对照|两者组合 MC|

9组各780题=7020条，480主测和300规模分别报告。每行三个控制器使用完全相同的一份建模预测，不分别生成。每题都按原题可行性/目标评分，无效建模计失败，不从分母中剔除。主比较M−B、C−B、MC−B及交互MC−M−C+B。

## 两处改进

**建模：** 复用90道冻结训练题及历史SFT预测，在候选/训练gold两侧运行多链及条件钳制采样，按原题认证反例。标准正确模型是训练目标，能唯一定位到约束或目标片段的反例，将对应token权重从1提高到2。模型无效或定位不明确时保持普通监督，不凭采样通过认定模型正确。plain与pbit均从同一个建模SFT继续90题×2遍=180步、lr5e-6，目标完全相同，区别仅token权重；训练归一化损失，避免简单扩大总梯度。

**控制：** 32链拆成16保护链+16探索链。保护链始终降温、不会钳制；探索链可降温、短暂升温、根据条件概率将一个不确定二元变量钳至群体少数值。钳制仅一段，保留所有历经最好解；无合适二元变量或只有一个可变变量时，clamp与cool等效并记录空钳制。规则对照在多样性低于0.25时选clamp（不可用则heat），否则cool。

控制标签来自三个建模器在90道**训练题**上的预测，按题ID+模型内容去重。每个有效模型4温度×4状态，上限4320状态。默认使用自然搜索状态，不强制链群收敛。每状态试三个动作，比较第一动作之后继续降温直到剩余时域结束的最终最好解；不用群体均值破平。非降温候选必须在4次独立复测中无负收益且至少2次正收益，才成为标签；否则选cool。这样标签直接回答“是否值得偏离降温”。分类损失按题内样本数及动作频数平方根加权，不伪造或强制非降温标签。

准备阶段在90个训练gold模型上测得1440状态中仅5个确认非降温收益、覆盖4题；尝试人为低多样性状态后只有4个、仍覆盖4题，因此未启用该配置。诊断使用正确训练模型，不能代替正式生成模型上的标签或测试成绩。控制信号可能仍稀疏，最低运行门槛不代表数据已足够或方法有效；最终应检查是否再次恒定降温。首动作后固定cool的标签也不保证多次自适应动作组合最优。

控制器从Qwen3-32B基座新建独立LoRA（q/v rank16，scale2），2遍分类SFT，lr2e-5；建模器则从历史建模SFT继续。两个适配器分别保存、顺序使用，**不直接相加或合并权重**。所有组合复用同一个控制器；输入为模型规模/目标方向/归一化尺度、条件概率、当前最好解和近四段统计，不再反复输入完整约束矩阵，以减少推理上下文。不能看到gold、已知最优、题ID或未来分支成绩。

每题128步热身+4×128步控制，每链640步、32链共20480次heat-bath尝试。各组匹配此预算，但临时钳制有额外赋值写入，另行记录；不声称等总物理操作、等CPU/GPU墙钟或真机p-bit性能。固定变量模型无需采样，记录0次尝试。单题episode时间不含模型生成、训练、加载及参考求解，训练成本另报告。

## CPU准备（联网CPU/登录端）

先在独立目录拉取，勿更新仍在运行的旧实验checkout：

```bash
cd /inspire/hdd/global_user/czxs25210022/why
git clone --branch codex/e005-1.13-joint --single-branch https://github.com/2333-why/QIHC.git QIHC-E005-1.13
cd QIHC-E005-1.13
git fetch origin codex/e005-1.13-joint
git checkout --detach FETCH_HEAD
git rev-parse HEAD  # 首次运行应与本次交付SHA一致，之后保持此checkout续跑
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_CYCLE_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.9_pvcd_cycle
export QIHC_JOINT_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.13_joint
bash experiments/pbit_joint_v1/00_cpu.sh
```

沿用同一Python3.11环境、已有本地32B模型、历史建模SFT适配器及E005-1.9冻结数据。没有新依赖，不自动创建环境或安装/升级；现有环境缺包时在联网CPU端补齐原约束后再跑。CPU准备核验依赖、源码、模型/数据/父适配器指纹、原生SCIP及p-bit，并生成建模反例证据。`QIHC_CYCLE_RUN`指向旧冻结目录，不依赖旧pilot是否通过。

## GPU运行（至少2张BF16 80GB GPU，建议16CPU核）

```bash
cd /inspire/hdd/global_user/czxs25210022/why/QIHC-E005-1.13
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_JOINT_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.13_joint
bash experiments/pbit_joint_v1/01_gpu.sh
```

入口自动离线，GPU端不运行git/pip或下载。全流程无需手工切回CPU：

1. 两个GPU各训练plain/pbit，并生成训练题和780评测题的冻结模型；base读取历史SFT预测。
2. GPU作业内的CPU进程基于训练题预测造控制标签，随后评分六个schedule/rule组合。此阶段GPU可能空闲。
3. 至少3道独立题有确认非降温收益、标签至少两类时训练控制器；若不足，保存原因与六组建模/规则结果后停止，不把未训练策略冒称新控制后训练。
4. 评分三个learned组合；自动生成7020条覆盖报告及主效应/交互项。

最多占两张GPU；没有开机自更新或自动多种子实验。服务器32B显存、依赖算子与真实训练吞吐以本地GPU预检及作业日志为准，开发机小模型测试不替代服务器验证。

## 查询与续跑

```bash
# 任意共享环境节点，不需要GPU
cd /inspire/hdd/global_user/czxs25210022/why/QIHC-E005-1.13
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_JOINT_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.13_joint
bash experiments/pbit_joint_v1/02_report.sh
```

CPU准备中断重跑`00_cpu.sh`；GPU任一阶段中断重跑`01_gpu.sh`。保持同一SHA、环境和结果目录。建模证据、标签、预测、评分按题原子保存；训练每8步保存优化器，最多重算未保存7步。已有完整文件跳过，不删除旧结果。报告可在未完成时查询；只有覆盖齐全且训练证据完整才`passed=true`及生成四项98.75%配对bootstrap区间。

回传`manifest.json`、`report.json`、`control_gate.json`、`gpu_preflight.json`及失败时`logs/*.log`即可初审；进一步机制审计需`cards/`、`control_labels/`、`models/*/steps/`、`controller/steps/`、`evaluations/`。完整适配器/检查点留服务器，不入Git。

详细配置与本地证据：[E005-1.13记录](../../docs/experiments/E005_posttraining/E005-1.13_record.md)。这轮是在已暴露780题上的单种子探索，不能据其选参后再声称未见测试泛化；新留出集和跨种子确认留给后续迭代。
