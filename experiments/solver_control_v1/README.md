# E005-1.12：扩充控制训练数据，直接评测780题

当前交付默认采用 **CPU全量准备 → GPU训练并评测780题**，无需先完成pilot。用户于2026-11-13确认每题16个训练状态：90道冻结训练题，各4条独立起点轨迹，初始温度依次0.5、0.25、1、2，每条4个状态，最多1440对偏好。平局或复测不一致仍丢弃，不保证产生1440对有效偏好。780题只用于评测，不用于标签构造。数据扩充不等于增加独立训练题，当前独立训练题仍为90道。

七组不变：fixed、schedule、rule、random、best_sft、trajectory_sft、trajectory_dpo。每组480主测+300规模，共5460条评测。评测仍32链、128步热身+4×128步控制，单题20480比特更新。SFT各2遍、DPO1遍，实际步数按确认标签计算；上限分别2880、2880、1440步。新配置为`config_full.json`，与旧`config.json`分离。

2026-11-13本地完整90题造数：1440状态得到479对有效偏好，覆盖80题，五类动作都有，但428对选择降温。固定降温基线必须保留。按该计数SFT各958步、DPO479步；服务器仍重新核验冻结输入并造数，GPU训练与780题性能待运行。

## 当前交付：服务器拉取与CPU准备

在可联网CPU/登录节点使用独立checkout；已有E005-1.11作业保留原checkout、原SHA和原目录。

```bash
cd /inspire/hdd/global_user/czxs25210022/why
git clone --branch codex/e005-1.12-full780 --single-branch https://github.com/2333-why/QIHC.git QIHC-E005-1.12
cd QIHC-E005-1.12
# 用交付消息中的实际SHA固定版本；已clone时先fetch再checkout。
git fetch origin codex/e005-1.12-full780
git checkout --detach <交付SHA>
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_CYCLE_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.9_pvcd_cycle
export QIHC_CONTROL_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.12_controller_full780
bash experiments/solver_control_v1/03_cpu_full.sh
```

CPU端核验共享环境、本地模型、旧冻结划分与源文件哈希，准备90题的全部标签。它不依赖E005-1.9或E005-1.11的pilot成绩，只依赖E005-1.9的冻结manifest/inputs和本地模型仍在。缺依赖时在联网CPU端补齐后重跑，不创建新环境或在运行中升级。准备成功才进入GPU阶段。

## GPU训练、全量评测及续跑

申请至少2×80GB且支持BF16的GPU（建议H100）和16 CPU核。在GPU节点重新设置相同变量；不运行git、pip或下载。

```bash
cd /inspire/hdd/global_user/czxs25210022/why/QIHC-E005-1.12
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_CONTROL_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.12_controller_full780
bash experiments/solver_control_v1/04_gpu_full.sh
```

入口自动设置离线模式，先验证实际显存与BF16前反向，再运行两个SFT和四个CPU策略，随后从本轮轨迹SFT继续DPO。每个训练组只有完成全部优化步、保存适配器且损失/梯度有限、有非零梯度才开始自身780题评测。四卡可用但最多占用两卡；p-bit搜索在CPU软件运行。

```bash
# CPU准备中断：重跑03_cpu_full.sh；GPU训练/评测中断：重跑04_gpu_full.sh。
# 任意共享环境节点查询进度；不需要GPU，尚未齐全时明确标记未完成。
bash experiments/solver_control_v1/05_report_full.sh
```

续跑保持同一SHA、环境和结果目录。造数/评测每题原子落盘，训练每8步保存优化器；不得将E005-1.11目录用于本轮。CPU与GPU脚本都会固定加载`config_full.json`。报告位于`$QIHC_CONTROL_RUN/full/report.json`，只有5460条齐全且训练检查通过才生成三组配对比较区间。分母、完成数、可行率、最优命中率、参考证明覆盖、动作分布、搜索/决策/episode时间分别报告；未完成时的比例仅代表已完成样本。

本地兼容性检查及CPU造数不是服务器GPU实测。实验记录、实际标签数与验证证据见[E005-1.12](../../docs/experiments/E005_posttraining/E005-1.12_record.md)。

---

## E005-1.11原协议说明（历史入口保留）

本轮固定正确的数学模型，训练 Qwen3-32B LoRA 选择搜索动作，不训练建模、不编辑约束。属于 E005 后训练的新迭代，不替代 P-VCD 和正在运行的 E005-1.10。

每段128次随机单比特更新/链，32链；共享128步热身后作4次决策，共640步、20480次比特更新。链状态连续保留，所有经过的赋值按原始可行性和目标排序归档。五个动作：保持、降温、升温、提高全部惩罚、将部分惩罚从低违反率行移到高违反率行。惩罚改变后重算能量，不跨不同能量函数比较能量高低。

七组：固定、固定降温、规则控制、随机控制、仅当前最好解的SFT控制器、含近四段轨迹的SFT控制器、轨迹SFT后再作一遍分类动作DPO。后两组的轨迹包含可行比例、约束违反率、多样性、变化率和改进次数。模型输出限定为五个单token动作；不是自由文本生成后解析。DPO是五动作分类策略上的偏好目标，不是原论文建模方法的复现。

训练数据仅来自旧冻结划分的90题；验证24题。每题4个预先探索状态，每个状态5动作×2个初选随机流，优劣候选再用4个独立随机流同状态复测。先比较全链最好解质量，打平时再比较各链历史最好解的平均质量；该字典序在平均结果中为正、至少一半复测获益且无负向复测才入库；平局不造标签。这是有限次数经验确认，并非统计显著性证明。奖励以原始可行性优先，在可行时优化按目标绝对上界归一化的目标；不使用参考最优值和原始能量下降作为奖励。输入不含case ID、未来动作成绩或参考答案。

## 先导与正式筛查边界

先做12训练题pilot和24题验证。造数门槛：至少3道独立题有确认偏好、至少一个非保持标签、至少两类chosen动作；训练/评测门槛：七组齐全、各训练组实际有梯度。门槛不依据验证准确率选择配置。不满足就停止并分析，不能把平局强行变成优劣对。

通过pilot后从基座重新开始90题单种子screen，测试480主测+300规模=780题，每题相同seed17，七组共5460条。pilot168条。此轮是**正确模型求解轨道**，不能和“自然语言→数学模型→求解”的42.34%直接比较。主测、规模和780题等权总分分别报告；测试历史已被查看，属于探索性验证，不能称未见测试。参考最优仅在决策全部完成后用于评分。

主比较：轨迹SFT−最好解SFT、轨迹DPO−轨迹SFT、轨迹SFT−规则。原题为单位5000次配对bootstrap，三比较Bonferroni 98.33%区间。单种子不提供跨种子稳定性结论。所有组匹配比特更新次数，**总墙钟不匹配**；报告搜索、决策与episode时间，模型加载和训练成本另保留日志。动作只针对下一128步形成训练标签，是短视离线策略，不是在线长时域RL。

## Git 部署与统一环境

联网CPU/登录节点操作；使用单独checkout，GPU节点不执行git/pip/下载。默认依赖与模型均复用已有共享环境和E005-1.9的冻结输入；无需安装新包，不创建环境。

```bash
cd /inspire/hdd/global_user/czxs25210022/why
git clone --branch codex/e005-1.11-controller --single-branch https://github.com/2333-why/QIHC.git QIHC-E005-1.11
cd QIHC-E005-1.11
git rev-parse HEAD
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_CYCLE_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.9_pvcd_cycle
export QIHC_CONTROL_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.11_controller
bash experiments/solver_control_v1/00_cpu.sh pilot
```

已有独立checkout可先 `git fetch origin codex/e005-1.11-controller`，再 `git checkout --detach <交付SHA>`。首次CPU准备把实际SHA、代码、包版本、模型文件内容哈希与输入快照写入manifest；绑定后拒绝源码或环境改变。完整模型哈希可能耗时。准备不依赖旧E005-1.9的pilot通过，但要求其冻结manifest和数据仍在。

GPU作业申请至少2×H100 80GB、建议16 CPU核；4卡作业也可，当前最多使用2卡，不声称4卡满载。GPU端同样设置上述环境变量，在共享checkout运行：

```bash
bash experiments/solver_control_v1/01_gpu.sh pilot
bash experiments/solver_control_v1/02_report.sh pilot
```

两个SFT控制器分别占一卡，四个传统策略在CPU并发；轨迹DPO随后从该轮轨迹SFT继续训练。GPU本地检查实际显存、BF16前反向算子，记录torch/CUDA与显卡。控制动作在GPU推理，p-bit采样在CPU软件运行，不是真机p-bit硬件训练。

pilot通过后，在CPU节点生成全训练数据，再在GPU节点运行screen：

```bash
# CPU/登录节点
bash experiments/solver_control_v1/00_cpu.sh screen
# GPU作业节点
bash experiments/solver_control_v1/01_gpu.sh screen
# 任意共享环境节点，只读结果汇总
bash experiments/solver_control_v1/02_report.sh screen
```

上述各命令原样重跑即续跑。造数每题落盘，训练每8步保存优化器和LoRA，评测每题落盘；中断可重算最多7个训练步或一题评测，累计时间不含未保存的中断成本。不要覆盖旧结果或切换绑定源码。日志为`$QIHC_CONTROL_RUN/{pilot,screen}/logs/*.log`；回传`report.json`、`label_gate.json`、训练`steps/`和manifest即可初审，权重和完整轨迹留在实验存储。

当前为实现交付，真实32B训练、有效标签数量、GPU吞吐、780题成绩均待服务器返回。详细记录：[E005-1.11](../../docs/experiments/E005_posttraining/E005-1.11_record.md)。
