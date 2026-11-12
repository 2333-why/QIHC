# E005-1.11：学习控制 p-bit 求解过程

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
