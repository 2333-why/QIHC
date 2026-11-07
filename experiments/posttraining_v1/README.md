# E005-1.7 八组公开数据比较

从 E005-1.4 的训练入口扩展。结果目录默认 /inspire/hdd/global_user/czxs25210022/why/results/E005-1.7_eight。CPU-GPU 共用 envs/qihc-cu128，四张独立分配的 H100 80GB、24核CPU。建议使用独立Git工作树部署固定SHA。

| 组 | 实现 |
|---|---|
| sft / solver_rl / sdpo / pvcd | 原四种建模训练，独立生成IR |
| uniform_weight / rule_weight | 在同一种子的SFT模型上使用全局权重 / 用采样摘要和规则修订 |
| best_lm / trajectory_lm | 同一个SFT建模结果、同一探针。能量调整策略从Qwen3-32B分别做能量动作SFT和冻结参考DPO，只改变训练和部署时是否提供分温度轨迹摘要 |

## 联网CPU

```bash
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_POST_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.7_eight
bash experiments/posttraining_v1/00_cpu.sh
```

会检查本地模型、现有软件环境、公开数据、标准模型标签，并校验Git内的锁定偏好文件，恢复90个训练题的校准和复测记录。19题提供可用偏好。若要重新生成，在新结果目录设置 QIHC_REBUILD_PREFS=1，默认12个进程，QIHC_PREP_WORKERS可调整。每题输出原子落盘。中断后用同一命令续跑。等待 E005_EIGHT_CPU_READY。若没有获得真实偏好，日志报告原因。

## 离线GPU

```bash
bash experiments/posttraining_v1/01_gpu.sh pilot
bash experiments/posttraining_v1/01_gpu.sh formal
```

pilot只评开发验证的24题，1个种子，独立适配器；formal从基座重新训练。重新提交相同GPU命令即续跑。SFT检查点供其余建模组初始化，SFT预测供四个能量组共用。A/B用CPU执行，C/D及原四个可训练组按种子调度到四卡。评分并行、逐题落盘。训练日志在对应模式的 logs。数据与权重不在GPU下载。

## 规模和报告

开发120题×1、主测480×3、规模300×2、CP-Bench 10×2，每组2180轨迹，八组17440轨迹。开发集含90道训练题，它只用于诊断。E004特有的反馈重放和反馈消融属于原题子集，不增加本次独立样本量。

```bash
bash experiments/posttraining_v1/02_report.sh formal
```

部分报告分阶段给各组轨迹进度、八组共同题、与E004 C0/P2的同题比较，以及20秒末端结果和固定更新结果。默认只读 E004_2.2 运行目录，可显式设置 QIHC_E004_RUN。外部身份不兼容时，八组内部配对仍可查看。开发以外的模型完整正确、原题可行和已证最优分别报告；能量组复制SFT的IR，所以完整建模正确率和SCIP成绩是共享输入的参照。固定更新列对所有方法使用统一的p-bit采样器，每温度32链×128次探针更新，加32链×512次最终搜索；保留探针期最好样本。固定更新与20秒结果独立。20秒名义预算中能量组扣除探针CPU时间，模型推理耗时额外记录。

[实验记录](../../docs/experiments/E005_posttraining/E005-1.7_record.md)

标准模型上的能量复测保存在 formal/gold 下，四组额外8720条，仅评价能量选择。这些记录不计入八组端到端的17440条。同一批910题，合计26160条评价。修复训练数据冻结于 energy_labels.json，19对来自19个独立原题，低于早期方案建议门槛；在该规模下应作为探索性后训练比较。
