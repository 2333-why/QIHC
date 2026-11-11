# E005-1.9 P-VCD 双向闭环验证

[实验记录与预算](../../docs/experiments/E005_posttraining/E005-1.9_record.md)。本轮独立于E005-1.7/1.8；不修改旧环境、旧权重、旧结果。6组：sft、pvcd、cycle、cycle_uniform、cycle_static、cycle_no_guard。最后4组各训练3种子，正式13080条评价。

## 联网CPU/登录端：Git部署与准备

建议独立checkout，避免切换正在运行作业的代码：

```bash
cd /inspire/hdd/global_user/czxs25210022/why
git clone --branch codex/e005-1.9-pvcd-cycle --single-branch https://github.com/2333-why/QIHC.git QIHC-E005-1.9
cd QIHC-E005-1.9
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_HISTORY=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.7_eight
export QIHC_MODEL=/inspire/hdd/global_user/czxs25210022/why/models/Qwen3-32B
export QIHC_CYCLE_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.9_pvcd_cycle
bash experiments/pvcd_cycle_v1/00_cpu.sh
```

已有该专用checkout时，只在CPU/登录端、该实验未运行时更新：

```bash
git fetch origin codex/e005-1.9-pvcd-cycle
git checkout codex/e005-1.9-pvcd-cycle
git pull --ff-only origin codex/e005-1.9-pvcd-cycle
```

按交付消息中的SHA固定版本后准备；准备清单会记录真实SHA，源码或输入改变后拒绝续跑。准备检查同一共享环境所有CPU/GPU依赖、数据锁、完整本地模型/分词器、三种子SFT/P-VCD适配器、训练SFT父预测，进行SCIP和p-bit运行预检，并复制小型输入快照。成功输出`PVCD_CYCLE_READY`。反复执行同一准备命令应幂等。

本轮不增加依赖。缺包时先在联网CPU端处理，在现有实验结束前不要更新共享环境；历史依赖约束见`experiments/feedback_active_v3/requirements-{cpu,gpu}.txt`。安装源默认清华，可通过`PIP_INDEX_URL`覆盖，清华失败可用阿里云重试相同约束。GPU节点不运行pip/git/下载命令。历史输入必须是包含适配器的原始结果目录，单独的分析报告不能替代。

## 离线GPU作业：四卡启动与自动继续

调度器申请4×H100 80GB、建议24 CPU核；沿用调度器设置的`CUDA_VISIBLE_DEVICES`，不要在脚本里覆盖为其他作业的设备。以下作为作业命令：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/QIHC-E005-1.9
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_HISTORY=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.7_eight
export QIHC_MODEL=/inspire/hdd/global_user/czxs25210022/why/models/Qwen3-32B
export QIHC_CYCLE_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.9_pvcd_cycle
bash experiments/pvcd_cycle_v1/01_gpu.sh all
```

脚本先pilot，之后只有机制门槛通过才进入formal。失败退出时查看`pilot/gate.json`和相应worker日志，不修改配置继续覆盖原结果。formal从冻结SFT重新训练，与pilot目录和权重隔离。已检查依赖、源码、历史指纹后才加载模型；自动启用HF/Transformers/Datasets离线模式；实际GPU驱动兼容性、BF16前后向及显存本地预检。

**续跑：在相同checkout、环境变量、结果目录下重新提交同一条`01_gpu.sh all`命令。** 保存的优化器/步数继续，完成预测与评价跳过，旧结果保留。也可仅运行`01_gpu.sh pilot`或`01_gpu.sh formal`；formal仍检查pilot门槛。不要同时对同一结果目录提交两个作业（进程锁会拒绝）。新配置必须另立迭代、另选结果目录。

## 查看、报告和返回证据

```bash
tail -n 40 "$QIHC_CYCLE_RUN/formal/logs/17-cycle.log"
bash experiments/pvcd_cycle_v1/02_report.sh pilot
bash experiments/pvcd_cycle_v1/02_report.sh formal
```

报告可在运行中读取原子保存的已完成文件，覆盖不齐明确标识，不输出全量排名区间。保留`manifest.json`、`gpu_preflight.json`、`pilot/gate.json`、`formal/report.md`、`formal/summary.json`、各组`trace/`、`steps/`、`evaluations/`和日志。最终回传这些小型证据，不需要上传`adapter.pt/checkpoint.pt`或基础模型。`summary.json`有完整初稿转移矩阵、分题型/变量数/种子、训练奖励、修复漏斗、token与耗时。

先看pilot实测单步耗时及修复触发率，再评估正式作业所需时长；本地测试无法预测服务器四卡吞吐。初稿/修复各有4096token上限、采样150秒上限，但不是整个作业的完成时间保证。
