# E005-1.10：单种子机制筛查

[实验记录](../../docs/experiments/E005_posttraining/E005-1.10_record.md)。四组sft/pvcd/positive/localized；三个训练组从同一个SFT开始，预算一致。先12题pilot，再90题单种子screen，共96+3640条评价。不自动跑三种子正式实验。

## CPU/登录端准备

必须新checkout，旧E005-1.9保留用于复现：

```bash
cd /inspire/hdd/global_user/czxs25210022/why
git clone --branch codex/e005-1.10-verified --single-branch https://github.com/2333-why/QIHC.git QIHC-E005-1.10
cd QIHC-E005-1.10
git checkout --detach origin/codex/e005-1.10-verified
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_CYCLE_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.9_pvcd_cycle
export QIHC_VERIFIED_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E005-1.10_verified
bash experiments/pvcd_verified_v1/00_cpu.sh
```

按交付回复中的完整SHA固定checkout后准备。前轮manifest定位真实模型、同种子17 SFT适配器及预测路径，不依赖种子29/43的训练父预测；核对数据/模型/适配器哈希与环境。输入快照固定，已有不同配置的结果目录会拒绝覆盖。成功输出`VERIFIED_SCREEN_READY`。无新增依赖，不改变共享环境；缺包只在联网CPU端、无实验占用环境时按旧约束处理，国内镜像默认清华、允许PIP_INDEX_URL覆盖。GPU不下载。

## 四卡GPU作业启动与续跑

申请4×H100 80GB，建议24CPU核，保留调度器CUDA_VISIBLE_DEVICES：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/QIHC-E005-1.10
bash experiments/pvcd_verified_v1/01_gpu.sh all
```

默认环境和结果路径与上面相同。若CPU准备时覆盖了路径，GPU端也需导出相同变量。`all`先pilot、满足执行与正样本门槛后自动screen；正式训练不继承pilot权重。也可分别指定pilot/screen。中断后重提完全相同命令，逐题预测和优化器断点继续；不删除checkpoint，不覆盖旧目录。

修复仍为0会在报告显示repair_active=false；本轮允许继续screen检验已验证初稿正样本路径，不代表修复成功。若正样本路径也未触发，screen不会启动。

## CPU报告、评分续跑与回传

```bash
bash experiments/pvcd_verified_v1/02_report.sh pilot
bash experiments/pvcd_verified_v1/02_report.sh screen
```

如果预测已完成而作业在评分时被中断，可在CPU端单独补评分，无需占GPU：

```bash
/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128/bin/python -m experiments.pvcd_verified_v1.run score screen
```

CPU评分入口同样检查冻结数据与依赖，加载本地数学求解器，不加载LLM。结果在`results/E005-1.10_verified/{pilot,screen}/`，每组日志`logs/<arm>.log`；返回manifest、report.md、summary.json以及有代表性的trace，无需返回大适配器。保存全部原始证据，训练修复和验证集分开解释。
