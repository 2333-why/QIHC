# E004_2.5 冻结基座反馈

联网登录端使用既有共享环境 qihc-cu128，指定 QIHC_E004_RUN 为原始 E004_2.2_active_feedback 结果目录，QIHC_OPT_RUN 为专属新结果目录。无需适配器。准备只读模型与历史记录，不升级共享包。历史 initial、manifest、评价和原始 requests 应保留；旧响应缺失时编译修复可使用题面及解析诊断。

```bash
bash experiments/optimal_feedback_v1/00_cpu.sh
```

看到 E004_2.5_CPU_READY 后，在调度器提交单独4×H100 80GB、24核CPU离线GPU作业。历史Transformers直接generate，每卡一个持续驻留的原始基座副本。pilot 和 formal 使用独立输出。续跑重复对应命令，保留初稿、逐题结果与评分缓存。

```bash
bash experiments/optimal_feedback_v1/01_gpu.sh pilot && bash experiments/optimal_feedback_v1/01_gpu.sh formal
```

CPU查看部分或最终报告：

```bash
bash experiments/optimal_feedback_v1/02_report.sh formal
```

A 无反馈，B 仅修编译失败，C 使用SCIP寻找目标排序冲突，D 使用多温度p-bit寻找冲突。E为p-bit单字段候选枚举，F为多字段候选枚举，G为SCIP+p-bit加二次确认，H直接采用来源目标。新增组仍共享编译失败修复。C/D仅能修改单个目标字段；无证据、缺少不同的验证赋值对、复核失败时保留初稿。报告和配对统计文件位于结果目录的 formal/report.md、formal/summary.json。模型文件指纹或初稿改变时须使用新结果目录。完整设置见[实验记录](../../docs/experiments/E004_feedback/E004_2.5_record.md)。

默认 QIHC_PROBE_LANES=2、QIHC_SCORE_WORKERS=8。正式八组17440条。formal/logs/gpu.csv记录可用的GPU采样；formal/timing记录最近一次各worker服务时间。创建新工作树并显式设置QIHC_OPT_RUN为E004_2.5_base_optimal_feedback，避免复用旧实验目录。
