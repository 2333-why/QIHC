# E004-1.1：多机制反馈实验

方案：[统一实验方案](../../docs/experiments/E004_feedback/E004-1.1_plan.md)。
实施记录：[运行与验收记录](../../docs/experiments/E004_feedback/E004-1.1_execution.md)。

运行条件：一个 worker，四张 **H100 80GB**，约 80 个 CPU 核；默认四个单卡 Qwen3-32B 副本、每副本 batch=1、48 个单线程 CPU 子进程。共享环境和权重沿用 E003，不重新下载模型，不改旧实验。CPU 联网节点只准备，搜索、推理和报告在同一 GPU 作业中运行。

## 顺序运行

先在联网 CPU 节点创建独立工作树；若目录已存在，先检查是否是此前的 E004 工作树，不覆盖或强制切换正在运行的目录：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && git fetch origin llm-cvrp && git worktree add --detach ../qihc-E004-1.1 origin/llm-cvrp
```

CPU 准备（需要原 E003 服务器目录仍存在）：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.1 && bash experiments/feedback_study_v1/00_prepare_cpu.sh
```

四卡 GPU 作业的唯一启动命令；不要套 `torchrun` 或每卡 `srun`：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.1 && bash experiments/feedback_study_v1/01_run_4gpu.sh
```

中断后仍用同一条 GPU 命令。无需重跑准备；配置、数据和源码不一致会拒绝混用结果。状态按每题、每轮、每阶段恢复，完整响应不重复生成。

CPU 节点轻量查看进度和日志新鲜度：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.1 && bash experiments/feedback_study_v1/05_status.sh
```

自动报告在运行中周期更新。手动生成含配对统计的报告：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.1 && bash experiments/feedback_study_v1/03_report.sh
```

打包（大型完整诊断包可能较大）：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.1 && bash experiments/feedback_study_v1/04_pack.sh
```

若所有推理已完成、仅剩 CPU 评分，可在 CPU 计算作业使用 `02_resume_cpu.sh`。如还有模型请求，该脚本保存 `needs_gpu` 后退出，不假装完成；然后恢复四卡命令。不要与四卡脚本同时启动同一结果目录，运行锁会阻止双写。

## 目录与环境

默认 `QIHC_BASE=/inspire/hdd/global_user/czxs25210022/why`：

- 环境：`$QIHC_BASE/envs/qihc-cu128/bin/python`，脚本直接使用此解释器，无需手动 conda activate。
- 权重：`$QIHC_BASE/models/Qwen3-32B`，可在准备时用 `QIHC_MODEL` 指定已有目录。
- E003 只读来源：`$QIHC_BASE/results/E003-1.1_public_solvers`，可用 `QIHC_E003_RUN` 指定。
- E004 结果：`$QIHC_BASE/results/E004-1.1_feedback`，可用 `QIHC_E004_RUN` 指定独立目录；准备和运行须一致。
- 若设备显存不适合每卡一个模型，在**首次准备时**可加 `--gpu-replicas 2`，改用两个双卡副本、batch=2；可用 `--cpu-workers N` 指定 1–64 个 CPU 子进程。配置写入 manifest，已开始的实验不可直接改配置续跑，须用新结果目录记录运行差异。
- `status.json`：阶段、完成数、CPU 并发、模型队列、当前阶段 ETA。
- `report.md`、`summary.json`、`paired_public.json`、`paired_extension.json`：报告。
- `logs/launch-*.log`、`logs/gpu-*.log`、`logs/cpu/`：启动及子进程异常。
- `states/`、`rounds/`、`mathematics/`：阶段检查点、完整每轮证据、按内容哈希保存的数学模型。
- `evaluation/`：独立原题检查与参考，只供评分；`requests/` 不含 gold 或标准答案。

## 执行含义和限制

1. 自动依次完成开发调试、开发验收、900 题公开集、1200 题扩展集。开发集单独报告，不混入正式结果。验收要求真实模型输出过可执行补丁并进入搜索；不按语义正确率筛题。
2. 正式 56160 条轨迹，加开发 1440 条，默认总计 **57600**。终止评分数是工程完成度，不等于正确解数。
3. 单题超时、格式错误、原题不可行结果保留并继续。持续设备/进程异常会熔断；应查看对应日志后续跑，不无限隐藏错误。
4. 模型调用开始前记录事件。平台回收时若没有返回响应，记为已尝试且 token 成本未知，不作为免费重试。
5. 真实模型推理依赖服务器验收；本地 CPU 与模拟响应测试不代表真实 32B 建模效果。长上下文 OOM 计入失败并保留日志，不悄悄截断原题。
6. 不执行无用矩阵运算或人为占显存。四卡常驻独立模型并通过 CPU/GPU 队列并行处理真实任务；无法保证平台绝不回收，依靠原子落盘、进度监控和恢复。
7. 所有生成时间按实际 batch 分摊，不冒称 GPU 小时；CPU 子进程事件另存启动/导入/计算总墙钟。未知成本和初稿共享成本单独报告。
