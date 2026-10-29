# E003-1.1：公开数据求解器对比

完整设置、来源和边界：[实验记录](../../docs/experiments/E003_public_solvers/E003-1.1_record.md)。固定提交命令见该目录发布文档。

本实验独立于 E001/E002，不读取它们的状态或等待语义 gate。默认结果为 `/inspire/hdd/global_user/czxs25210022/why/results/E003-1.1_public_solvers`。

## 顺序与资源

1. 联网 CPU：`bash experiments/public_solver_v1/00_prepare_cpu.sh`。复用项目 Python、Qwen3-32B 权重和已下载求解器；优先用随 Git 提供的可校验公开数据快照补齐 18 个固定版本 CSV，完整 900 题。省略快照的源码分发才回退 HF 国内镜像／官方源。联网 CPU 完成后退出。
2. CPU 计算任务（建议 32 核、24 workers）：`bash experiments/public_solver_v1/02_solve_cpu.sh --track canonical`。立即运行标准模型轨道，不需要 GPU/LLM 输出。
3. GPU 计算任务（一个 worker、四张 H100）：`bash experiments/public_solver_v1/01_model_4gpu.sh`。可与第 2 步同时启动；两组双卡、每组 batch=4，真实建模，无语义自检。GPU 只做生成，完成就退出。
4. GPU 生成完成后，CPU 计算任务：`bash experiments/public_solver_v1/02_solve_cpu.sh`。两轨道补齐，已完成标准模型评分不会重复。
5. 自动报告持续生成，也可执行 `bash experiments/public_solver_v1/03_report.sh`；打包执行 `bash experiments/public_solver_v1/04_pack.sh`。

如果只想先看求解器能力，先完成第 1、2 步即可；标准模型轨道与自然语言建模收益不能混为一谈。CPU 不需要等整个 LLM 队列：默认运行会消费当前已有模型，耗尽就退出；稍后用同一命令续跑。不要同时启动两个 CPU 求解主进程。

80 核 CPU 计算配额可设置 `QIHC_CPU_WORKERS=64`；32 核建议默认 24。每个求解器单线程，实际 CPU 硬件和并行度写入 launches。资源回收阈值由平台决定，代码不能保证不回收：本方案通过计算阶段分离消除 GPU 等 CPU 的空闲尾段，逐步落盘使外部中断可恢复，不运行无用矩阵计算或虚占显存。

## 失败、断点与公平性

- 每道题一个固定 LLM 数学模型，七种后端共用；语义不正确但可编译仍进入搜索。
- 无效模型对全部后端统一计 invalid_model；执行错误、原生进程崩溃、硬超时单列，继续其他任务，不从分母抹掉。
- 每题每后端使用独立子进程，避免 OR-Tools、SCIP、HiGHS 动态库冲突。每个预算/种子完成即原子落盘；硬超时只影响当前题后端未完成的项。
- 外部 SIGTERM 时停止子进程并保留检查点；SIGKILL 后下次启动继续。因实验错误跳过的结果不自动重抽，防止选择性重跑；修正算法需另记迭代。
- GPU OOM 批次拆小；单题异常记录后跳过；设备级崩溃最多自动重启两次。反复加载失败等基础环境错误仍显式终止，不能假装全部建模完成。
- 配置、代码、数据和冻结输出身份冲突仍停止，防止把不同实验悄悄拼接。旧运行目录不覆盖。
- 不承诺所有 LLM 输出都正确；保证坏输出可记账、可跳过、不会因语义 gate 阻断全套实验。

## 当前进度

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
RUN="$BASE/results/E003-1.1_public_solvers"
for f in modeling_status.json solver_status.json; do
  if [ -f "$RUN/$f" ]; then echo "=== $f ==="; cat "$RUN/$f"; echo; fi
done
cat "$RUN/report.md"
```

`complete` 只表示预设任务有终态，不代表没有失败、语义正确或 p-bit 胜出。必须同时看失败表、有效模型比例、原题可行率和配对样本数。
