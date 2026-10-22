# 四卡 H100 + 80 核：一个任务运行 CPU、GPU、报告

适用范围：12 组 distribution-feedback 实验。不是旧版 `co_comparison_v1` 的启动器。
不修改旧入口、科学算法、提示词、实例、预算或源文件校验值。已有环境/模型/数据继续复用。

## 1. 联网 CPU：只准备，不常驻求解

在独立 checkout 中运行；不要对正在运行的 checkout 执行 pull/checkout。

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-fourcard
bash tools/runtime/distribution_job/00_prepare_only.sh
```

本命令检查已有模型身份、复制审计过的实例、复用满足完整性校验的初始回答，随后退出。
它不启动搜索、不加载模型、不重新下载权重，也不安装/修改共享环境。
已有环境及模型无需重建；首次部署仍须先完成项目原有下载与环境准备。

默认只读来源：`why/results/co_comparison_v1`。
默认新结果：`why/results/distribution_feedback_4gpu80_v1`。
若来源目录不同，两端均设置 `QIHC_DISTRIBUTION_SOURCE`。
联网端准备可省略，GPU 统一入口也会完成同样的离线检查和准备。

## 2. GPU 平台：一条启动命令

申请 **1 worker、4 H100、80 核 CPU**，共享个人目录和已有 Python 环境。
内存需容纳 32 个搜索进程、两份模型加载及缓存；80 核本身不代表内存一定足够。
不要套 torchrun，不要同时保留同一结果目录的独立 CPU/GPU 进程。

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-fourcard && bash tools/runtime/distribution_job/01_run_4gpu80.sh
```

| 部分 | 默认配置 |
|---|---|
| CPU 求解 | 32 个 spawn 工作进程；每个 BLAS/OpenMP 单线程 |
| GPU 推理 | 两个模型副本，各占两卡；不是张量并行 |
| 推理批量 | 每副本最多 8 条真实请求；沿用 OOM 拆分重试 |
| 反馈提交 | 固定第 0 种子完成立即排队，其余种子仍全部完成 |
| 报告 | CPU 独立子进程，每 60 秒刷新，结束再生成最终报告 |
| 隔离 | CPU 和报告子进程看不到 CUDA；GPU 子进程只读取请求和模型配置，不读取答案文件 |

如内存不足，重跑同一目录可降低 CPU 并行或 batch，不必重算已完成记录：

```bash
QIHC_DISTRIBUTION_CPU_WORKERS=16 QIHC_DISTRIBUTION_BATCH=4 bash tools/runtime/distribution_job/01_run_4gpu80.sh
```

有实际 CPU/内存余量时可将 CPU 工作进程调到 48（最多 64），不建议直接开满 80 核。
两副本各两卡是显存余量与吞吐的折中；不保证每卡持续满载或显存达到某个百分比。

## 3. 续跑与旧任务保护

重复第 2 步命令即续跑。原子写入的回答、每种子搜索记录、最终实例不会重算；被强制中断且
未写完的任务重新执行。批量执行可能改变浮点生成结果，实际执行配置及运行器哈希写入
`execution/*.json`，不得将新旧执行方式宣称为逐位等价。

默认新目录不会接管 `distribution_feedback_v1` 或旧 CO 实验。要接管**已经停止**的
distribution-feedback 实验，可显式指定：

```bash
QIHC_DISTRIBUTION_RUN=/inspire/hdd/global_user/czxs25210022/why/results/distribution_feedback_v1 bash tools/runtime/distribution_job/01_run_4gpu80.sh
```

科学配置/源码不匹配会拒绝续跑，不自动迁移。活动的 `cpu.lock` / `gpu.lock` 会阻止接管，
不会杀掉旧进程。每个统一任务只终止自己启动的进程组，包括搜索进程池。
如果原四卡还在跑旧任务，应另申请四卡任务或等待结束，不能靠目录隔离避免硬件竞争。

## 4. 空闲、退出与平台回收

32 路真实 CPU 搜索、提前提交固定种子反馈、两副本批量推理用于缩短依赖等待。
模型不再因暂时空队列在 120 秒后退出重载。队列空闲达到 180 秒时每 30 秒输出警告。
默认连续 **900 秒**没有待处理请求且实验未完成，将尝试完成当前种子检查点并退出整个任务，
释放资源（退出码非零）；同一命令可续跑。可用 `--max-idle 600` 改为 10 分钟。
某个 GPU 副本的队列可能先处理完，单副本空闲不等于全局无任务。

这不是平台回收豁免，也不是 GPU 利用率保证。CPU-only 收尾或长搜索依赖可能导致正常等待；
不要添加空矩阵计算、空占显存来伪造利用率。超过等待阈值应检查 CPU 日志与实际资源，而非盲目重启。
CPU/GPU/报告异常会让本任务失败并清理自己的子进程。收到 SIGTERM/SIGINT 后给进程最多
120 秒完成检查点，然后清理进程组；平台 SIGKILL 无法保证执行清理，但已落盘记录仍可续跑。

## 5. 状态、报错、自动报告

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
RUN="$BASE/results/distribution_feedback_4gpu80_v1"
cat "$RUN/job_status.json"
cat "${RUN}_reports/report.md"
ls -lt "$RUN/job_logs/"
```

每次启动各有 `*-cpu.log`、`*-gpu-0.log`、`*-gpu-1.log`、`*-report.log`。
平台主日志显示它们的完整路径和每 30 秒队列心跳。最终报告包括关键配对统计与实测速度 ETA。
ETA 自动使用本次启动的 CPU 并行数和两副本配置，不需要手动启动另一份报告进程。

## 本地验证边界

测试覆盖真实 CPU 求解和 spawn 进程池、模拟模型回复的 12 组完整闭环与续跑、固定种子反馈
等价性、活动目录拒绝接管、设备分组、故障传播、空闲超时和清理路径。
没有在远端四卡 H100 上做满模型吞吐/显存验证；首次运行需查看 placement 和 GPU 日志。
