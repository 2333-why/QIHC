> 2026-11-05 启动修复：GPU入口不再自动安装任何依赖。先在能联网的终端以相同QIHC_ENV运行00_setup.sh，再提交01_run.sh；GPU启动先离线检查依赖，缺失立即退出。

> 环境更新（2026-11-04）：所有实验统一使用 `QIHC_ENV` 指定的已有环境；未指定时优先当前激活的Conda/venv，否则固定共享 `why/.venv-qihc`。不再按CPU/GPU创建环境。此条替代下文旧环境说明。

# E004_2.1：公开数据上的主动反馈实验

已实现；服务器尚未启动。正式数据仅使用公开题：NLCO 六任务完整 900 题子集，以及 CP-Bench 10 题已适配子集。不是完整 NLCO/CP-Bench 榜单评测。第二模型关闭。

## 一键运行

通过 Git 获取代码和公开数据。在分配到 **4×H100 80GB、24 CPU 核** 的 Linux GPU 作业容器终端运行以下命令；也可把同样命令填入平台作业启动脚本。不要在没有 GPU 配额的登录节点启动。

```bash
cd /inspire/hdd/global_user/czxs25210022/why
git clone --branch codex/e004-2.1-public https://github.com/2333-why/QIHC.git qihc-E004_2.1-public
cd qihc-E004_2.1-public
bash experiments/feedback_active_v3/01_run.sh
```

第一次自动创建独立虚拟环境并安装 CPU/GPU 依赖，需要能访问 pip；数据已通过 Git 提供，不需在线下载。Python 3.10–3.12，Linux x86_64；CUDA 驱动须兼容所安装的 vLLM/PyTorch。启动时检查 jsonschema、SCIP、HiGHS、CP-SAT、四张显卡和模型文件，再启动推理。

模型默认 `/inspire/hdd/global_user/czxs25210022/why/models/Qwen3-32B`。
结果默认 `/inspire/hdd/global_user/czxs25210022/why/results/E004_2.1_active_feedback`。
通过 `QIHC_MODEL`、`QIHC_ACTIVE_RUN`、`QIHC_ACTIVE_ENV` 覆盖。重用结果目录必须保持代码、模型、数据、配置不变。

```bash
# 中断后原命令续跑；已完成调用和 CPU 任务读取缓存。
bash experiments/feedback_active_v3/01_run.sh
# 仅先运行公开开发集；后续同目录运行默认命令补齐各阶段。
bash experiments/feedback_active_v3/01_run.sh --stages development --skip-mechanism
# 如需单独检查服务器环境而不启动实验：
bash experiments/feedback_active_v3/01_run.sh --preflight-only
```

部分请求若在中断时结果未知，记录为已消耗的失败调用，不自动获得额外生成预算。

## 查看和打包结果

```bash
RUN=/inspire/hdd/global_user/czxs25210022/why/results/E004_2.1_active_feedback
tail -n 30 "$RUN/logs/pipeline.log"
cat "$RUN/report.md"
# 任意时刻生成部分报告；失败、缺组、零卡片均可报告。
bash experiments/feedback_active_v3/02_report.sh
# 运行完成后导出证据包：
bash experiments/feedback_active_v3/03_pack.sh
```

`summary.json`/`report.md` 给出建模、原题可行率、已证最优命中、失败、unknown，以及固定 M0/GOLD 的纯求解器对照。`mechanism/summary.json` 是公开开发题上控制注错的机制结果。完整调用、补丁、卡片、解、求解器检查点、模型 hash、GPU 利用率均留在 run 目录。报告的 complete 仅代表已登记阶段评价结束，不代表性能达标或所有附加审计完成。

## 数据、方法和并行设置

- NLCO：KP/MIS/MVC/MAXCUT/MCP/MDS，各 S/M/L 50 题，共 900。开发 120，主测 480，规模 300；无合成正式题。原始 CSV、固定 revision、SHA、划分随包提供。NLCO 曾用于本项目 E003，不能宣称历史未见。
- CP-Bench：10 个经明确适配的公开问题作为补充。其余 CP-Bench、MAMO、RetailOpt 原始资料只用于覆盖审计，不计入已运行题数；不臆造整数界限凑齐数量。
- 主测 C0/S/X/M/R/D/P2。主测反馈种子 17/29/43；C0 共享一份初稿。消融 P0/P1/随机钳制 PR 使用主测 120 题和两种子；重放 60 题；规模 C0/X/R/P2 两种子。
- R/D/B 是受相关工作启发的统一 IR 实现，不是作者完整代码复现。P2 是软件 p-bit 风格 heat-bath 采样，不能宣称硬件性能优势。
- 每轮反馈 10 秒、最多 4 卡/1024 token，两轮修补，每次最多 4096 输出 token，32768 上下文。超时/截断保留；不静默裁题。
- 四卡各一份 BF16 32B 模型，每卡 8 路请求、最多 16 条服务序列；连续批处理、前缀缓存，CPU 反馈/评价与 GPU 推理重叠。默认反馈 12、评价 4、机制 2 个 CPU 进程；最后 CPU 收尾扩至 16。资源利用率记录在 telemetry.jsonl，不承诺始终 100%。
- 全流程最多 26060 次 LLM 调用；实际时间由服务器吞吐决定，支持分阶段数天运行。没有执行第二模型。
- 合成 `data.py` 仅作为单元测试夹具；正式启动器拒绝合成数据 manifest。

## 正确性边界

反馈进程无标准模型输入；标准答案仅供独立评价。建模失败保留在端到端分母，同一 M0/GOLD 交叉求解用于隔离求解器能力。语义正确使用代数同一证明或小域穷举；反例可证错误；未能判定标 unknown，不视为正确或直接删除。参考最优值须求解器证明，发布的参考答案不自动视为最优证明。

所有主测初稿/标准模型运行 SCIP、pbit、CP-SAT、HiGHS、SA、PT、penalty-PT；所有最终模型交叉 SCIP/pbit，120 题主测子集补全其他后端。C0 另做独立 40 秒求解对照，补偿两轮 10 秒反馈 CPU 时间，但并不补偿 LLM 开销；实际 token 和墙钟时间另报。

控制注错使用 60 个公开开发题基础模型，得到 474 个已证错误模型与 60 个正确模型；6 个无效扰动记录为不适用，不冒充错误。该机制轨道使用 oracle 局部检查，与端到端结果分开。

本地已检查数据、反馈隔离、补丁与续跑、原生求解器和共享组件回归。H100/vLLM 真实推理与吞吐须服务器启动后验证；本地冒烟结果不算实验成绩。

详细设置和历史变更见 `docs/experiments/E004_feedback/E004_2.1_plan.md` 与 `E004_2.1_record.md`。

Git交付更新（2026-11-04）：使用 `04_cpu_prepare.sh` 准备CPU数据环境，`01_run.sh` 启动四卡完整实验。源码及原始公开数据直接纳入Git，无需上传或解压代码包。当前部署以Git提交为准，旧压缩包清单仅为历史证据。
