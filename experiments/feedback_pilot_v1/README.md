# E004-1.2：18 题反馈快速筛查

[完整实验记录](../../docs/experiments/E004_feedback/E004-1.2_record.md)。18 题，六任务 × 三难度每层一题，seed=0；C0/C1/U2/B2 四组，两轮，共 72 条轨迹、最多 216 次新 LLM 调用。只读复用 E003 初稿、参考以及同题七后端 20 秒评分，无需新建模或重跑七后端。

本入口用于快速筛查，split 明确标为 `pilot`；不伪造 development_acceptance，不将旧实验验收用于新配置。主要方法预先固定 U2，不按返回结果换主方法。选题按固定哈希，不排除无效初稿；顺序按 S/M/L 分层，每个难度先跨六个任务，最多十二条在途，尽早形成小批完整轨迹。

## 交付与启动

交付包 `E004-1.2_pilot_code_20261102.tar.gz` 仅包含新入口、对应记录与测试；在基点 `5a813cd4610e5d3e3e2e5350da4fab194aeb71a1` 的独立工作树解包。旧 E004-1.1 工作树与运行依赖保持不变。包不包含权重、数据或 Python 环境，不需要重新下载。

把交付包上传到 `/inspire/hdd/global_user/czxs25210022/why/` 后，在 CPU 终端创建新工作树（只执行一次）：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc
git worktree add --detach ../qihc-E004-1.2 5a813cd4610e5d3e3e2e5350da4fab194aeb71a1
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.2
tar -xzf /inspire/hdd/global_user/czxs25210022/why/E004-1.2_pilot_code_20261102.tar.gz
bash experiments/feedback_pilot_v1/00_prepare_cpu.sh
```

此交付不是已推送到远端分支的提交；只有 `git fetch` 不会获得新入口。基点提交与新增源码内容哈希共同固定实际代码，manifest 保存二者；已经准备后不再编辑代码。

旧大实验先通过平台正常暂停并保留目录，将四卡作业用于本小实验。单 worker 四卡 H100 80GB 启动/断点续跑使用同一条命令，不套 torchrun：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.2
bash experiments/feedback_pilot_v1/01_run_4gpu.sh
```

查看阶段状态、手动报告与结果打包（均可在另一 CPU 终端）：

```bash
cat /inspire/hdd/global_user/czxs25210022/why/results/E004-1.2_feedback_pilot/status.json
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.2
bash experiments/feedback_pilot_v1/02_report.sh
bash experiments/feedback_pilot_v1/03_pack.sh
```

报告位于 `results/E004-1.2_feedback_pilot/report.md` 和 `summary.json`；包输出到个人根目录，名称 `E004-1.2_analysis_<北京时间>.tar.gz`，同时生成 SHA256 文件。可在部分完成时打包。

可覆盖 `QIHC_BASE`、`QIHC_ENV_PREFIX`、`QIHC_FORMAT_DEPS`、`QIHC_SOLVER_DEPS`、`QIHC_E003_RUN`、`QIHC_MODEL`、`QIHC_PILOT_RUN`；准备与启动必须指向相同结果目录。入口拒绝 E004-1.1 的 manifest。

## 比较与预算

- C0：无反馈 p-bit，控制额外搜索预算；C1：纯自审+p-bit；U2：自适应反馈+p-bit；B2：相同反馈协议+SCIP。后端的能力差异仍照实记录，SCIP 不是热浴分布。
- 审阅 2048、补丁 4096 tokens，两轮共最多 12288 输出 tokens/轨迹；单请求输入仍 32768，累计输入 131072；反馈包 4096。每轮非 LLM 核心预算 20 秒，诊断单列。
- 有限逻辑深度四层、布尔运算合法元数、一个审阅假设、四个局部编辑，所有组一致。通用索引/源表含义提示与错误重建提示不读取 gold 或题型标签。
- 新 GPU worker 将 RecursionError 单独记为请求失败并保留未知 token 成本，避免按设备故障重启模型。有限深度 schema 限制逻辑递归；未提高 Python 全局递归上限。新 CPU 包装器丢弃空/失效 warm start 与旧域候选并记录丢弃数量。
- 没有更改 `qihc/` 或 `experiments/feedback_study_v1/`；新入口在自己的进程内绑定旧实验的调度/协议组件，复用核心生成与搜索，GPU 请求隔离和 CPU 空候选清理由新入口处理，不复制整个 runtime，也不修改共享核心模块全局变量。新源码和所依赖核心共同进入 fingerprint；没有改动旧实验源码哈希。

优先看 U2 对 C1、B2 和各个 E003 LLM 基线的同题胜平负及全分母最优命中数。E003 canonical 为正确模型参考；旧 LLM 基线只有 20 秒搜索，新方法有 40 秒和额外修订调用，不能称等总成本比较。只优于旧基线、却不优于 C1/B2，说明改进可能来自一般修订而非 p-bit 独有贡献。

18 题单种子没有足够覆盖度证明稳定领先，不输出一层一题的退化 bootstrap 窄区间。报告保留失败、未结束与费用差异；先看是否存在值得扩大验证的信号。运行时间由真实模型吞吐决定，216 次是调用次数上限，不承诺固定完成小时数。
