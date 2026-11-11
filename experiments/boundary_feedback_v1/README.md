# E004_2.6 原题赋值复核与主动 p-bit 边界反馈

实验记录：[E004_2.6](../../docs/experiments/E004_feedback/E004_2.6_record.md)。冻结原始 Qwen3-32B，无 SFT。沿用 E004_2.5 的全部 910 题、2180 条题目－种子输入及 B 组编译修复输出。六组正式共 **13080** 条评分，不改变已有作业。所有新入口在本目录，共享运行器在 `qihc/experiment_runtime/feedback_job.py`，历史运行器不修改。

## 六组比较

|组|方法|要回答的问题|
|---|---|---|
|A|原初稿|原始基线|
|B|复用上一轮仅修编译结果|有效基线，排除重新生成差异|
|C|共享候选，原题双复核选择，无采样|一般额外推理是否已经足够|
|S|SCIP多解池，赋值复核|精确求解方法找反例的对照|
|P|普通p-bit多链池，赋值复核|一般概率采样的对照|
|Q|温度、软惩罚和条件响应钳制的主动p-bit池|主动能量探测的增量价值|

候选允许改约束、逻辑、目标；固定变量名称和边界以保证赋值比较有效。所有组共用同一候选。S/P/Q不把模型目标假设当真值，判断调用只看原题、变量顺序和赋值，不看两份模型的判定或采样来源。第一轮须候选零冲突、原模型至少一个冲突；第二轮使用新的赋值与独立判断调用，候选出现冲突即回退。每轮至少两个已判断标签。判断仍是同一基座模型的推断，不是独立证明器；离线审计标签错误率，金标准不会反馈给采样或选模。

S/P/Q搜索预算上限均20秒（发现10、验证10），32条p-bit链；S每轮4个窗口、两份模型各2次，多解枚举及扰动目标；P每轮4个普通采样窗口；Q每轮8个窗口，基线、低惩罚高温、条件响应差最大的变量低/高值钳制。选择和能量响应计算计入墙钟预算，内核初始化可能有少量超时，保存实际秒数。C及S/P/Q最多共用1次候选生成＋2次复核，输出上限4096 tokens/次，早退时实际用量较少。按实际成本报告，不能称相同总算力。

## 联网 CPU 准备

先按交付消息用 Git 获取独立工作目录和固定提交。已完成 E004_2.5 的原目录必须可读；不要只保留 report.md。

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
export QIHC_ENV="$BASE/envs/qihc-cu128"
export QIHC_PREVIOUS_RUN="$BASE/results/E004_2.5_base_optimal_feedback"
export QIHC_BOUNDARY_RUN="$BASE/results/E004_2.6_boundary_feedback_fix1"
cd "$BASE/qihc-E004_2.6-fix1-git"
bash experiments/boundary_feedback_v1/00_cpu.sh
```

成功标记 `E004_2.6_CPU_READY`。检查已安装 CPU/GPU 依赖、公开数据版本、上一轮输入/B组预测完整性、原模型所有分片与分词器及SCIP。复用已有共享环境，不安装或升级依赖。缺失依赖会明确报错，需在联网端补齐后再准备；本迭代未新增依赖。输入、代码、模型、数据均写入不可变 manifest。

## 离线四卡 GPU 启动及续跑

申请4×H100 80GB、24 CPU，启动命令如下。环境变量在每个新任务中重新设置。

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
export QIHC_ENV="$BASE/envs/qihc-cu128"
export QIHC_BOUNDARY_RUN="$BASE/results/E004_2.6_boundary_feedback_fix1"
export QIHC_PROBE_LANES=2
export QIHC_SCORE_WORKERS=8
cd "$BASE/qihc-E004_2.6-fix1-git"
bash experiments/boundary_feedback_v1/03_start.sh
```

先运行开发集固定24题pilot，完成144条评分后检查接口故障，再自动进入正式实验。若超过20%的实际调用发生上下文不足/输出截断，或S/P/Q任一组超过6/24条候选/判断解析失败，则停止并提示检查原文。不按效果优劣挑选方法，不要求pilot必须显著提升。正式结果目录独立于pilot。

重复同一命令续跑会跳过已有产物。通过pilot后也可直接 `bash experiments/boundary_feedback_v1/01_gpu.sh formal`。GPU入口没有pip、Git或下载操作，使用本地模型和离线模式。四个常驻单卡模型，每卡两条CPU探针/推理流水线，同卡生成串行，另有8个CPU评分进程。保留Transformer LocalModel历史推理方式，没有vLLM或新算子编译。

## 部分结果与日志

```bash
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_BOUNDARY_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E004_2.6_boundary_feedback_fix1
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004_2.6-fix1-git
bash experiments/boundary_feedback_v1/02_report.sh formal
tail -n 40 "$QIHC_BOUNDARY_RUN/formal/logs/worker0.log"
```

`formal/report.md` 和 `summary.json` 包含计划/完成分母、正确/可行/最优、对B新增/丢失、动作细分、判断标签离线错误率、配对置信区间。`formal/shared` 保存共享候选和原始调用，`formal/models/*/*/predictions` 保存赋值、判断原文与探针统计，`evaluations` 是评分器独立产物。`formal/logs/gpu.csv`记录可获取的GPU利用率和显存，`timing`记录生成服务时间。新实验未在H100实测，运行时长和利用率待pilot返回。

## 2026-11-12 接口修复 fix1

分支codex/e004-boundary-fix1。赋值以x0…、赋值对以p0…分别编号，给出完整null模板，严格检查键集合和标签类型。解析器仅为六字段裸模型补model外层，不改变数学内容。未知标签不补猜，非原文引用仍拒绝。重新运行pilot与正式阶段，禁止复用失败pilot中的判断或放宽gate。新结果目录使用fix1后缀；原题、种子、搜索与生成预算不变。
