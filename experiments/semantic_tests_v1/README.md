# E004_2.7 通用主动语义测试反馈

[完整实验记录](../../docs/experiments/E004_feedback/E004_2.7_record.md)。原始Qwen3-32B，不使用SFT；四张H10080GB、24 CPU。代码位于独立工作树，结果默认保存到 `results/E004_2.7_semantic_tests`。复用 E004_2.6 fix1 的初稿、B组编译修复及C组双复核结果，禁止读取它们的评分来选择在线动作。

## 方案与对照

|组|流程|目的|
|---|---|---|
|A|冻结原初稿|历史起点|
|B|冻结仅修编译结果|基本修复基线|
|C|冻结上一轮双复核结果|本轮增强方法的共同起点|
|R|新假设＋无测试局部双复核|额外LLM调用与局部修订对照|
|S|SCIP多解测试＋局部语义双复核|精确搜索提供测试点|
|P|普通p-bit测试＋同一复核|普通采样提供测试点|
|Q|主动p-bit测试＋同一复核|温度、惩罚和条件钳制的组合价值|

共享一次原题审阅产生完整候选，再分解为目标、算术约束、逻辑约束三个可编译的单块假设。变量顺序和域固定。S/P/Q最多用10秒在全部假设上寻找差异，按可行性差异比例及最多32点的排序差异比例选择一个假设；两项比例相加，不按样本数量评分。Q使用既有通用p-bit条件能量响应、温度与软惩罚、变量钳制；S多解枚举与扰动目标；P普通多链热浴。S/P/Q采用相同筛选器，均不读取任务家族、gold或历史结果标签。

解释器计算一个可行性冲突或一对目标排序冲突，并给出两个解释下的结果。LLM只回答一个语义选择 `choice=0/1/null`，同时引用1..3个原文句子编号并简述原因，不再输出8个可行标签和28个排序标签。程序验证编号与格式，不把引用存在当成语义正确。第一次选择修改后，再用10秒查找不同赋值复测；展示左右顺序反转，再次支持修改才接受。没有新测试点、弃权、格式失败或保留意见均回退。

R按固定种子选择同一共享集合中的一个假设，进行两次无测试复核；它不做采样选假设。因此Q−R是“主动选假设＋测试证据＋复测”的整体比较，不能单独归因为证据文本。Q−P和Q−S控制上述后续流程，用于比较探针方式。

方法不含背包、图或其他题型的专用修补规则，不直接追加提取的线性约束。通用性限于已有有界整数/布尔IR所能表达的问题：线性/二次目标、线性约束和布尔逻辑。变量域修复、任意连续与非线性问题不在本次范围。LLM仍可能选错；离线评分独立核对每次选择与原题在该测试点的计算是否一致，并统计最终可行/最优、语义退化。局部一致不代表整模型已证正确。

## CPU联网准备

通过Git取固定提交后，在新目录执行。上一轮完整 `E004_2.6_boundary_feedback_fix1` 必须保留，至少包含manifest、inputs和全部B/C预测。

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
export QIHC_ENV="$BASE/envs/qihc-cu128"
export QIHC_PREVIOUS_RUN="$BASE/results/E004_2.6_boundary_feedback_fix1"
export QIHC_SEMANTIC_RUN="$BASE/results/E004_2.7_semantic_tests"
cd "$BASE/qihc-E004_2.7-git"
bash experiments/semantic_tests_v1/00_cpu.sh
```

成功标记 `E004_2.7_CPU_READY`。检查共享环境现有CPU/GPU依赖、本地模型/分词器、数据版本、原模型身份和历史预测完整性，冻结源码与输入哈希。没有新增依赖或自动升级，不创建环境。

## 离线四卡GPU一键启动与续跑

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
export QIHC_ENV="$BASE/envs/qihc-cu128"
export QIHC_SEMANTIC_RUN="$BASE/results/E004_2.7_semantic_tests"
export QIHC_PROBE_LANES=2
export QIHC_SCORE_WORKERS=8
cd "$BASE/qihc-E004_2.7-git"
bash experiments/semantic_tests_v1/03_start.sh
```

固定24开发题pilot共168条评分，通过接口检查后自动进入正式15260条评分。检查采用“实际候选/语义调用次数”作为分母，解析失败超过20%时停止；不按效果好坏筛选方法，不把合法弃权当接口错误。报告同时列弃权率，避免把流程成功误认作语义能力可靠。GPU节点无Git、pip或下载。历史Transformers LocalModel推理，四个单卡模型、每卡两条CPU/推理流水线、同卡生成互斥，另有8个CPU评分进程。

重复启动命令续跑，跳过已完成产物。也可单独执行 `01_gpu.sh pilot` 或 `01_gpu.sh formal`，正式运行前应先通过pilot检查。不同协议/输入必须用新结果目录，不能覆盖或混用旧结果。

## 查看部分结果

```bash
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
export QIHC_SEMANTIC_RUN=/inspire/hdd/global_user/czxs25210022/why/results/E004_2.7_semantic_tests
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004_2.7-git
bash experiments/semantic_tests_v1/02_report.sh formal
# 尚在pilot时改为：bash experiments/semantic_tests_v1/02_report.sh pilot
```

每组计划2180：开发120×1、主测480×3、规模300×2、公共10×2。种子17/29/43沿用历史。终端p-bit/SCIP均20秒；A/B另做40秒评分。增强组最多一次共享假设生成＋两次局部复核，输出上限4096 tokens/次；无额外语义重试。实际搜索耗时、tokens、接口分母、块选择、弃权/接受、GPU利用率和完整原文都保留。硬件吞吐和效果尚待服务器pilot返回。

`formal/report.md`和`summary.json`是报告；`shared`保存候选，`models/*/*/predictions`保存测试点与复核原文，`evaluations`是独立评分，`logs/worker*.log`和`logs/gpu.csv`用于排错与利用率检查。报告比较Q−C、Q−R、Q−S、Q−P，以独立题为单位平均种子，5000次bootstrap及符号置换，阶段内Holm校正。复用C的既有成本与本轮增量成本区分记录，不能声称与A/B端到端等算力。
