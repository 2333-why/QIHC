# 实验 01：通用提示词效率与建模可靠性

- 状态：脚本及本地验证完成；未启动服务器真实模型实验。
- 日期：2026-10-26（Asia/Shanghai）。
- 原始基线：`5e053fead75ea6b1b45ca56bf4624362fe7d5521`。
- 本轮完整代码版本：以本轮交付提交和服务器 manifest 的代码指纹为准。
- 前一轮：[现有提示词审查](../20261026_modeling_prompt_baseline/record.md)。
- 入口及完整设置：[prompt_efficiency_v1](../../../experiments/prompt_efficiency_v1/README.md)。
- 固定参数：[config.json](../../../experiments/prompt_efficiency_v1/config.json)。
- 默认结果：`/inspire/hdd/global_user/czxs25210022/why/results/prompt_efficiency_v1/`。

## 研究问题

1. 精简接口指令与重复上下文，能否降低输入／输出及调用成本，同时保持或提高原题可行率？
2. 相同简化表示下，分步建模是否优于一次输出？
3. 通用量词、目标局部赋值与需求覆盖清单，是否改善分步建模？

## 设置

沿用已准备 NLCO 240 题与合成 60 题，5 组提示词，共 1500 个正式逻辑建模请求。
模型路径从旧 co_comparison_v1 的 manifest 读取，预期沿用 Qwen3-32B，不下载或升级权重。
不做后训练，不加入搜索反馈；每个预测模型同时供 CP-SAT 与 p-bit 评分。
每组／题：CP-SAT 5 秒单线程一次；p-bit 3 种子、16 链、每链 8192 候选评分上限。
原题与变量布局辅助共享，输出总预算和格式约束一致。实际 FLOPs、输入 token 与调用数不保证相等，全部记录。
正式服务器环境：1 worker、4 H100、80 核 CPU；推理两个双卡副本，每副本 batch 2；32 个 CPU 搜索进程。
thinking 关闭，确定性生成。实际 Torch/Transformers 版本在 CPU 预检和启动日志记录。

## 记录与验证

- 2026-10-26：从冻结的历史模块抽取通用建模、推理、检查点与独立评分组件；旧入口保持原位。
- 2026-10-26：新增精简与语义自查提示，建立五组配对与独立运行目录。
- 本地：完成共享运行编排抽取后，相关新旧回归组合 179 项通过；其中本轮实验及真实小模型接口专项 18 项通过。包括模拟模型输出 + 真实 CP-SAT／p-bit 搜索、子步骤重放、篡改拒绝、失败计入统计、受控子进程验收到正式阶段。
- 真实小模型：本地保存并重新加载随机初始化 GPT2，验证批量生成与实际 schema 过滤；仅为接口测试，不是求解能力评测。
- CPU tokenizer 预检：Qwen3-32B 真实 tokenizer、五种 schema、300 题 × 5 组共 1500 个首轮提示全部通过，最长 10949 tokens。
- GPU 根因：此前 v6 子进程异常尚缺详细日志。本轮已补主进程错误回显，但不宣称此前根因已修复。

## 运行命令

固定本轮 checkout 后：

```bash
bash experiments/prompt_efficiency_v1/00_prepare_cpu.sh
# 在四卡任务里执行；同配置重复此命令可续跑
bash experiments/prompt_efficiency_v1/01_run_4gpu.sh
# CPU 端可单独重新汇总
bash experiments/prompt_efficiency_v1/02_report.sh
```

## 结果与结论

仅首轮提示长度的本地实测（含 chat template；不是模型推理或求解结果）：

| 提示组 | 平均输入 tokens | 最大输入 tokens |
|---|---:|---:|
| legacy_flat | 3189.26 | 10949 |
| concise_flat | 2390.26 | 10150 |
| legacy_staged | 2409.18 | 10124 |
| concise_staged | 2072.26 | 9832 |
| semantic_staged | 2182.26 | 9942 |

精简 flat 首轮输入比原版减少约 25.1%；精简 staged 的 PLAN 首轮比原版减少约 14.0%。
后续步骤长度和总调用成本依赖真实模型输出，需要服务器实验测量，不能由首轮节省直接外推。

真实模型结果：待返回。暂不声称提示改进提高了可行率、速度或 p-bit 优势。
先按实例平均种子，再进行配对统计。失败计入分母，未完成单列。
CP-SAT 与 p-bit 停止预算不同，后端比较只报告实际质量／成本，不称为等算力实验。
NLCO 参考值不一律视为已证明最优；既有调试样本不视为全新盲测。

下一轮：根据本轮数据确定是否测试 thinking 或 few-shot；需独立建档，避免与本轮因素混淆。
