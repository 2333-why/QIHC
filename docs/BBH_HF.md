# BBH Case A · Hugging Face 指南

## 数据源

| 来源 | 命令 | 说明 |
|------|------|------|
| bundled | `--source bundled` | 本地 40 题合成集（含互斥约束） |
| **HF 真实 BBH** | `--source hf` | `Joschka/big_bench_hard`（失败回退 `lukaemon/bbh`） |

默认 10 个子任务 × 50 题/任务 = **500 题**。

## 安装

```bash
pip install -e ".[hf]"          # 仅 HF 数据
pip install -e ".[hf,llm]"      # HF + DistilGPT-2 真实 logits
```

## 快速开始

```bash
# 1. 下载并缓存
python experiments/download_bbh_hf.py --limit-per-task 50

# 2. 伪 logits（快速，无 GPU）
python experiments/run_vci_bbh.py --source hf --budget-steps 200

# 3. 真实 LLM logits（DistilGPT-2 答案似然打分）
python experiments/run_vci_bbh.py --source hf --logits llm --limit 50 --budget-steps 200
```

## 输出路径

| 配置 | 目录 |
|------|------|
| bundled | `experiments/outputs/vci_bbh/` |
| HF + pseudo | `experiments/outputs/vci_bbh_hf/` |
| HF + LLM | `experiments/outputs/vci_bbh_hf_llm/` |

## 全量 500 题结果（pseudo logits, budget=200）

| 方法 | 可行率 | 精确匹配 |
|------|--------|----------|
| greedy | 100% | 95.2% |
| vci-1 | 100% | 92.8% |
| vci-2 | 100% | 92.0% |

## DistilGPT-2 真实 logits（n=50, budget=200）

| 方法 | 可行率 | 精确匹配 |
|------|--------|----------|
| greedy | 100% | 34.0% |
| vci-1 | 100% | 34.0% |
| vci-2 | 100% | 32.0% |

> 单选题 top_k=1；伪 logits 对 gold 有偏置，贪心已很强。`--logits llm` 用于真实语义势场；小模型下精确匹配整体偏低，协推理优势主要体现在约束合成集与高噪声 handoff。

## 代码入口

- `qihc/orchestrator/bbh_parser.py` — 解析 HF 行
- `qihc/orchestrator/bbh_hf.py` — HF 加载与缓存
- `qihc/orchestrator/bbh_llm.py` —因果 LM 选项打分
- `experiments/run_vci_bbh.py` — 实验脚本

## 缓存

`qihc/data/bbh_hf_cache.json`（gitignore，可离线复用）

```bash
python experiments/download_bbh_hf.py --refresh-cache
```
