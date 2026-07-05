# VCI 变分协推理指南

> **QIHC** = 平台 · **VCI** = 范式  
> 对应国自然方案总纲 v4 · 代码路径 `qihc/orchestrator/`

## 1. 核心方程

\[
\mathcal{F}(q,s) \approx \lambda_\text{KL}\,\|q-q_0\|^2 + \beta\,E_\text{Ising}(s|q) - H(q) + \lambda_v\,\#\text{violations}
\]

实现：`qihc/orchestrator/free_energy.py`

## 2. 协推理循环

| 步骤 | 模块 | 函数 |
|------|------|------|
| s-步 | `backend.py` | `PBitBackend.solve` |
| 评估 | `free_energy.py` | `compute_free_energy` |
| q-步 | `frontend/base.py` | `BaseFrontend.refine` |
| 调度 | `vci_scheduler.py` | `VCIOrchestrator.solve_subset` |

## 3. 运行模式

```python
from qihc.orchestrator import VCIConfig, VCIOrchestrator, demo_problem

cfg = VCIConfig.tier_a(sampling_steps=250, max_rounds=2)
orch = VCIOrchestrator(cfg)

for mode in ["greedy", "vci-1", "vci-2"]:
    r = orch.solve_subset(demo_problem(), mode=mode)
    print(mode, r.final_feasible, r.n_rounds, r.final_free_energy)
```

| 模式 | 轮次 | q-步 refine |
|------|------|-------------|
| greedy / vci-0 | 0 | 否 |
| vci-1 | 1 | 否（≈ CR） |
| vci-2 | ≤2 | 是 |
| vci-full | ≤4 | 是 + F 收敛 |

## 4. Case A 玩具实验

```bash
python experiments/run_vci_reasoning.py --problems 32 --steps 250
```

## 5. BBH mini-set（40 题）

```bash
python experiments/run_vci_bbh.py --budget-steps 250
```

| 方法 | 可行率 | 精确匹配 |
|------|--------|----------|
| greedy | 90% | 87.5% |
| vci-1 | 100% | 45% |
| vci-2 | 100% | 52.5% |

## 6. Handoff 探针

```bash
python experiments/run_vci_handoff.py --budget-steps 200
```

σ≥0.15 时 VCI-2 可行率增益约 +5~10%。

## 7. Case B TTS 标度

```bash
python experiments/run_sampler_scaling.py --nodes 12 14 16 18 20 --steps 800
```

产物：`experiments/outputs/sampler_scaling/tts_scaling.png`

## 6. 架构图

```bash
python experiments/plot_qihc_vci_architecture.py
```

输出：`docs/QIHC_VCI_architecture.png`

## 7. 扩展路线

1. `frontend.refine`：logits 重加权 → prompt 追加 feedback → routing head
2. Case A：BBH 子集替换 `generate_toy_problems`
3. handoff 判据：记录 \(\Delta F_q\) vs \(\Delta F_s\)
4. 同算力表：固定 `sampling_steps × max_rounds`

## 8. 测试

```bash
pytest tests/test_vci.py -v
```
