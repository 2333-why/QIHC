# QIHC v0.2.0 发布清单

## 版本信息

- **Tag**: `v0.2.0`
- **主题**: VCI 变分协推理 + Case A/B 实验脚本

## 发布前检查

- [x] `pytest` 全绿（21 tests）
- [x] `CHANGELOG.md` 更新
- [x] `pyproject.toml` version = 0.2.0
- [x] 实验可复现（见下方命令）
- [ ] `git push origin main && git push origin v0.2.0`（需本地执行）

## 一键复现实验

```bash
cd QIHC
pip install -e ".[dev]"

pytest
python experiments/run_vci_reasoning.py --problems 32 --steps 250
python experiments/run_vci_bbh.py --budget-steps 250
python experiments/run_vci_handoff.py --budget-steps 200
python experiments/run_sampler_scaling.py --nodes 12 14 16 18 20 --steps 800
python experiments/plot_qihc_vci_architecture.py
python experiments/run_moe_poc.py --tier a   # 快速 smoke
```

## 产物路径（gitignore，本地生成）

| 实验 | 输出 |
|------|------|
| Case A 玩具 | `experiments/outputs/vci_reasoning/` |
| BBH 40 题 | `experiments/outputs/vci_bbh/` |
| Handoff | `experiments/outputs/vci_handoff/` |
| TTS 标度 | `experiments/outputs/sampler_scaling/` |
| MoE | `experiments/outputs/moe_poc/` |
| 架构图 | `docs/QIHC_VCI_architecture.png` |

## Git 发布命令（在项目 QIHC/ 目录）

```bash
git add .
git commit -m "release: QIHC v0.2.0 with VCI co-inference and Case A/B experiments"
git tag -a v0.2.0 -m "VCI variational co-inference, BBH mini-set, handoff probe"
git push origin main
git push origin v0.2.0
```

## GitHub Release 说明（粘贴用）

**QIHC v0.2.0 — VCI Variational Co-Inference**

- VCI loop: `free_energy.py`, `VCIOrchestrator`, `frontend.refine()`
- Case A: toy + BBH 40-task mini-set (`run_vci_bbh.py`)
- Handoff/temperature probe (`run_vci_handoff.py`)
- TTS scaling benchmark (`run_sampler_scaling.py`)
- Architecture diagram + docs (`VCI_GUIDE.md`, `LITERATURE_GAP.md`)

See `CHANGELOG.md` for details.
