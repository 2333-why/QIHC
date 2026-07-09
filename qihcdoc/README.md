# QIHC 文档索引（qihcdoc）

| 文档 | 说明 |
|------|------|
| [`双4090逐步运行手册.md`](./双4090逐步运行手册.md) | **2×4090 联网**：环境、清华源、ModelScope、逐步命令、排错 |
| [`Pro6000单卡2小时快速实验.md`](./Pro6000单卡2小时快速实验.md) | **1×Pro 6000 约 2h**：单卡冒烟 + 7B 主表 |
| [`双机实验流程_4090与H200.md`](./双机实验流程_4090与H200.md) | 4090 下载打包 → 离线 H200 全量实验 |

代码入口：

- `experiments/nsfc_evidence/run_dual_4090.sh` — 双 4090 全量
- `experiments/nsfc_evidence/run_dual_pro6000.sh` — 双 Pro 6000 全量
- `experiments/nsfc_evidence/run_pro6000_quick_2h.sh` — **单 Pro 6000 约 2h 快速**
- `experiments/nsfc_evidence/run_online_4090.sh` — 单 4090 下载 + 轻量
- `experiments/nsfc_evidence/run_offline_h200.sh` — 4×H200 离线批量
- `experiments/nsfc_evidence/cn_mirror_env.sh` — 清华 PyPI + HF 镜像
- `experiments/nsfc_evidence/download_model_hf.py` — 模型下载（ModelScope / HF）

**环境问题**：`CUDA: False` / `driver too old` → [双4090手册 §4.1](./双4090逐步运行手册.md#41-pytorch-与驱动不匹配cuda-false--driver-too-old)
