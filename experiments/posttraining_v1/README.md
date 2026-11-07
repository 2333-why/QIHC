# E005-1.4 四种后训练方法

本实验训练 Qwen3-32B 的四种适配器，使用 E004_2.2 的公开数据、提示、格式解析和终端评分。根目录与运行目录均独立。以下命令在服务器新工作树根目录执行。

## 联网 CPU

```bash
export QIHC_ENV=/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128
bash experiments/posttraining_v1/00_cpu.sh
```

看到 E005_CPU_READY 即可提交 GPU 作业。准备程序使用 E004 已有依赖，校验 Git 中的数据、真实本地分词器、权重 SHA256、SCIP 调用和训练标签往返解析。它只读共享环境，不装包。缺依赖时会打印具体包名。模型、分词器和数据均从本地加载。CPU/GPU 默认运行根为 /inspire/hdd/global_user/czxs25210022/why/results/E005-1.4_four。

## 四卡 GPU

在调度器单独分配的 4×H100 80GB、24 CPU 核节点运行。反馈作业仍占用同四张卡时，提交排队任务。建议第一次用真实模型先跑 pilot。
```bash
bash experiments/posttraining_v1/01_gpu.sh pilot
```

pilot 用 12 个训练题、1 个种子，评价 24 道开发验证题。formal 从头训练，各自保存检查点。
```bash
bash experiments/posttraining_v1/01_gpu.sh formal
```

同一命令续跑，保留原结果目录和模型。每8个训练题保存 LoRA、优化器和步数，重启最多重复7个已完成步骤。逐题生成种子和训练题顺序固定。GPU 任务不需要网络。每卡一次运行一个方法/种子，SFT 适配器可用后启动其他方法，CPU 同时评分。日志在 pilot/logs 或 formal/logs。

## 报告

```bash
bash experiments/posttraining_v1/02_report.sh formal
```

可覆盖 QIHC_E004_RUN 指向 E004_2.2 原始运行目录。脚本只读取其 manifest 和已落盘的评价，不使用 report.md 的不同分母做横向排名。训练种子17/29/43，C0为每题一个结果，P2和各训练方法按种子在题内平均。部分报告同时给出已评价轨迹数、已配齐种子的题数、严格共同题数。结果见 formal/report.md、paired_cases.json 和 models/<种子>/<方法>/evaluations。training_signals.json 给出实际带奖励更新的组数、产生证据的候选数和已蒸馏修复数。若修复数为零，结果需标明蒸馏项没有获得有效修复。

[实验记录](../../docs/experiments/E005_posttraining/E005-1.4_record.md)
