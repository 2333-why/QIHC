# E004_2.3 与 E005-1.8 独立作业

两个 GPU 作业各申请四张 H100 80GB 和24 CPU核；并发需要调度器分配两组不同的四卡。共用固定只读代码、已有 qihc-cu128 与 Qwen3-32B。默认历史输入为 results/E005-1.7_eight，要求服务器原始目录包含三个种子 sft/solver_rl/pvcd 的 adapter.pt 和完整的SFT预测；此前导出的分析包排除了适配器，不能替代原目录。QIHC_HISTORY 可覆盖历史位置。

一次在联网CPU运行：
```bash
bash experiments/selective_v1/00_cpu.sh
```
检查权重、公开题目/训练划分、历史适配器和预测指纹、SCIP、两个运行清单；完成标志 BOTH_JOBS_READY。历史预测缺失仅允许在后训练组从其适配器重新生成；缺SFT初稿直接报告文件名。新作业可读旧基线，写入不同结果目录。

两个离线GPU作业分别运行：
```bash
bash experiments/selective_v1/01_feedback_gpu.sh pilot && bash experiments/selective_v1/01_feedback_gpu.sh formal
```
```bash
bash experiments/selective_v1/02_posttrain_gpu.sh pilot && bash experiments/selective_v1/02_posttrain_gpu.sh formal
```

pilot为24验证题、种子17，两个新训练组使用12个训练题。formal为910道不同公开题，每组2180条。调度器并行按方法和种子分派，每卡一个子进程，同时最多8个CPU进程评分。新适配器每8个训练题和最后一步原子保存权重/优化器/步数；分类、反馈和评分逐题保存。重提相同命令续跑。formal模型独立于pilot。默认结果为 results/E004_2.3_selective 和 results/E005-1.8_paired，各含 pilot/formal 目录、logs、models、report.md。QIHC_NEW_RESULTS 指向两个新目录的父目录。

```bash
bash experiments/selective_v1/03_report.sh feedback formal
bash experiments/selective_v1/03_report.sh train formal
```

按阶段先在原题内平均种子；结果包含已评分轨迹、共同原题、编译失败恢复、完整正确模型退化。E004历史运行目录可用QIHC_E004_RUN指定。共享SFT基线是两份报告之间的桥梁。外部历史匹配时额外列出C0/P2同题比较。四卡真实吞吐、显存与实际训练信号以pilot日志为准。

正式报告也逐组读取E005-1.7的八组历史评价，与本实验各组按共同题和所需种子单独配对；例如旧P-VCD覆盖不齐不会缩小其他历史方法的比较范围。
