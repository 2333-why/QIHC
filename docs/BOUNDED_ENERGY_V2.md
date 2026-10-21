# 有界整数、逻辑编译、128 位分块搜索

## 已实现与边界

采用独立 `bounded-integer-v2` 接口；旧 `sparse-binary-v1`、旧编译器、旧实验和训练入口未修改。
新接口可用 `qihc.generic.bounded.prompt/parse` 接收 LLM 建模输出，用 `block_search.solve` 搜索。
本轮没有把旧 SFT/DPO 数据自动改成 v2，也没有将公开自然语言数据自动认定为已正确建模。

| 功能 | 实现 | 限制 |
|---|---|---|
| 有界整数 | x=lower+sum(weight*bit)，权重覆盖完整且仅覆盖指定整数区间；支持负下界、固定变量、平方目标和交叉项 | 最多 256 原始变量；边界、源系数绝对值≤10000；不猜未知边界、不接收连续变量 |
| 逻辑 | not/and/or/implies/iff 嵌套表达式，自动生成 gate 和线性约束，再转 QUBO | 叶子必须是 [0,1] 变量；不支持直接把任意数值不等式当作逻辑叶子；源约束总数≤512 |
| 分块 | 每轮活动块≤128 比特，块外状态固定，边界耦合进入偏置；轮换随机分块 | 主机仍存完整稠密矩阵；默认编译上限2048位，可在 Python API 中设到4096；不是无限规模支持 |
| 全局反馈 | 全局原始约束违反率与辅助约束残差共同调惩罚；完整解按原始可行性和原始目标归档 | 有限时间随机搜索不保证全局最优，可能返回不可行解并明确标记 |

逻辑根表示必须为真；互斥可写 `not(and(a,b))`，恰好选一可写线性等式。
整数条件约束例如 `production <= 7*open_line` 使用原始线性约束表达；不自动假定任意 big-M。
变换保留目标常数偏移；不等式使用精确有界松弛变量。
编译为数学上的等价 QUBO，当前数值实现采用 float64；极端系数的数值误差仍需独立评估。

## 分块正确性

使用 E(z)=z^T Q z+c、对称 Q。固定块外 y 后，活动块 B 的 QUBO 为：

`Q_local = Q_BB + diag(2 Q_BO y)`

块外常数为 `y^T Q_OO y+c`。因此跨块约束没有被删除。
`conditional_qubo` 可导出单个链的子块矩阵及常数；软件搜索使用等价的全局条件场计算。
不同链边界状态不同，硬件接入时必须分别生成偏置，不能把一条链的偏置广播给所有链。

每 `block_updates` 次单比特更新切换活动块；默认128，完整随机排列的所有块用完后重新分组。
`steps` 是每条链的总单比特更新次数，不是每块/每轮更新次数；不足一个遍历的预算不保证所有位都访问。
反馈打开/关闭共享相同的预算、初态随机流和75%进度固定惩罚强化日程，差别仅在自适应反馈。
均匀随机对照不使用能量反馈。反馈从编译行映射回其原始约束，同时检查完整解，不能只看子块可行性。

**尚未实现 FPGA 通信或 int8 量化。** 返回 backend 明确为 `software_float64_block_reference`。
128 位容量只是软件子块大小约束，不能据此报告真机时延、能耗或精度。

## 运行

在仓库根目录、已有 QIHC Python 环境中运行：

```bash
python -m qihc.generic.block_search tools/benchmarks/bounded_v2_example.json --output results/bounded_v2_example.json --steps 2048 --chains 8 --block-size 128
```

示例包含0..7整数产量、两个开关变量、产能条件及“开线必须配人”的逻辑蕴含。
参考最优为 `[7,1,1]`、目标16；这只是功能检查，不是论文性能证据。

运行两类问题、四组同预算对照、三个种子的集成检查：

```bash
python tools/benchmarks/check_bounded_v2.py --output results/bounded_v2_checks.json
```

对照：p-bit+反馈、p-bit无反馈、Metropolis+反馈、随机扫描。
第二类140比特跨块平衡约束仅验证跨块流程，不用于证明性能优势。
所有输出包含完整解、原始违反量、目标值、编译位数、更新预算及反馈轨迹。

测试：

```bash
python -m pytest tests/test_bounded_energy.py tests/test_generic_energy.py tests/test_semantic_training.py tests/test_public_benchmarks.py -q
```

覆盖整数完整区间、负下界/固定变量、原始域检查、目标常数/平方项、逻辑真值表、
辅助变量最小惩罚等价、跨块全部候选能量等价、不可行实例、反馈消融预算及140比特流程。

## 旧实验与续跑

虽然未修改旧算法，新文件位于 qihc/ 内，会改变旧流水线计算的全目录源码指纹。
**不要将本版直接覆盖到正在续跑的锁定实验目录，也不要绕过原有指纹校验。**
旧实验保持原提交/原代码目录；新功能使用独立 checkout/新实验目录。源码版本一致后再准备新的正式实验。
本轮未更改自动下载模型、环境配置、旧结果或 prepared manifest。
