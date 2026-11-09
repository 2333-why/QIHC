# E004_2.2 实验报告

状态：complete；推理 12980/12980，评价 12980/12980。
范围仅含已登记阶段；缺失、unknown、失败不删除。以下为轨迹计数；正式检验以题为单位平均种子。

所有正式题目来自公开数据：NLCO 六任务 900 题分开发/主测/规模；CP-Bench 10 题补充。第二模型关闭，人工审计另行记录。

| 阶段 | 方法 | 推理完成/计划 | 已评价 | 可执行 | 完整正确 | 已证错误 | unknown | 初始失败恢复 | 正确退化 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ablation | P0 | 240/240 | 240 | 225 | 58 | 163 | 4 | 33 | 4 |
| ablation | P1 | 240/240 | 240 | 226 | 59 | 163 | 4 | 34 | 4 |
| ablation | PR | 240/240 | 240 | 225 | 57 | 164 | 4 | 33 | 4 |
| development | C0 | 120/120 | 120 | 94 | 34 | 58 | 2 | 0 | 0 |
| development | D | 120/120 | 120 | 111 | 31 | 78 | 2 | 17 | 8 |
| development | M | 120/120 | 120 | 111 | 29 | 80 | 2 | 17 | 9 |
| development | P2 | 120/120 | 120 | 111 | 30 | 79 | 2 | 17 | 8 |
| development | R | 120/120 | 120 | 111 | 31 | 78 | 2 | 17 | 8 |
| development | S | 120/120 | 120 | 111 | 34 | 76 | 1 | 17 | 5 |
| development | X | 120/120 | 120 | 112 | 33 | 77 | 2 | 18 | 6 |
| main | C0 | 480/480 | 480 | 370 | 128 | 242 | 0 | 0 | 0 |
| main | D | 1440/1440 | 1440 | 1329 | 357 | 955 | 17 | 219 | 41 |
| main | M | 1440/1440 | 1440 | 1326 | 357 | 951 | 18 | 216 | 43 |
| main | P2 | 1440/1440 | 1440 | 1326 | 361 | 947 | 18 | 216 | 41 |
| main | R | 1440/1440 | 1440 | 1325 | 349 | 954 | 22 | 215 | 42 |
| main | S | 1440/1440 | 1440 | 1327 | 352 | 959 | 16 | 217 | 42 |
| main | X | 1440/1440 | 1440 | 1328 | 357 | 958 | 13 | 218 | 48 |
| public | B | 20/20 | 20 | 20 | 4 | 6 | 10 | 0 | 0 |
| public | C0 | 20/20 | 20 | 20 | 4 | 8 | 8 | 0 | 0 |
| public | P2 | 20/20 | 20 | 20 | 4 | 7 | 9 | 0 | 0 |
| public | R | 20/20 | 20 | 18 | 4 | 6 | 8 | 0 | 0 |
| replay | replay_P2 | 60/60 | 60 | 57 | 18 | 39 | 0 | 8 | 4 |
| replay | replay_R | 60/60 | 60 | 57 | 19 | 38 | 0 | 8 | 3 |
| scale | C0 | 300/300 | 300 | 179 | 63 | 108 | 8 | 0 | 0 |
| scale | P2 | 600/600 | 600 | 491 | 110 | 355 | 26 | 133 | 21 |
| scale | R | 600/600 | 600 | 492 | 89 | 362 | 41 | 134 | 23 |
| scale | X | 600/600 | 600 | 491 | 113 | 353 | 25 | 133 | 19 |

| 阶段 | 反馈 | 终端求解器 | 已评价/计划 | 原题可行 | 已证最优命中 | 无效输入 | 覆盖范围 |
|---|---|---|---:|---:|---:|---:|---|
| ablation | P0 | scip | 240/240 | 142 | 75 | 15 | full_cohort |
| ablation | P0 | pbit | 240/240 | 143 | 73 | 15 | full_cohort |
| ablation | P1 | scip | 240/240 | 143 | 78 | 14 | full_cohort |
| ablation | P1 | pbit | 240/240 | 141 | 74 | 14 | full_cohort |
| ablation | PR | scip | 240/240 | 140 | 76 | 15 | full_cohort |
| ablation | PR | pbit | 240/240 | 138 | 72 | 15 | full_cohort |
| development | C0 | scip | 120/120 | 71 | 42 | 26 | full_cohort |
| development | C0 | pbit | 120/120 | 69 | 40 | 26 | full_cohort |
| development | D | scip | 120/120 | 74 | 35 | 9 | full_cohort |
| development | D | pbit | 120/120 | 75 | 35 | 9 | full_cohort |
| development | M | scip | 120/120 | 74 | 35 | 9 | full_cohort |
| development | M | pbit | 120/120 | 75 | 35 | 9 | full_cohort |
| development | P2 | scip | 120/120 | 73 | 35 | 9 | full_cohort |
| development | P2 | pbit | 120/120 | 74 | 35 | 9 | full_cohort |
| development | R | scip | 120/120 | 76 | 37 | 9 | full_cohort |
| development | R | pbit | 120/120 | 76 | 36 | 9 | full_cohort |
| development | S | scip | 120/120 | 73 | 38 | 9 | full_cohort |
| development | S | pbit | 120/120 | 73 | 38 | 9 | full_cohort |
| development | X | scip | 120/120 | 74 | 39 | 8 | full_cohort |
| development | X | pbit | 120/120 | 78 | 39 | 8 | full_cohort |
| main | C0 | scip | 480/480 | 248 | 143 | 110 | full_cohort |
| main | C0 | pbit | 480/480 | 250 | 141 | 110 | full_cohort |
| main | C0 | cp_sat | 480/480 | 247 | 143 | 110 | full_cohort |
| main | C0 | highs | 480/480 | 244 | 140 | 110 | full_cohort |
| main | C0 | sa | 480/480 | 248 | 141 | 110 | full_cohort |
| main | C0 | pt | 480/480 | 248 | 141 | 110 | full_cohort |
| main | C0 | penalty_pt | 480/480 | 246 | 140 | 110 | full_cohort |
| main | D | scip | 1440/1440 | 831 | 433 | 111 | full_cohort |
| main | D | pbit | 1440/1440 | 834 | 421 | 111 | full_cohort |
| main | D | cp_sat | 360/360 | 218 | 114 | 31 | prespecified_120_case_subset |
| main | D | highs | 360/360 | 215 | 112 | 31 | prespecified_120_case_subset |
| main | D | sa | 360/360 | 215 | 112 | 31 | prespecified_120_case_subset |
| main | D | pt | 360/360 | 209 | 106 | 31 | prespecified_120_case_subset |
| main | D | penalty_pt | 360/360 | 214 | 111 | 31 | prespecified_120_case_subset |
| main | M | scip | 1440/1440 | 838 | 432 | 114 | full_cohort |
| main | M | pbit | 1440/1440 | 837 | 423 | 114 | full_cohort |
| main | M | cp_sat | 360/360 | 209 | 111 | 31 | prespecified_120_case_subset |
| main | M | highs | 360/360 | 205 | 108 | 31 | prespecified_120_case_subset |
| main | M | sa | 360/360 | 203 | 107 | 31 | prespecified_120_case_subset |
| main | M | pt | 360/360 | 198 | 102 | 31 | prespecified_120_case_subset |
| main | M | penalty_pt | 360/360 | 203 | 105 | 31 | prespecified_120_case_subset |
| main | P2 | scip | 1440/1440 | 836 | 430 | 114 | full_cohort |
| main | P2 | pbit | 1440/1440 | 842 | 422 | 114 | full_cohort |
| main | P2 | cp_sat | 360/360 | 211 | 113 | 31 | prespecified_120_case_subset |
| main | P2 | highs | 360/360 | 207 | 110 | 31 | prespecified_120_case_subset |
| main | P2 | sa | 360/360 | 205 | 109 | 31 | prespecified_120_case_subset |
| main | P2 | pt | 360/360 | 200 | 104 | 31 | prespecified_120_case_subset |
| main | P2 | penalty_pt | 360/360 | 204 | 107 | 31 | prespecified_120_case_subset |
| main | R | scip | 1440/1440 | 829 | 426 | 115 | full_cohort |
| main | R | pbit | 1440/1440 | 841 | 423 | 115 | full_cohort |
| main | R | cp_sat | 360/360 | 210 | 111 | 31 | prespecified_120_case_subset |
| main | R | highs | 360/360 | 209 | 111 | 31 | prespecified_120_case_subset |
| main | R | sa | 360/360 | 210 | 112 | 31 | prespecified_120_case_subset |
| main | R | pt | 360/360 | 203 | 105 | 31 | prespecified_120_case_subset |
| main | R | penalty_pt | 360/360 | 209 | 110 | 31 | prespecified_120_case_subset |
| main | S | scip | 1440/1440 | 833 | 423 | 113 | full_cohort |
| main | S | pbit | 1440/1440 | 840 | 418 | 113 | full_cohort |
| main | S | cp_sat | 360/360 | 211 | 113 | 31 | prespecified_120_case_subset |
| main | S | highs | 360/360 | 210 | 112 | 31 | prespecified_120_case_subset |
| main | S | sa | 360/360 | 213 | 115 | 31 | prespecified_120_case_subset |
| main | S | pt | 360/360 | 204 | 106 | 31 | prespecified_120_case_subset |
| main | S | penalty_pt | 360/360 | 208 | 110 | 31 | prespecified_120_case_subset |
| main | X | scip | 1440/1440 | 839 | 426 | 112 | full_cohort |
| main | X | pbit | 1440/1440 | 848 | 427 | 112 | full_cohort |
| main | X | cp_sat | 360/360 | 208 | 110 | 31 | prespecified_120_case_subset |
| main | X | highs | 360/360 | 209 | 111 | 31 | prespecified_120_case_subset |
| main | X | sa | 360/360 | 210 | 110 | 31 | prespecified_120_case_subset |
| main | X | pt | 360/360 | 207 | 107 | 31 | prespecified_120_case_subset |
| main | X | penalty_pt | 360/360 | 208 | 110 | 31 | prespecified_120_case_subset |
| public | B | scip | 20/20 | 10 | 10 | 0 | full_cohort |
| public | B | pbit | 20/20 | 8 | 6 | 0 | full_cohort |
| public | C0 | scip | 20/20 | 8 | 8 | 0 | full_cohort |
| public | C0 | pbit | 20/20 | 6 | 6 | 0 | full_cohort |
| public | P2 | scip | 20/20 | 9 | 9 | 0 | full_cohort |
| public | P2 | pbit | 20/20 | 8 | 6 | 0 | full_cohort |
| public | R | scip | 20/20 | 10 | 10 | 2 | full_cohort |
| public | R | pbit | 20/20 | 8 | 6 | 2 | full_cohort |
| replay | replay_P2 | scip | 60/60 | 36 | 21 | 3 | full_cohort |
| replay | replay_P2 | pbit | 60/60 | 35 | 20 | 3 | full_cohort |
| replay | replay_R | scip | 60/60 | 37 | 22 | 3 | full_cohort |
| replay | replay_R | pbit | 60/60 | 37 | 21 | 3 | full_cohort |
| scale | C0 | scip | 300/300 | 121 | 72 | 121 | full_cohort |
| scale | C0 | pbit | 300/300 | 124 | 71 | 121 | full_cohort |
| scale | P2 | scip | 600/600 | 297 | 137 | 109 | full_cohort |
| scale | P2 | pbit | 600/600 | 301 | 132 | 109 | full_cohort |
| scale | R | scip | 600/600 | 300 | 132 | 108 | full_cohort |
| scale | R | pbit | 600/600 | 300 | 127 | 108 | full_cohort |
| scale | X | scip | 600/600 | 302 | 138 | 109 | full_cohort |
| scale | X | pbit | 600/600 | 302 | 134 | 109 | full_cohort |

固定模型对照：各后端使用完全相同的初稿 M0 或标准模型 GOLD；40 秒为独立重跑。

| 模型/预算 | 求解器 | 已评价/计划 | 原题可行 | 已证最优命中 | 无效输入 |
|---|---|---:|---:|---:|---:|
| GOLD_20s | cp_sat | 480/480 | 480 | 480 | 0 |
| GOLD_20s | highs | 480/480 | 480 | 480 | 0 |
| GOLD_20s | pbit | 480/480 | 480 | 480 | 0 |
| GOLD_20s | penalty_pt | 480/480 | 480 | 480 | 0 |
| GOLD_20s | pt | 480/480 | 480 | 480 | 0 |
| GOLD_20s | sa | 480/480 | 480 | 480 | 0 |
| GOLD_20s | scip | 480/480 | 480 | 480 | 0 |
| M0_20s | cp_sat | 480/480 | 247 | 143 | 110 |
| M0_20s | highs | 480/480 | 244 | 140 | 110 |
| M0_20s | pbit | 480/480 | 250 | 141 | 110 |
| M0_20s | penalty_pt | 480/480 | 246 | 140 | 110 |
| M0_20s | pt | 480/480 | 248 | 141 | 110 |
| M0_20s | sa | 480/480 | 248 | 141 | 110 |
| M0_20s | scip | 480/480 | 248 | 143 | 110 |
| M0_40s_independent | cp_sat | 480/480 | 247 | 143 | 110 |
| M0_40s_independent | highs | 480/480 | 244 | 140 | 110 |
| M0_40s_independent | pbit | 480/480 | 250 | 141 | 110 |
| M0_40s_independent | penalty_pt | 480/480 | 246 | 140 | 110 |
| M0_40s_independent | pt | 480/480 | 248 | 141 | 110 |
| M0_40s_independent | sa | 480/480 | 248 | 141 | 110 |
| M0_40s_independent | scip | 480/480 | 248 | 143 | 110 |

最优命中只使用已证明参考；完整值及检查点在 evaluation/trajectories/*.json 的 cross/score；全部模型 hash 可追溯。

LLM 调用尝试：26060；输入 token：86486994；输出 token：15582199。
