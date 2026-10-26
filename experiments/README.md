# 独立实验入口

新实验采用 `experiments/<实验名>_vN/` 独立目录，包含配置、CPU 准备、GPU 运行、报告入口及说明。

- [prompt_efficiency_v1](prompt_efficiency_v1/README.md)：五组提示词，固定编译与求解后端，评估建模质量和实际成本。

通用建模位于 `qihc/modeling/`，共享实验基础组件位于 `qihc/experiment_runtime/`，
搜索算法继续位于 `qihc/generic/`。实验设置不得通过修改这些组件的模块全局变量临时注入。
每次改变共享代码都必须固定新的代码提交并使用独立结果目录，禁止混用旧检查点。

已有平铺脚本和 `tools/runtime/modeling_v2` 至 `modeling_v6` 作为历史可复现入口保留，
本次不迁移或删除，避免影响运行中实验及其源代码校验。
实验文档和时间线统一保存于 [docs/experiments](../docs/experiments/README.md)。
