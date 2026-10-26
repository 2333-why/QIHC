# 建模实验共享运行组件

`runner.py` 管理四卡分组、常驻模型副本、CPU 进程池、验收到正式阶段、失败日志回显、检查点与运行时间线。
入口显式传入实验定义模块，不导入任何具体实验目录，不修改模块全局配置来切换实验。

实验定义提供 NAME、RUN_ENV、DEFAULT_SOURCE、ENTRY_MODULE，以及：
`validate`、`prepare_runtime`、`request`、`tasks`、`evaluate_one`、`summarize`。
当前 runner 面向“两种后端评分、先 gate 后 formal”的建模实验；不同流程应扩展接口并增加测试。

`storage.py`：原子 JSON、指纹、文件锁。`identity.py`：本地权重元数据指纹。
`evaluation.py`：只在 CPU 评分端使用的独立原题校验。
实验特有组别、提示组合、预算与配对报告留在对应 `experiments/<name>/`。
