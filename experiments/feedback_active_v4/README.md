# E004_2.2

格式接口修复；复用v3运行组件，通过config.protocol_version=v2选择新协议。

联网端00_setup.sh，离线GPU端01_run.sh；默认新结果目录E004_2.2_active_feedback。先开发120题接口门槛，通过后自动继续，失败不进入主测。历史LocalModel、统一qihc-cu128、四卡各一路、无第二模型。

完整设置与启动记录：docs/experiments/E004_feedback/E004_2.2_record.md。
