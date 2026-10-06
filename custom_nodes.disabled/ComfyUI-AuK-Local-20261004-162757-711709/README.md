# AuK 本地语音风格转换

在 ComfyUI 中搜索 `AuK`。连接：Load Audio → AuK 语音风格转换 → Save Audio。

- 在节点里编辑提示词；默认 64 步、CFG 2.0、种子 42，种子行为设为 fixed。
- 整段音频自动在低音量处分段、逐段转换、拼接。输入内容不截掉；模型生成可能改变发音，跨段音色一致性仍需试听。
- 运行结束后子进程退出并释放模型。取消 ComfyUI 执行会终止转换。
- 只读取已有本地权重；不安装包、不下载模型、不调用云端服务。
- 后台文件和日志保存在 ComfyUI `output/auk/runs/`；Save Audio 另保存可试听的 FLAC。
- 使用 `/home/yu/.venv/auk/bin/python` 和 `/home/yu/Workspace/AuK`。迁移时可设置 `AUK_PYTHON`、`AUK_ROOT`。
- 这是调用已验证独立环境的自定义节点，不是把 AuK 模型直接加载进 ComfyUI 进程。执行时会通过 ComfyUI 的模型管理器卸载闲置模型，以腾出 GPU 显存。

模型检查、分段及生成逻辑来自本项目的 AuK 测试入口；本节点固定自己的 runner 副本，后续更改测试脚本默认提示词不会改动节点。实际提示词来自节点输入，并记录在每次运行的 `result/result.json`。
