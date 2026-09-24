# Qwen-Audio 语音推理与评测优化

基于官方 Qwen-Audio-Chat 与 Silero VAD 的可追溯评测工程，研究提示词约束和整段语音门控。

**状态：工程实现、评分测试和真实 Silero VAD 实验已运行；Qwen-Audio-Chat 权重推理尚未成功，未报告提示词收益、WER 改善或大模型推理加速。**

- [安装与复现](outputs/qwen_audio_eval/README.md)
- [中文实验报告](outputs/qwen_audio_eval/REPORT_CN.md)
- [环境和实际阻塞](outputs/qwen_audio_eval/ENVIRONMENT.md)
- [推理入口](outputs/qwen_audio_eval/pipeline.py)
- [核心评测测试](outputs/qwen_audio_eval/test_metrics.py)
- [实际 VAD 结果](outputs/qwen_audio_eval/results/vad_summary.json)

核心冒烟集 24 条，另有 12 条短语音补充诊断。10 项核心测试通过。冻结 VAD 在核心 test 中保留 12/12 条语音并拒绝 4/4 条合成非语音；仅代表该小样本，不能推广至真实环境非语音。

目录保留原始交付结构。环境在本地 `work/.venv`，未上传；模型权重未上传。原始日志中的本地路径为运行时证据，在其他设备复现应按相对路径清单重新运行。

本仓库是独立评测工程，不是 Qwen 官方仓库。`official_repo` 与 `official_model_metadata` 保留对应官方许可证；公开语音数据来源和哈希在交付目录中记录。
