# Qwen-Audio 语音大模型推理与评测优化

实际状态：工程、评分、真实 Silero VAD 实验已运行；**Qwen-Audio-Chat 权重推理未成功，不能报告提示词收益或 A—D 质量/速度提升**。详细数据见 `REPORT_CN.md`，机器记录见 `ENVIRONMENT.md`。

## 文件与版本

- `pipeline.py`：单条/清单逐条推理、A—D、VAD-only、严格断点续跑、评分与复核 CSV。没有实现或宣称张量 batch。
- `metrics.py`、`test_metrics.py`：语料级 S/D/I/N、字符/词评分、空参考处理、ID 对齐、缓存键测试。
- `config.json`：Qwen dev 初始配置；B 是未完成实证选择的候选。
- `config.vad_frozen.json`：只冻结 VAD 的 0.5 工作点，不允许据此启动 Qwen test。
- `manifest.jsonl`：24 条核心冒烟样本，dev 8 / test 16。
- `manifest.stress.jsonl`：12 条冻结后短语音诊断，不混入核心 test。
- `data/original`：10 段公开源 FLAC；`data`：实际使用的 WAV 和变体。
- `results/abcd_dev_final.jsonl`：32 条 A—D **运行尝试**，不是 32 条真实模型预测。
- `results/vad_summary.json`、`vad_stress_summary.json`：实际 VAD 指标。
- `evidence`：环境、下载哈希、实际错误、测试日志、加载/分词器探测、文件版本清单。
- `official_repo`、`official_model_metadata`：按 commit/revision 固定的官方源码和小型元数据；**不含大模型权重**。

仓库 commit：`b50fb958438081d36e1a14e93dbbc2f329c7f10e`。
模型：`Qwen/Qwen-Audio-Chat`，revision `8b1c0dc720d34da5498f93535f416e3590bf3a71`。

## 当前机器直接复现

以下 PowerShell 命令在最初工作目录执行。现成环境位于 `work/.venv`，只读复用系统 CPU Torch 等包，并在专用环境覆盖 Transformers 等依赖；未升级原项目环境。新结果使用新路径，防止覆盖实测记录。

```powershell
$py = (Resolve-Path work/.venv/Scripts/python.exe).Path
$project = (Resolve-Path outputs/qwen_audio_eval).Path
& $py "$project/pipeline.py" run --vad-only --split dev --output "$project/results/reproduce_dev.jsonl"
& $py "$project/pipeline.py" run --vad-only --split test --config "$project/config.vad_frozen.json" --performance --output "$project/results/reproduce_test.jsonl"
& $py "$project/pipeline.py" run --sample-id 1272-128104-0000_original --groups A --output "$project/results/reproduce_single.jsonl"
& $py "$project/pipeline.py" run --split dev --groups A B C D --output "$project/results/reproduce_abcd.jsonl"
& $py "$project/pipeline.py" score --split dev --results "$project/results/reproduce_abcd.jsonl" --output "$project/results/reproduce_summary.json"
Push-Location $project
& $py -m unittest -v test_metrics
Pop-Location
```

当前机器后两项 Qwen 运行仍应记录 `model_unavailable`。这是真实错误路径，不是模拟模式。`--sample-id` 支持清单中的任意单条；新音频先添加规范 manifest 行。音频路径相对于 manifest 文件。

需要续跑时对相同命令追加 `--resume`；已完成行不重写。音频、manifest 内容、模型 revision/精度、预处理、提示词、生成、VAD 版本/模型文件或流水线代码变化将报错，必须使用新输出。失败后重试也建议新输出，以保留错误证据。`--performance` 禁止同时使用 `--resume`，没有模型输出缓存。

## 新建环境与数据

先安装 FFmpeg 并加入 PATH。干净 CPU 环境可按下列固定依赖安装（当前实际环境及全部传递依赖快照另见 `evidence/environment.freeze.txt`，其中包含只读继承的无关系统包，不应全部盲装）：

```powershell
python -m venv work/.venv-clean
work/.venv-clean/Scripts/python -m pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
work/.venv-clean/Scripts/python -m pip install -r outputs/qwen_audio_eval/requirements-overlay.txt --extra-index-url https://download.pytorch.org/whl/cpu
work/.venv-clean/Scripts/python outputs/qwen_audio_eval/prepare.py
```

`prepare.py` 缺少源数据时下载 9,192,059 字节 parquet，严格验证 SHA-256，然后按种子 `20260924` 重建相同 24 条清单。`stress.py` 从同一源容器建立短语音补充集；它写入已存在结果时会拒绝覆盖。源划分是 HF dummy 的 `clean/validation`，不是完整 LibriSpeech 官方测试集。公开文本来自原语料，未用模型制造参考。

构建自己的正式数据时，每行包括 `sample_id,audio_path,reference_text,language,category,source,speaker_id,split,parent_id,annotation_status`，建议另加 `condition,synthetic,original_split`。`category` 用 `speech/non_speech/boundary`；人工确认后标为 `human_verified`，未确认的 `pending_review` 不计正式指标。优先按 speaker 分组；本次单说话人微型集按原录音分组，变体不跨 dev/test。不可把同一 parent 拆开。歌声、电视人声和难辨低音量应进入边界/听审集合。

## 在资源就绪后完成 Qwen 主线

1. 使用足够显存/内存的现有计算资源。官方示例给出的 GPU 需求约 24GB；本项目不把这一数字当作所有长度下的保证。安装 CUDA 版 Torch 后先检查 `torch.cuda.is_available()` 和 BF16 支持；另建环境，保留本次 CPU 实验环境。PyTorch 官方为 2.7.0 提供 cu128 安装源。
2. 下载**同一固定 revision**全部权重，预计 16,792,529,560 字节权重分片，另需缓存/运行空间。可执行：

```python
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Audio-Chat', revision='8b1c0dc720d34da5498f93535f416e3590bf3a71')
```

3. 保持 `local_files_only=true`，先跑上述单条入口并检查原始输出、生成 token 数、截断标记和耗时。适配器从固定 snapshot 的本地路径加载，避免官方远程分支单独下载未固定版本的 mel 滤波文件。模型常驻、eval、inference_mode，每条 `history=None`；官方 chat 解码分离提示与输出。
4. 仅在 dev 用配置副本比较两个 B 候选。先看真实语音质量/空输出和非语音格式错误，再结合 VAD 误拒选择工作点；不得因本次合成数据完美门控直接宣称最优。所有组保持模型、精度、加载方式和生成配置相同。
5. 选定后创建独立配置，填入选择依据并设置 `frozen=true`、`prompt_selection_complete=true`。首次 test 前存档清单/配置哈希。然后 `run --split test --config <冻结文件> --groups A B C D --performance --output <新文件>`，再 `score --split test`。不得根据 test 重调参数。

本机不自动启用 4bit、CPU offload 或更换模型；兼容性未通过真实权重验证。只释放本机少量内存或安装 CUDA 包不能让 BF16 权重装入 8GB 显存。没有使用付费资源、Whisper 替代模型、Qwen2-Audio 或微调。

## 评分与耗时口径

归一化固定：NFKC、casefold、Unicode 标点替换空格；中文去空白按字符，英文空白分词；数字不转写。参考/预测相同规则。仅整个去首尾空白后的字符串精确等于 `[NO_SPEECH]` 时解析为空；“这是音乐”等保留并进入人工分类，不自动称为幻觉。

CER/WER=(S+D+I)/N，先汇总计数。语音被 VAD 跳过或推理失败按空预测计算保守删除错误，失败同时单列完成率；`quality_claim_allowed=false` 时禁止据此写模型优化结论。非语音的空参考不计算 WER/CER，分别统计输出比例和字符/词插入量；未完成样本提供上下界。待复核 CSV 包含原始回复和重复筛查，重复筛查不是最终错误标签。

`preprocess_s` 为音频校验及 16k 单声道解码；`vad_s` 为 Silero；`model_s` 为同步后的 generate（包括音频编码器）；`model_frontend_s` 为官方 chat 的特征/提示准备与解码，其内部会再次读取完整原音频。`e2e_s` 为每条入口端到端时间（含哈希和准备，不含写 JSONL）。加载和单独预热记录于旁边 `.meta.json`。本次无真实模型调用，不能把诊断耗时当作推理延迟。VAD 数字含其首次使用开销，未用于比较阈值的速度优劣。

RTF=样本耗时之和/原始音频时长之和，非逐条 RTF 均值。报告全工作负载与实际调用模型子集，峰值 GPU allocated memory 同步采集；没有 CUDA 数据时为 null。

超过 30 秒标为 `needs_segmentation`，不调用模型、不静默截断。完整长音频分段/拼接实验尚未实现，必须作为独立扩展处理，不能将此状态当作成功转写。环境音/音乐理解用 `mode=understanding` 时绕过 C/D 的门控；该模式非本次转写评测。
