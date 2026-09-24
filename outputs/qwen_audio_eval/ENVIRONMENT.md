# 环境与兼容性记录

检查/运行时间来自 `evidence/runtime.json`：2026-09-24T13:03:22.722988+00:00（北京时间 21:03）。不是项目起止周期。初始目录只有空的 work/outputs，无既有 Qwen 基础版代码或预测可保留。

| 项目 | 实际观察 |
|---|---|
| OS | Windows 11，build 26200 |
| Python | 系统 3.12.3；另有 3.11；本项目选 3.12.3 |
| GPU | RTX 4060 Laptop，8188MiB；记录时空闲 4452MiB（随其他程序变化） |
| 驱动 | 596.08；nvidia-smi 显示 CUDA 13.2 是驱动能力，不是已安装 Torch CUDA runtime |
| RAM | 总计 16,849,256,448 bytes，runtime 记录时可用 1,089,183,744 bytes |
| Torch | 2.7.0+cpu；CUDA build=null；cuda_available=false |
| Transformers | 专用环境 4.37.2；原系统 4.51.3 保留 |
| Accelerate | 0.30.1 |
| Silero / ONNX | 6.2.0 / 1.23.2，CPUExecutionProvider |
| FFmpeg | 7.0.2 essentials，PATH 可用 |
| 模型缓存 | HF_HOME=F:/LocalAI/Cache/huggingface；仅发现 MiniLM 缓存，未发现 Qwen-Audio 权重；默认用户缓存也无权重 |
| 磁盘 | C 初查约 25GB 空闲，F 约 729GB；逐盘精确值见 disk.json |

使用 `work/.venv` 的 system-site-packages 模式复用只读 CPU Torch/Numpy 等，专用环境覆盖安装较小依赖。`pip check` 通过。依赖选择并非复制官方旧版本全集：官方 4.32.0 的旧 tokenizers 对 Python 3.12 wheel 不利，选择保留旧 Qwen API 的 Transformers 4.37.2/tokenizers 0.15.2；已验证官方类导入、meta 构造、真实音频 tokenizer，**没有验证模型 forward**。未装无关的 captioning Java、TensorBoard、Web demo、FlashAttention 依赖。

官方模型 meta 构造实际统计 8,394,302,464 参数；BF16 参数 16,788,604,928 bytes（15.64GiB），FP32 33,577,209,856 bytes（31.27GiB），还不含激活/KV cache/音频编码器临时张量。官方模型 9 个 safetensors 文件共 16,792,529,560 bytes。没有下载这些权重；仅下载约几 MB 的元数据、词表和源码，逐文件 URL/字节数/SHA-256 见 downloads.json。

CUDA 版 Torch 尚未安装，不能在当前进程验证 GPU BF16 运算。8GB 显存明显不够 BF16 全量加载。CPU FP32 参数本身超过机器全部 RAM，CPU offload 也没有充足空闲内存。KV 量化开关在官方源码中存在，但它不等价于权重量化。当前 bitsandbytes 文档列出 Windows GPU wheel，不能据此断言它与此自定义 Qwen-Audio 代码、音频模块、旧 Transformers 组合兼容；未安装/未执行量化。Accelerate 的通用 CPU/disk offload 功能也不证明这里可无修改运行；适配器拒绝未验证的 auto/offload 加载方式。

实际失败证据：

- `results/single_attempt.jsonl` 及 `.meta.json`：CUDA unavailable，未生成文本。
- `evidence/actual_load_attempt.json`：官方 `from_pretrained` 真实调用，首个 safetensors 分片缺失 FileNotFoundError。
- `evidence/meta_model_probe_error.json`：第一次独立 tokenizer 探测没有传 audio_info，发生 TypeError；已依据真实源码增加 process_audio 并修复，最终 tokenizer_probe.json 成功（159 tokens、mel [1,80,3000]）。这个错误从未作为模型预测。
- Git clone/ls-remote 遇到网络重置；GitHub API 达到限流。通过 GitHub 页面 currentOid 取得 commit，并以该 commit 的 raw URL 下载官方文件；不是伪造本地 git HEAD。

最小继续条件：有足够资源的现成 CUDA 计算环境、同 revision 的完整权重、单条官方 Chat 推理成功。以官方约 24GB GPU 建议为起点，并留出 CPU RAM 与磁盘空间；安装命令参考 [PyTorch 固定版本页](https://pytorch.org/get-started/previous-versions/)。本机仅装 CUDA 包并不足以解决显存缺口。

兼容性资料：[Qwen 官方说明](https://github.com/QwenLM/Qwen-Audio/blob/b50fb958438081d36e1a14e93dbbc2f329c7f10e/README_CN.md)、[bitsandbytes 安装](https://huggingface.co/docs/bitsandbytes/main/en/installation)、[Accelerate 大模型推理](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling)。
