"""Reports are derived only from saved executions, never synthesized predictions."""
import csv, hashlib, json
from pathlib import Path
from pipeline import ROOT, read_json, read_lines, write_json

def main():
    vad=read_json(ROOT/'results/vad_summary.json'); q=read_json(ROOT/'results/abcd_dev_final_summary.json')
    stress=read_json(ROOT/'results/vad_stress_summary.json'); test=vad['test']
    table='\n'.join(f"| {g} | {r['n']} | {r['status_counts'].get('model_unavailable',0)} | {r['status_counts'].get('vad_skip',0)} | {r['completion_rate']:.0%} | 未测得 |" for g,r in q.items())
    vtable='\n'.join(f"| {t} | {r['speech_rejected']}/{r['speech']} | {r['non_speech_passed']}/{r['non_speech']} |" for t,r in vad['dev'].items())
    report=f'''# 中文实验报告：Qwen-Audio 语音推理与评测优化

本轮完成可执行工程、真实音频处理、真实 Silero VAD 实验和评分逻辑验证。**没有成功执行 Qwen-Audio-Chat 权重推理，不能回答提示词是否降低非语音无关输出，也不能报告 A/B/C/D 的转写提升或系统加速。** 下文严格区分已运行的门控结果与未运行的模型结论。

## 执行与数据

执行日期由运行证据记录为 2026-09-24。模型固定为 Qwen-Audio-Chat，未改用基础版、Qwen2-Audio 或 Whisper。仓库 commit 和模型 revision 见 README；真实环境详见 ENVIRONMENT。初始目录无原有基础版基线。

核心冒烟集 24 条：18 条英文语音（6 段独立原录音，每段原始、幅度×0.03、首尾各加 1 秒静音），6 条程序生成静音/白噪声。语音来自带公开人工转写的 LibriSpeech dummy 子集。种子 20260924，按 parent 原录音划分 dev/test=8/16；同 parent 及全部变体处于同划分。所有真实录音来自同一说话人 1272，不具备跨说话人泛化证据。原音频与下载容器哈希已保留。

核心总时长 {vad['dev']['0.5']['duration_s']+test['duration_s']:.3f} 秒，dev {vad['dev']['0.5']['duration_s']:.3f} 秒、test {test['duration_s']:.3f} 秒。另有冻结后短语音诊断 12 条（4 段 1.64–1.91 秒的公开原录音及低音量、10dB SNR 白噪声变体），共 {stress['duration_s']:.3f} 秒。8 条新短语音变体尚未人工听审，不进入正式 ASR 成绩。

合计文件清单 36 条，不是 600 条正式实验集；中文 0 条，真实环境非语音 0 条。没有声称人工确认过音乐、机械声、环境噪声、歌声或电视人声。合成非语音标签来自生成过程，未用 VAD 输出当真值，合成样本只提供软件冒烟证据。没有对完整 AISHELL/LibriSpeech 官方基准进行评测。

## A/B/C/D 主线实际运行状态

A 普通提示，无 VAD；B 约束提示，无 VAD；C 普通提示加整段 VAD 门控；D 约束提示加同一门控。通过 VAD 的音频仍整段送入官方接口，没有裁剪。各组配置同一模型、BF16、cuda、确定性贪心参数；但这些生成参数尚未通过真实 forward 验证。B 仅是预设候选，另一个英文候选已保存，无法进行模型层面的 dev 选择，因此没有运行 Qwen test。

| 组 | dev 样本 | 模型不可用 | VAD 跳过 | 系统完成率 | 真实模型 WER/延迟 |
|---|---:|---:|---:|---:|---|
{table}

完成率把正确执行的门控跳过视为完成；C/D 的 25% 只代表 2 条非语音被门控跳过，**不代表转写成功**。四组实际模型调用次数均为 0；raw_output 全部为 null。Qwen API 的单条尝试保存音频路径、哈希、提示词、耗时与真实错误，没有模拟回复。

失败空预测的保守记账：各组英文 S=0、D=111、I=0、N=111，保守 WER=100%。这是资源阻塞导致的系统失败记账，不是已测得的模型识别能力，也不是四组质量相同的证据。A/B 的非语音输出率不可识别，完成情况对应界限为 0%–100%；C/D 这 2 条已知合成非语音的系统输出均为空，不能与 A/B 缺失输出相减宣称降低 100%。

## Silero VAD 的实际实验

Silero 6.2.0 ONNX CPU，16k 单声道。固定 min_speech_duration_ms=250、min_silence_duration_ms=100、speech_pad_ms=30。区间以 16k 采样点保存。没有人工时间区间真值，不计算时间级召回率。

| dev threshold | 真实语音误拒 | 合成非语音放行 |
|---|---:|---:|
{vtable}

三个 dev 点在门控标签上相同。以语音误拒优先、再比较非语音放行、打平取离 0.5 最近的规则，冻结 0.5。选择记录早于 test，test 后没有调阈值或提示词。因为只有 2 段独立 dev 语音和 2 条合成非语音，不宣称最优工作点或显著改善。

冻结 test：{test['speech_rejected']}/{test['speech']} 语音被误拒，门控保留率 100%；{test['non_speech_passed']}/{test['non_speech']} 合成非语音被放行。12 条语音实际只有 4 段独立原录音，不能当 12 个独立观测。短语音补充诊断 12/12 放行，4/4 原始短语音放行；低音量/加噪变体结论仅为操作性门控结果，待听审。没有测得转写正确率，不能把门控保留率当作识别准确率。

## 耗时

冻结 test 共 16 条、{test['duration_s']:.3f} 秒音频，CPU VAD 时间中位数 {test['vad_median_ms']:.2f}ms、P95 {test['vad_p95_ms']:.2f}ms；校验/解码加 VAD 的端到端诊断中位数 {test['preprocess_plus_vad_median_ms']:.2f}ms、P95 {test['preprocess_plus_vad_p95_ms']:.2f}ms，RTF={test['preprocess_plus_vad_rtf']:.5f}。RTF 是总耗时/总原始音频时长。

这些数字不含 Qwen 生成。ONNX 模型加载在逐条计时外，首次 VAD 调用未另行预热；机器同时运行其他应用，dev 各阈值耗时不作速度排名。没有模型加载成功时间、正式生成时间或峰值推理显存，全部记为 null；不能从快速失败时间推导加速比例。实际调用模型子集为空。核心 4/16 test 输入可被门控跳过，但大模型系统节省时长仍未测量。

## 评分、复核和案例

评分固定 NFKC、casefold、Unicode 标点到空格，保留数字字面形式；中文按字符、英文按空白分词。S/D/I/N 语料级累加。精确 `[NO_SPEECH]` 标记解析为空；无语音说明、声音描述、伪造语音及其他文本需人工判别。空参考单独统计输出和插入量，不硬算 WER。推理失败单列，并按空预测给出保守质量。

- 官方示例同 ID 的真实录音 `1272-128104-0000_original`：官方 tokenizer 实际得到 159 个输入 token、mel [1,80,3000]；模型尝试记录 CUDA unavailable，无法判断漏字、重复或无关输出。
- `synthetic_white_noise_1`：dev 三个阈值均拒绝，C/D 跳过；A/B 未产生模型输出，无法据此评价提示词。
- `1272-141231-0013_short_original`（公开文本 THE TWENTIES）：冻结 0.5 下放行；这只能排除该条整段 VAD 误拒，不能证明 Qwen 能正确识别。
- 没有可报告的模型改善/退化/无明显变化配对案例，因为不存在真实模型预测。没有重复生成错误案例；重复筛查只在未来文本上提示人工检查，不能将真实口吃/重复判为生成错误。

复核表 `results/abcd_dev_final_summary.json.review.csv` 保留每个 ID/组/状态/参考/原始输出。`results/annotation_review.csv` 列出 8 条短语音变体的听审任务。尚缺的中文、真实非语音及边界类别列于 `results/data_gaps.json`，未用虚构 sample_id 填充。未计算配对置信区间：Qwen 配对输出不存在，VAD 独立录音也很少且同说话人。零观测误拒不意味着总体误拒率为零。

## 验证与完成边界

已实现且已运行：输入校验/哈希/16k 解码、整段 VAD、按 parent 防泄漏、三阈值 dev 与冻结 test、断点与配置失效校验、ID 对齐、语料级评分、复核表、10 项单元测试、4 项 CLI 防误用校验。pip check 无损坏依赖。代码/交付文件 SHA-256 清单随附。

已实现但未通过真实权重验证：Chat 模型常驻生成、有效生成参数保存、GPU 同步计时、峰值显存、长度截断监测、官方提示/回复分离。基础模型没有替换为 Chat 后再虚报收益。

未完成：Qwen 单条成功推理、提示词 dev 选择、A—D 完整 test、CER/WER 实测、真实非语音人工分类、600 条数据、中文与边界语音、量化/offload 实证、长音频分段/拼接、张量 batch、配对模型置信区间。超过 30 秒已拦截为 needs_segmentation，但完整分段流程未实现。

继续所需最少操作：提供足够资源的现成 CUDA 环境和同 revision 完整权重，先验证单条；补充可靠标注的中文/真实非语音与人工听审；只在 dev 完成提示词/阈值选择，冻结后首次运行完整 test。README 提供命令。当前结果只支持本组数据下的 VAD 门控观察，不支持模型内部能力提升。
'''
    (ROOT/'REPORT_CN.md').write_text(report,encoding='utf-8')
    pending=[m for m in read_lines(ROOT/'manifest.stress.jsonl') if m['annotation_status']=='derived_pending_review']
    for m in pending: m.update(review_question='Listen: intelligible speech present? Is public source transcript still fully audible?',reviewer='',review_status='pending')
    with (ROOT/'results/annotation_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pending[0])); w.writeheader(); w.writerows(pending)
    write_json(ROOT/'results/data_gaps.json',dict(chinese_speech='0 available; target 200',english_speech='10 unique recordings, 30 waveform entries including variants; target 200 independently labeled items',real_non_speech='0 available; target 200; need manually checked silence/environment/instrumental/mechanical audio',boundary='singing/TV voices/very low intelligibility missing',formal_600='not completed'))
    (ROOT/'RESUME_CN.md').write_text('''# 可用于简历的项目描述

**Qwen-Audio 语音推理与评测工程（资源受限验证）**

- 基于官方 Qwen-Audio-Chat 源码搭建可追溯的转写评测流水线，支持 A/B/C/D 提示词与 Silero VAD 对照配置、清单逐条执行、音频哈希、断点续跑和严格配置失效检查。
- 实现中英文 CER/WER 的语料级 S/D/I/N 统计、失败/门控跳过的保守计分、非语音响应复核和阶段耗时记录；10 项核心单元测试通过。
- 完成 24 条冒烟样本的真实 Silero VAD 实验及 12 条短语音补充诊断；冻结工作点在核心 test 中保留 12/12 条语音并拒绝 4/4 条合成非语音，明确单说话人、小样本与合成数据的适用范围。

面试时应主动说明：Qwen 权重推理因 8GB 显存、内存不足及缺少权重尚未跑通；没有测得提示词收益、WER 改善、真实环境幻觉下降或大模型推理加速。不能写“将幻觉降低 X%”“提高识别准确率”“完成 600 条基准”等未验证结果。项目日期只可使用真实参与日期，不扩写为虚构周期。
''',encoding='utf-8')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_sha256.json'}
    write_json(ROOT/'evidence/delivery_sha256.json',hashes)
    print('Report and review artifacts generated from actual result files.')
if __name__=='__main__': main()
