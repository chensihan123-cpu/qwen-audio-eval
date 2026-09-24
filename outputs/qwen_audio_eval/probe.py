"""Save actual machine/runtime and immutable official sources without downloading weights."""
import datetime, hashlib, importlib.metadata, json, platform, traceback
from pathlib import Path
import psutil, requests, torch
ROOT=Path(__file__).resolve().parent
def dump(name,obj): (ROOT/'evidence'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    model=cfg['model']; rev=model['revision']
    runtime=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),memory=psutil.virtual_memory()._asdict(),packages={x:importlib.metadata.version(x) for x in ['torch','torchaudio','transformers','accelerate','silero-vad','onnxruntime','numpy','soundfile','librosa']})
    dump('runtime.json',runtime)
    files=['config.json','configuration_qwen.py','modeling_qwen.py','tokenization_qwen.py','audio.py','qwen_generation_utils.py','cpp_kernels.py','generation_config.json','tokenizer_config.json','qwen.tiktoken','mel_filters.npz','model.safetensors.index.json','LICENSE']
    folder=ROOT/'official_model_metadata'; folder.mkdir(exist_ok=True); downloads=[]
    for name in files:
        url=f'https://huggingface.co/{model["id"]}/resolve/{rev}/{name}'
        r=requests.get(url,timeout=40); r.raise_for_status(); (folder/name).write_bytes(r.content)
        downloads.append(dict(url=url,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest()))
    commit=json.loads((ROOT/'evidence/repo_commit.json').read_text())['commit']
    repo=ROOT/'official_repo'; repo.mkdir(exist_ok=True)
    for name in ['README_CN.md','requirements.txt','eval_audio/EVALUATION.md','eval_audio/evaluate_asr.py','audio.py','modeling_qwen.py','tokenization_qwen.py','qwen_generation_utils.py','LICENSE']:
        url=f'https://raw.githubusercontent.com/QwenLM/Qwen-Audio/{commit}/{name}'
        r=requests.get(url,timeout=40); r.raise_for_status(); path=repo/name; path.parent.mkdir(exist_ok=True,parents=True); path.write_bytes(r.content)
        downloads.append(dict(url=url,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest()))
    dump('downloads.json',downloads)
    from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig
    try:
        AutoModelForCausalLM.from_pretrained(str(folder),trust_remote_code=True,local_files_only=True,device_map='cpu',fp32=True,use_flash_attn=False)
    except Exception:
        dump('actual_load_attempt.json',dict(status='failed',error=traceback.format_exc(),note='Real from_pretrained attempt against downloaded official metadata; weights intentionally absent; no generation'))
    try:
        from accelerate import init_empty_weights
        config=AutoConfig.from_pretrained(str(folder),trust_remote_code=True)
        config.fp32=True; config.bf16=False; config.fp16=False; config.use_flash_attn=False
        with init_empty_weights(): m=AutoModelForCausalLM.from_config(config,trust_remote_code=True)
        params=sum(p.numel() for p in m.parameters())
        dump('meta_model_probe.json',dict(status='success_no_weights_no_forward',parameters=params,bf16_weight_bytes=params*2,fp32_weight_bytes=params*4,note='Meta device only: verifies construction/import, NOT inference, offload, or quantization'))
        tok=AutoTokenizer.from_pretrained(str(folder),trust_remote_code=True)
        query=tok.from_list_format([{'audio':str(ROOT/'data/1272-128104-0000_original.wav')},{'text':cfg['prompts']['A']}])
        audio_info=tok.process_audio(query)
        tokens=tok(query,return_tensors='pt',audio_info=audio_info)
        dump('tokenizer_probe.json',dict(status='success',query=query,input_tokens=int(tokens.input_ids.shape[-1]),note='Real official tokenizer/audio path; no model output'))
    except Exception:
        dump('meta_model_probe_error.json',dict(error=traceback.format_exc()))
if __name__=='__main__': main()
