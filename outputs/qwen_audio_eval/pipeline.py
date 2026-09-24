"""Sequential manifest execution; deliberately not tensor batching."""
import argparse, copy, csv, hashlib, json, os, subprocess, time, traceback
from pathlib import Path
import numpy as np
from metrics import align, cache_key, counts, normalize, parse_output, quality

ROOT=Path(__file__).resolve().parent
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read_lines(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def write_json(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_manifest(path,split=None):
    rows=read_lines(path); ids=set(); splits={}
    for r in rows:
        required=['sample_id','audio_path','reference_text','language','category','source','split','parent_id','annotation_status']
        if any(k not in r for k in required): raise ValueError('missing manifest fields')
        if r['sample_id'] in ids: raise ValueError('duplicate sample_id')
        ids.add(r['sample_id'])
        parent=r['parent_id'] or r['sample_id']
        if parent in splits and splits[parent]!=r['split']: raise ValueError('parent leaks across split')
        splits[parent]=r['split']
    return [r for r in rows if not split or r['split']==split]

def audio_check(path,cfg):
    import soundfile as sf
    if not path.is_file(): raise FileNotFoundError(str(path))
    info=sf.info(str(path))
    if info.frames/info.samplerate>cfg['max_duration_s']:
        return None,dict(duration_s=info.frames/info.samplerate,original_sample_rate=info.samplerate,original_channels=info.channels,original_subtype=info.subtype,peak=None)
    raw,sr=sf.read(str(path),dtype='float32',always_2d=True)
    if not raw.size or not np.isfinite(raw).all(): raise ValueError('empty or nonfinite input')
    duration=len(raw)/sr
    meta=dict(duration_s=duration,original_sample_rate=sr,original_channels=raw.shape[1],original_subtype=info.subtype,peak=float(np.abs(raw).max()))
    if duration>cfg['max_duration_s']: return None,meta
    cmd=['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(path),'-f','s16le','-ac','1','-acodec','pcm_s16le','-ar','16000','-']
    result=subprocess.run(cmd,capture_output=True,check=True)
    x=np.frombuffer(result.stdout,dtype='<i2').astype(np.float32)/32768.0
    if not len(x): raise ValueError('decoded audio empty')
    return x,meta

class Vad:
    def __init__(self):
        import torch, silero_vad, importlib.metadata
        torch.set_num_threads(1)
        self.torch=torch; self.timestamps=silero_vad.get_speech_timestamps
        self.model=silero_vad.load_silero_vad(onnx=True)
        model_path=Path(silero_vad.__file__).parent/'data/silero_vad.onnx'
        self.version={'version':importlib.metadata.version('silero-vad'),'onnx_sha256':sha(model_path)}
    def run(self,x,params):
        return self.timestamps(self.torch.from_numpy(x.copy()),self.model,sampling_rate=16000,return_seconds=False,**params)

class Qwen:
    def __init__(self,cfg):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.torch=torch
        model=cfg['model']; revision=model['revision']; dtype=model['precision']
        if model['id']!='Qwen/Qwen-Audio-Chat': raise ValueError('This adapter only supports Qwen-Audio-Chat; do not silently switch model')
        if len(revision)!=40: raise ValueError('immutable model revision required')
        if model['device_map']!='cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in installed PyTorch; no Qwen generation performed')
        if dtype=='bf16' and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unsupported')
        if model['device_map'] not in ['cpu','cuda']: raise ValueError('CPU offload/device_map auto is not validated by this adapter')
        import psutil
        needed=8394302464*(4 if dtype=='fp32' else 2)+3*1024**3
        available=psutil.virtual_memory().available if model['device_map']=='cpu' else torch.cuda.mem_get_info()[0]
        if available<needed: raise RuntimeError(f'Insufficient free memory: {available} bytes available, conservative minimum {needed}; quantization/offload not verified')
        t=time.perf_counter()
        common=dict(revision=revision,trust_remote_code=True,local_files_only=model['local_files_only'])
        # Official remote loader fetches mel_filters without a revision. A pinned local
        # snapshot also pins that asset and avoids mixing main with the requested weights.
        from huggingface_hub import snapshot_download
        local=snapshot_download(repo_id=model['id'],revision=revision,local_files_only=model['local_files_only'])
        self.tokenizer=AutoTokenizer.from_pretrained(local,**common)
        self.model=AutoModelForCausalLM.from_pretrained(local,device_map=model['device_map'],use_flash_attn=False,bf16=dtype=='bf16',fp16=dtype=='fp16',fp32=dtype=='fp32',**common).eval()
        self.load_s=time.perf_counter()-t
        self.generation=copy.deepcopy(self.model.generation_config)
        self.generation.update(**cfg['generation'])
        self.last_token_count=None
        self.last_generation_s=None
        # Inspect the actual generated token sequence while preserving official chat decoding.
        original=self.model.generate
        def monitored(*args,**kwargs):
            self.sync(); started=time.perf_counter()
            result=original(*args,**kwargs)
            self.sync(); self.last_generation_s=time.perf_counter()-started
            inputs=kwargs.get('input_ids',args[0] if args else None)
            self.last_token_count=int(result.shape[-1]-inputs.shape[-1])
            return result
        self.model.generate=monitored
    def sync(self):
        if self.torch.cuda.is_available(): self.torch.cuda.synchronize()
    def run(self,path,prompt):
        q=self.tokenizer.from_list_format([{'audio':str(path)},{'text':prompt}])
        self.sync()
        if self.torch.cuda.is_available(): self.torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        with self.torch.inference_mode():
            response,_=self.model.chat(self.tokenizer,query=q,history=None,generation_config=self.generation)
        self.sync()
        peak=self.torch.cuda.max_memory_allocated() if self.torch.cuda.is_available() else None
        elapsed=time.perf_counter()-t
        return response,self.last_generation_s,peak,self.last_token_count,elapsed-self.last_generation_s

def run(args):
    cfg=read_json(args.config); rows=load_manifest(args.manifest,args.split)
    if cfg['preprocess']['sample_rate']!=16000 or cfg['preprocess']['channels']!=1 or cfg['preprocess']['denoise'] or cfg['preprocess']['normalize_loudness']:
        raise ValueError('Only official 16k mono preprocessing without denoising/normalization is implemented')
    if args.sample_id: rows=[r for r in rows if r['sample_id']==args.sample_id]
    if not rows: raise ValueError('no selected samples')
    if args.split=='test' and not cfg.get('frozen'): raise ValueError('freeze configuration before test')
    if args.split=='test' and not args.vad_only and not cfg.get('prompt_selection_complete'): raise ValueError('Qwen prompt selection on dev is incomplete')
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and not args.resume: raise FileExistsError('use a new output or --resume')
    if args.resume and args.performance: raise ValueError('performance requires fresh output, no resume/cache')
    old=read_lines(out) if out.exists() else []
    vad=Vad() if args.vad_only or any(g in args.groups for g in 'CD') else None
    model=None; load_error=None; warmup_s=None
    if not args.vad_only:
        try:
            model=Qwen(cfg)
            # Explicitly separate one warmup from measured samples.
            for row in rows:
                path=(Path(args.manifest).parent/row['audio_path']).resolve()
                x,meta=audio_check(path,cfg['preprocess'])
                if x is not None:
                    _,generation_s,_,_,frontend_s=model.run(path,cfg['prompts']['A']); warmup_s=generation_s+frontend_s; break
        except Exception:
            load_error=traceback.format_exc(); model=None
    metadata=dict(config=cfg,started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),load_error=load_error,model_load_s=model.load_s if model else None,warmup_s=warmup_s,effective_generation_config=model.generation.to_dict() if model else None,vad_version=vad.version if vad else None,execution='vad_only' if args.vad_only else 'qwen',performance=args.performance)
    write_json(str(out)+'.meta.json',metadata)
    groups=['VAD'] if args.vad_only else args.groups
    with out.open('a',encoding='utf-8') as f:
        for row in rows:
            for group in groups:
                start=time.perf_counter(); path=(Path(args.manifest).parent/row['audio_path']).resolve()
                prompt=cfg['prompts']['B' if group in 'BD' else 'A']
                key_config=dict(config=cfg,manifest_row=row,group=group,vad_version=vad.version if vad else None,prompt=prompt,pipeline_version='1.1',source_hashes={p:sha(ROOT/p) for p in ['pipeline.py','metrics.py']})
                audio_hash=sha(path) if path.is_file() else None; key=cache_key(audio_hash,key_config)
                previous=[r for r in old if r['sample_id']==row['sample_id'] and r['group']==group]
                if previous:
                    if previous[0]['cache_key']!=key: raise ValueError('stale resume: configuration/audio/code changed; use new output')
                    continue
                r=dict(row,group=group,audio_path=str(path),audio_sha256=audio_hash,cache_key=key,model_version=cfg['model'],prompt=prompt,generation_config=cfg['generation'],vad_params=cfg['vad'] if vad else None,vad_intervals=[],raw_output=None,scoring_text='',status=None,timings=dict(preprocess_s=0,vad_s=0,model_s=0,model_frontend_s=0),error=None,peak_gpu_bytes=None,model_called=False,generated_tokens=None,truncated=None,from_cache=False,preprocess=cfg['preprocess'])
                stage='audio'
                try:
                    t=time.perf_counter(); x,meta=audio_check(path,cfg['preprocess']); r.update(meta); r['timings']['preprocess_s']=time.perf_counter()-t
                    if x is None: r['status']='needs_segmentation'
                    else:
                        gate=args.vad_only or (group in ['C','D'] and cfg['mode']=='transcription')
                        if gate:
                            stage='vad'
                            t=time.perf_counter(); intervals=vad.run(x,cfg['vad']); r['timings']['vad_s']=time.perf_counter()-t; r['vad_intervals']=intervals
                        if gate and not intervals: r['status']='vad_skip'
                        elif args.vad_only: r['status']='vad_pass'
                        elif model is None: r['status']='model_unavailable'; r['error']=load_error
                        else:
                            r['model_called']=True
                            stage='model'; t=time.perf_counter()
                            raw,elapsed,peak,nt,frontend=model.run(path,prompt)
                            r.update(raw_output=raw,scoring_text=parse_output(raw),peak_gpu_bytes=peak,generated_tokens=nt,truncated=nt>=cfg['generation']['max_new_tokens'])
                            r['timings']['model_s']=elapsed
                            r['timings']['model_frontend_s']=frontend
                            r['status']='ok' if r['scoring_text'] else 'model_empty'
                except Exception:
                    r['status']='inference_error' if r['model_called'] else 'vad_error' if stage=='vad' else 'audio_error'; r['error']=traceback.format_exc()
                    r['timings'][{'audio':'preprocess_s','vad':'vad_s','model':'model_s'}[stage]]=time.perf_counter()-t
                r['timings']['e2e_s']=time.perf_counter()-start
                f.write(json.dumps(r,ensure_ascii=False)+'\n'); f.flush()
                print(row['sample_id'],group,r['status'],flush=True)

def summary(args):
    rows=load_manifest(args.manifest,args.split); results=read_lines(args.results)
    eligible={'public_transcript','human_verified','synthetic_known'}
    report={}; review=[]
    for group in sorted({r['group'] for r in results}):
        pairs=align(rows,results,group)
        use=[(m,r) for m,r in pairs if m['annotation_status'] in eligible]
        speech=[(m,r) for m,r in use if m['category']=='speech']; noise=[(m,r) for m,r in use if m['category']=='non_speech']
        valid={'ok','model_empty','vad_skip'}
        complete=sum(r['status'] in valid for m,r in use)
        ratio=lambda n,d:n/d if d else None
        q={lang:quality([(m,r) for m,r in speech if m['language']==lang]) for lang in sorted({m['language'] for m,r in speech})}
        def perf(p):
            times=[r['timings']['e2e_s'] for m,r in p]; dur=sum(r.get('duration_s',0) for m,r in p)
            return dict(n=len(p),audio_seconds=dur,median_s=float(np.median(times)) if times else None,p95_s=float(np.percentile(times,95)) if times else None,rtf=ratio(sum(times),dur),peak_gpu_bytes=max([r.get('peak_gpu_bytes') or 0 for m,r in p],default=0) or None)
        report[group]=dict(n=len(use),completion_rate=ratio(complete,len(use)),status_counts={s:sum(r['status']==s for m,r in use) for s in sorted({r['status'] for m,r in use})},conservative_quality=q,quality_claim_allowed=complete==len(use) and group!='VAD',vad_false_reject_rate=ratio(sum(r['status']=='vad_skip' for m,r in speech),len(speech)),speech_empty_rate=ratio(sum(not r['scoring_text'] for m,r in speech),len(speech)),non_speech_output_rate_observed=ratio(sum(bool(r['scoring_text']) for m,r in noise),len(noise)),non_speech_output_rate_upper=ratio(sum(bool(r['scoring_text']) or r['status'] not in valid for m,r in noise),len(noise)),performance_valid=complete==len(use) and group!='VAD',workload_diagnostic=perf(use),model_called_diagnostic=perf([(m,r) for m,r in use if r['model_called']]),by_condition={c:quality([(m,r) for m,r in speech if m.get('condition')==c]) for c in sorted({m.get('condition','unknown') for m,r in speech})})
        report[group]['speech_retention_rate']=1-report[group]['vad_false_reject_rate'] if speech else None
        report[group]['non_speech_insertions']={unit:sum(len(normalize(r['scoring_text'],lang)) for m,r in noise) for unit,lang in [('characters','zh'),('words','en')]}
        report[group]['by_duration']={label:quality([(m,r) for m,r in speech if lo<=r.get('duration_s',0)<hi]) for label,lo,hi in [('short_lt_3s',0,3),('3_to_10s',3,10),('10_to_30s',10,30.0001)]}
        if group=='VAD':
            for field in ['conservative_quality','speech_empty_rate','non_speech_output_rate_observed','non_speech_output_rate_upper','non_speech_insertions','by_condition','by_duration']:
                report[group][field]=None
        for m,r in pairs:
            text=r.get('scoring_text','')
            toks=normalize(text,m['language'])
            review.append(dict(sample_id=m['sample_id'],group=group,category=m['category'],condition=m.get('condition'),status=r['status'],reference=m['reference_text'],raw_output=r.get('raw_output'),scoring_text=text,review_label='pending' if text else 'no_output_or_unavailable',repetition_screen=len(toks)>=6 and len(set(toks))<len(toks)/2,review_note='Do not infer hallucination or repetition error without listening'))
    write_json(args.output,report)
    with Path(str(args.output)+'.review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(review[0]) if review else ['sample_id']); w.writeheader(); w.writerows(review)
    print(json.dumps(report,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('run'); r.add_argument('--config',default=str(ROOT/'config.json')); r.add_argument('--manifest',default=str(ROOT/'manifest.jsonl')); r.add_argument('--split',choices=['dev','test']); r.add_argument('--sample-id'); r.add_argument('--groups',nargs='+',choices=list('ABCD'),default=list('ABCD')); r.add_argument('--vad-only',action='store_true'); r.add_argument('--resume',action='store_true'); r.add_argument('--performance',action='store_true'); r.add_argument('--output',required=True)
    s=sub.add_parser('score'); s.add_argument('--manifest',default=str(ROOT/'manifest.jsonl')); s.add_argument('--split',choices=['dev','test']); s.add_argument('--results',required=True); s.add_argument('--output',required=True)
    a=p.parse_args(); run(a) if a.cmd=='run' else summary(a)
if __name__=='__main__': main()
