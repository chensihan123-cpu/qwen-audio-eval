"""Small, explicitly supplemental smoke corpus. No model-generated truth."""
import io, json, hashlib, random, re
from pathlib import Path
import numpy as np
import soundfile as sf
import requests
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    data = ROOT/'data'; data.mkdir(exist_ok=True)
    evidence = ROOT/'evidence'
    if not (evidence/'repo_commit.json').exists():
        html = requests.get('https://github.com/QwenLM/Qwen-Audio',timeout=30).text
        commits = re.findall(r'"currentOid":"([0-9a-f]{40})',html)
        save(evidence/'repo_commit.json', {'commit':commits[0] if commits else None,'source':'GitHub HTML currentOid'})
    src = ROOT.parent.parent/'work/librispeech.parquet'
    if not src.exists():
        src.parent.mkdir(parents=True,exist_ok=True)
        response=requests.get('https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/refs%2Fconvert%2Fparquet/clean/validation/0000.parquet',timeout=120)
        response.raise_for_status(); src.write_bytes(response.content)
    if hashlib.sha256(src.read_bytes()).hexdigest()!='4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603':
        raise ValueError('Dataset content changed; refuse non-reproducible sampling')
    records = pq.read_table(src).to_pylist()
    print('source records',len(records), 'fields', list(records[0]))
    rng = random.Random(20260924)
    eligible = []
    for r in records:
        x,sr = sf.read(io.BytesIO(r['audio']['bytes']))
        if 2 <= len(x)/sr <= 12:
            eligible.append((r,x,sr))
    picked = rng.sample(eligible,6)
    manifest=[]
    for i,(r,x,sr) in enumerate(picked):
        parent=str(r['id']); split='dev' if i<2 else 'test'
        for variant in ['original','low_volume','edge_silence']:
            y=x if variant=='original' else x*0.03 if variant=='low_volume' else np.pad(x,(sr,sr))
            sid=parent+'_'+variant; path=data/(sid+'.wav'); sf.write(path,y,sr,subtype='PCM_16')
            manifest.append(dict(sample_id=sid,audio_path='data/'+path.name,reference_text=r['text'],language='en',category='speech',source='hf-internal-testing/librispeech_asr_dummy (LibriSpeech derivative)',speaker_id=str(r.get('speaker_id','1272')),split=split,original_split='validation',parent_id=parent,annotation_status='public_transcript',condition=variant,synthetic=variant!='original'))
    nrng=np.random.default_rng(20260924)
    for i in range(6):
        sr=16000; n=sr*(i+1); kind='synthetic_silence' if i%2==0 else 'synthetic_white_noise'
        x=np.zeros(n) if i%2==0 else nrng.normal(0,0.025,n)
        sid=f'{kind}_{i}'; path=data/(sid+'.wav'); sf.write(path,x,sr,subtype='PCM_16')
        manifest.append(dict(sample_id=sid,audio_path='data/'+path.name,reference_text='',language='none',category='non_speech',source='procedural seed 20260924; supplemental only',speaker_id=None,split='dev' if i<2 else 'test',original_split=None,parent_id=sid,annotation_status='synthetic_known',condition=kind,synthetic=True))
    (ROOT/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest),encoding='utf-8')
    save(evidence/'dataset.json',dict(seed=20260924,source_url='https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/refs%2Fconvert%2Fparquet/clean/validation/0000.parquet',source_bytes=src.stat().st_size,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source_rows=len(records),selected_recordings=[str(r['id']) for r,_,_ in picked],n=len(manifest),split_unit='original recording (same speaker may occur across splits)',limitations='No Chinese; no manually audited real-world non-speech; synthetic supplements; same-speaker leakage limits generalization'))
    print('created',len(manifest))
if __name__=='__main__': main()
