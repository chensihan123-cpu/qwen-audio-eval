"""Preserve source containers/audio bytes and audit download sizes."""
import hashlib, json, shutil
from pathlib import Path
import pyarrow.parquet as pq
import requests
from pipeline import ROOT, read_lines, write_json

def main():
    source=ROOT.parent.parent/'work/librispeech.parquet'
    folder=ROOT/'data/original'; folder.mkdir(exist_ok=True)
    selected={r['parent_id'] for r in read_lines(ROOT/'manifest.jsonl') if r['category']=='speech'}
    ledger=[]
    for r in pq.read_table(source).to_pylist():
        if str(r['id']) in selected:
            audio=r['audio']['bytes']; path=folder/(str(r['id'])+'.flac'); path.write_bytes(audio)
            ledger.append(dict(sample_id=str(r['id']),source_path=r['file'],saved_path=str(path.relative_to(ROOT)),bytes=len(audio),sha256=hashlib.sha256(audio).hexdigest(),transcript=r['text'],speaker_id=r['speaker_id'],chapter_id=r['chapter_id']))
    write_json(ROOT/'evidence/source_audio.json',ledger)
    urls={'librispeech_dummy_README.md':'https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/raw/main/README.md','silero_README.md':'https://raw.githubusercontent.com/snakers4/silero-vad/master/README.md','silero_LICENSE':'https://raw.githubusercontent.com/snakers4/silero-vad/master/LICENSE'}
    for name,url in urls.items():
        r=requests.get(url,timeout=30); r.raise_for_status(); (ROOT/'evidence'/name).write_bytes(r.content)
    write_json(ROOT/'evidence/additional_downloads.json',[dict(url=u,bytes=(ROOT/'evidence'/n).stat().st_size,sha256=hashlib.sha256((ROOT/'evidence'/n).read_bytes()).hexdigest()) for n,u in urls.items()])
if __name__=='__main__': main()
