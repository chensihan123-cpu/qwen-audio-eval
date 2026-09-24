"""Post-freeze supplemental short/noisy speech audit; no threshold adjustment."""
import io, json, subprocess, sys
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from pipeline import ROOT, write_json
from vad_experiment import stats

def main():
    records=pq.read_table(ROOT.parent.parent/'work/librispeech.parquet').to_pylist()
    manifest=[]; rng=np.random.default_rng(20260924)
    for r in records:
        x,sr=sf.read(io.BytesIO(r['audio']['bytes']))
        if len(x)/sr>=2: continue
        for condition in ['short_original','short_low_volume','short_noise_snr10']:
            if condition=='short_original': y=x
            elif condition=='short_low_volume': y=x*.03
            else:
                noise=rng.normal(0,1,len(x)); noise*=np.sqrt(np.mean(x*x)/10/np.mean(noise*noise)); y=x+noise
            sid=str(r['id'])+'_'+condition; path=ROOT/'data'/(sid+'.wav'); sf.write(path,y,sr,subtype='PCM_16')
            manifest.append(dict(sample_id=sid,audio_path='data/'+path.name,reference_text=r['text'],language='en',category='speech',source='same public LibriSpeech dummy parquet; post-freeze supplement',speaker_id=str(r['speaker_id']),split='test',original_split='validation',parent_id=str(r['id']),annotation_status='public_transcript' if condition=='short_original' else 'derived_pending_review',condition=condition,synthetic=condition!='short_original'))
        (ROOT/'data/original'/(str(r['id'])+'.flac')).write_bytes(r['audio']['bytes'])
    path=ROOT/'manifest.stress.jsonl'; path.write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in manifest),encoding='utf-8')
    out=ROOT/'results/vad_stress.jsonl'
    subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'run','--manifest',str(path),'--split','test','--vad-only','--config',str(ROOT/'config.vad_frozen.json'),'--output',str(out)],check=True)
    write_json(ROOT/'results/vad_stress_summary.json',dict(**stats(out),scope='Post-freeze diagnostic only; altered short samples pending listening review; excluded from core test and formal ASR'))
if __name__=='__main__': main()
