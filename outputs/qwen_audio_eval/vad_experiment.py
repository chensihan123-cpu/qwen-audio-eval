"""Only VAD dev selection/test; never interprets absent Qwen outputs as predictions."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from pipeline import ROOT, read_lines, read_json, write_json

def stats(path):
    rows=read_lines(path)
    speech=[r for r in rows if r['category']=='speech']
    noise=[r for r in rows if r['category']=='non_speech']
    t=[r['timings']['vad_s'] for r in rows]; e=[r['timings']['e2e_s'] for r in rows]
    duration=sum(r['duration_s'] for r in rows)
    return dict(n=len(rows),speech=len(speech),non_speech=len(noise),speech_rejected=sum(r['status']=='vad_skip' for r in speech),non_speech_passed=sum(r['status']=='vad_pass' for r in noise),duration_s=duration,vad_median_ms=float(np.median(t)*1000),vad_p95_ms=float(np.percentile(t,95)*1000),preprocess_plus_vad_median_ms=float(np.median(e)*1000),preprocess_plus_vad_p95_ms=float(np.percentile(e,95)*1000),preprocess_plus_vad_rtf=sum(e)/duration,model_time=None)

def main():
    dev={}
    for suffix,threshold in [('03',0.3),('05',0.5),('07',0.7)]:
        dev[str(threshold)]=stats(ROOT/f'results/vad_dev_{suffix}.jsonl')
    # Predeclared selection: minimize speech rejection, then non-speech pass; ties closest to 0.5.
    threshold=min([0.3,0.5,0.7],key=lambda t:(dev[str(t)]['speech_rejected'],dev[str(t)]['non_speech_passed'],abs(t-.5)))
    cfg=read_json(ROOT/'config.json'); cfg['vad']['threshold']=threshold
    cfg['frozen']=True; cfg['freeze_scope']='VAD-only; B remains provisional; no Qwen test selection'
    write_json(ROOT/'config.vad_frozen.json',cfg)
    write_json(ROOT/'evidence/vad_selection.json',dict(dev=dev,chosen_threshold=threshold,rule='min speech rejects, min non-speech passes, tie nearest 0.5; synthetic dev only',test_not_seen=True))
    out=ROOT/'results/vad_test.jsonl'
    subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'run','--vad-only','--split','test','--config',str(ROOT/'config.vad_frozen.json'),'--output',str(out),'--performance'],check=True)
    write_json(ROOT/'results/vad_summary.json',dict(dev=dev,test=stats(out),scope='VAD CPU ONNX only; excludes model inference; small synthetic non-speech supplements'))
if __name__=='__main__': main()
