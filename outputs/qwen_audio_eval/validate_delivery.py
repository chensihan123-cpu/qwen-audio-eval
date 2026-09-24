"""Check real artifacts and CLI guardrails without simulated model responses."""
import json, subprocess, sys
from pathlib import Path
from pipeline import ROOT, read_lines, read_json, write_json

def main():
    exe=sys.executable; runner=str(ROOT/'pipeline.py'); result=ROOT/'results/abcd_dev_final.jsonl'
    before=len(read_lines(result))
    checks={}
    for name,argv in {
        'changed_config_resume_rejected':['run','--split','dev','--config',str(ROOT/'config.dev.0.3.json'),'--resume','--output',str(result)],
        'unfrozen_test_rejected':['run','--split','test','--output',str(ROOT/'results/should_not_exist.jsonl')],
        'vad_only_freeze_cannot_unlock_qwen_test':['run','--split','test','--config',str(ROOT/'config.vad_frozen.json'),'--output',str(ROOT/'results/should_not_exist.jsonl')],
        'performance_resume_rejected':['run','--split','dev','--performance','--resume','--output',str(result)]
    }.items():
        p=subprocess.run([exe,runner]+argv,capture_output=True,text=True)
        checks[name]={'passed':p.returncode!=0,'returncode':p.returncode,'stderr':p.stderr}
        assert p.returncode!=0,name
    assert len(read_lines(result))==before==32
    assert all(r['raw_output'] is None for r in read_lines(result))
    manifest=read_lines(ROOT/'manifest.jsonl')
    assert len(manifest)==24
    checks['artifact_integrity']={'passed':True,'dev_abcd_records':before,'smoke_samples':len(manifest),'real_model_predictions':0}
    write_json(ROOT/'evidence/delivery_checks.json',checks)
    print('All delivery guards passed')
if __name__=='__main__': main()
