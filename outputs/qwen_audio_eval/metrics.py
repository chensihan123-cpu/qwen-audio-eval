"""Frozen normalization v1: NFKC, casefold, punctuation=>space, digits unchanged."""
import hashlib, json, unicodedata

def normalize(text, language):
    text=unicodedata.normalize('NFKC',text).casefold()
    text=''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return list(''.join(text.split())) if language=='zh' else text.split()

def parse_output(text):
    return '' if text.strip()=='[NO_SPEECH]' else text.strip()

def counts(ref, hyp):
    # deterministic tie breaking: match/substitute, delete, insert
    dp=[[(j,0,0,j) for j in range(len(hyp)+1)]]
    for i,r in enumerate(ref,1):
        row=[(i,0,i,0)]
        for j,h in enumerate(hyp,1):
            a=dp[i-1][j-1]; b=dp[i-1][j]; c=row[j-1]
            candidates=[(a[0]+(r!=h),a[1]+(r!=h),a[2],a[3]),(b[0]+1,b[1],b[2]+1,b[3]),(c[0]+1,c[1],c[2],c[3]+1)]
            row.append(min(candidates,key=lambda v:v[0]))
        dp.append(row)
    _,s,d,i=dp[-1][-1]
    return dict(S=s,D=d,I=i,N=len(ref))

def cache_key(audio_hash,config):
    return hashlib.sha256(json.dumps({'audio':audio_hash,'config':config},sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def align(manifest, results, group):
    ids=[r['sample_id'] for r in manifest]
    if len(set(ids))!=len(ids): raise ValueError('duplicate manifest sample_id')
    chosen=[r for r in results if r['group']==group]
    pred={r['sample_id']:r for r in chosen}
    if len(pred)!=len(chosen): raise ValueError('duplicate result sample_id')
    if set(pred)!=set(ids): raise ValueError('sample ID mismatch: missing or extra results')
    return [(m,pred[m['sample_id']]) for m in manifest]

def quality(rows):
    total=dict(S=0,D=0,I=0,N=0)
    for m,r in rows:
        # Errors and VAD rejection count as empty, never removed from denominator.
        hyp=r.get('scoring_text','') if r['status'] in ('ok','model_empty','vad_skip') else ''
        c=counts(normalize(m['reference_text'],m['language']),normalize(hyp,m['language']))
        for k in total: total[k]+=c[k]
    return dict(**total,error_rate=(total['S']+total['D']+total['I'])/total['N'] if total['N'] else None)
