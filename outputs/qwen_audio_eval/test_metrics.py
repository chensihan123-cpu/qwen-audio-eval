import copy, unittest, tempfile, json
from pathlib import Path
import numpy as np
import soundfile as sf
from metrics import counts, normalize, parse_output, cache_key, align, quality
from pipeline import load_manifest, audio_check

class MetricsTests(unittest.TestCase):
    def test_sdi(self):
        self.assertEqual(counts(['a','b'],['a','c']),dict(S=1,D=0,I=0,N=2))
        self.assertEqual(counts(['a','b'],['a']),dict(S=0,D=1,I=0,N=2))
        self.assertEqual(counts(['a'],['a','b']),dict(S=0,D=0,I=1,N=1))
    def test_empty_reference(self):
        self.assertEqual(counts([],['noise']),dict(S=0,D=0,I=1,N=0))
        self.assertIsNone(quality([])['error_rate'])
    def test_chinese(self):
        self.assertEqual(counts(normalize('你好，世界！','zh'),normalize('你好世','zh')),dict(S=0,D=1,I=0,N=4))
    def test_skip_and_failure(self):
        for status in ['vad_skip','model_unavailable','inference_error']:
            q=quality([({'reference_text':'one two','language':'en'},{'status':status,'scoring_text':''})])
            self.assertEqual(q['D'],2); self.assertEqual(q['N'],2)
    def test_alignment(self):
        m=[{'sample_id':'a'},{'sample_id':'b'}]
        r=[{'sample_id':'b','group':'A'},{'sample_id':'a','group':'A'}]
        self.assertEqual([x[1]['sample_id'] for x in align(m,r,'A')],['a','b'])
        with self.assertRaises(ValueError): align(m,r[:1],'A')
        with self.assertRaises(ValueError): align(m,r+r[:1],'A')
    def test_cache_invalidation(self):
        cfg={'revision':'abc','precision':'bf16','preprocess':16000,'prompt':'a','vad':0.5,'generation':256}
        first=cache_key('audio1',cfg)
        self.assertNotEqual(first,cache_key('audio2',cfg))
        for k in cfg:
            other=copy.deepcopy(cfg); other[k]='changed'
            self.assertNotEqual(first,cache_key('audio1',other))
        self.assertEqual(first,cache_key('audio1',dict(reversed(list(cfg.items())))))
    def test_marker(self):
        self.assertEqual(parse_output(' [NO_SPEECH]\n'),'')
        self.assertEqual(parse_output('There is [NO_SPEECH]'),'There is [NO_SPEECH]')
    def test_corpus_not_sentence_mean(self):
        q=quality([({'reference_text':'a b c','language':'en'},{'status':'ok','scoring_text':'a b c'}),({'reference_text':'d','language':'en'},{'status':'vad_skip','scoring_text':''})])
        self.assertEqual(q['error_rate'],0.25)
    def test_parent_split_leak(self):
        row=dict(sample_id='x',audio_path='x.wav',reference_text='hi',language='en',category='speech',source='fixture',split='dev',parent_id='parent',annotation_status='public_transcript')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'m.jsonl'; p.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,sample_id='y',split='test')),encoding='utf-8')
            with self.assertRaises(ValueError): load_manifest(p)
    def test_audio_validation_and_long_routing(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.wav'; sf.write(p,np.zeros(31*16000),16000)
            x,m=audio_check(p,{'max_duration_s':30}); self.assertIsNone(x); self.assertEqual(m['duration_s'],31)
            p.write_bytes(b'broken')
            with self.assertRaises(Exception): audio_check(p,{'max_duration_s':30})
            with self.assertRaises(FileNotFoundError): audio_check(Path(td)/'missing.wav',{'max_duration_s':30})
if __name__=='__main__': unittest.main(verbosity=2)
