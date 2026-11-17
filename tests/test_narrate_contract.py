"""Exercise the CLI, gate and output schema with model inference replaced by fixed audio."""
import json
import sys
import numpy as np
import soundfile as sf
from qwen3_longform import narrate as N


def test_cli_preserves_script_metadata_and_writes_importable_chunks(tmp_path, monkeypatch):
    aliases = {'M I T': 'MIT'}
    lines = [{'ch': 'intro', 'text': 'Display text', 'say': 'M I T made this model'},
             {'ch': 'end', 'text': 'We have reached the end'}]
    script = tmp_path / 'script.json'
    script.write_text(json.dumps({'lines': lines, 'asr_equiv': aliases}))
    ref = tmp_path / 'ref.wav'; ref.with_suffix('.txt').write_text('A reference.')
    out = tmp_path / 'run'
    generated = []
    class Model:
        sr = 24000
        def __init__(self, *args, **kwargs): pass
        def generate(self, texts, *args):
            generated.extend(texts)
            return [np.full(48000, .1, np.float32) for _ in texts]
    monkeypatch.setattr(N, 'Qwen', Model)
    monkeypatch.setattr(N, 'sentence_pause', lambda *a, **kw: .42)
    monkeypatch.setattr(N.M, 'reference_profile', lambda *a: {'emb': np.array([1.]), 'lufs': -20.})
    monkeypatch.setattr(N.M, 'spk_embed', lambda *a: np.array([1.]))
    monkeypatch.setattr(N.M, 'lufs', lambda *a: -20.)
    def transcribe(ys, *args):
        results = []
        for text in generated[-len(ys):]:
            words = text.replace('M I T', 'MIT').split()
            results.append({'text': ' '.join(words), 'segments': [{'words': [
                {'word': w, 'start': i * .3, 'end': (i+1) * .3} for i, w in enumerate(words)]}]})
        return results
    monkeypatch.setattr(N.M, 'transcribe_packed', transcribe)
    analysis = []
    def analyse(*args, **kwargs):
        analysis.append(kwargs['asr_equiv']); return {'summary': {'dur': 4.}}
    monkeypatch.setattr(N.M, 'analyse', analyse)
    monkeypatch.setattr(sys, 'argv', ['qwen3-longform', '--ref', str(ref), '--script', str(script),
                                    '--out', str(out), '--no-normalize', '--cache-limit', '2', '--analyse'])
    N.main()
    run = json.loads((out / 'run.json').read_text())
    assert run['asr_equiv'] == aliases and run['normalize'] is False
    assert run['cache_limit_gb'] == 2 and run['failed'] == []
    assert analysis == [aliases]
    assert sorted(generated) == sorted(l.get('say', l['text']) for l in lines)
    assert [c['i'] for c in run['chunks']] == [0, 1]
    assert [c['ch'] for c in run['chunks']] == ['intro', 'end']
    assert [c['text'] for c in run['chunks']] == [l.get('say', l['text']) for l in lines]
    for c in run['chunks']:
        assert c['pause_after'] >= 0 and c['pick']['ok'] and c['pick']['wer'] == 0
        info = sf.info(out / f"chunks/{c['i']:03d}.wav")
        assert info.channels == 1 and info.samplerate == 24000 and info.frames > 0
    assert sf.info(out / 'narration.wav').frames > 0
    assert (out / 'metrics.json').exists()
