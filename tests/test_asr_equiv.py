"""Script-local aliases must help spelling checks without hiding unrelated errors."""
import json
import numpy as np
import pytest
from qwen3_longform import metrics as M
from qwen3_longform.script import load_script, load_script_document, make_chunks
from qwen3_longform.takes import check, cut_span


def test_document_preserves_spoken_text_and_normalization_choice(tmp_path):
    path = tmp_path / 'script.json'
    lines = [{'id': 'a', 'ch': 'intro', 'text': 'Display text', 'say': 'Dr. Lee paid $5.'}]
    aliases = {'M I T': 'MIT'}
    path.write_text(json.dumps({'lines': lines, 'asr_equiv': aliases}))
    raw = load_script_document(path, normalize=False)
    assert raw == {'lines': lines, 'asr_equiv': aliases}
    normalized = load_script_document(path)
    assert normalized['asr_equiv'] == aliases
    assert normalized['lines'][0]['say'] == 'Doctor Lee paid 5 dollars.'
    assert load_script(path, False) == lines  # existing list-returning API
    assert make_chunks(raw['lines'])[0]['text'] == lines[0]['say']


@pytest.mark.parametrize('value', [None, [], {'MIT': 3}, {'': 'MIT'}, {'MIT': ''}])
def test_invalid_script_equivalences_fail_early(tmp_path, value):
    path = tmp_path / 'script.json'
    path.write_text(json.dumps({'lines': [], 'asr_equiv': value}))
    with pytest.raises(ValueError, match='asr_equiv'):
        load_script_document(path)


def test_equivalences_are_symmetric_whole_phrases_and_do_not_leak(tmp_path):
    aliases = {'M I T': 'MIT', 'data set': 'dataset', 'Saqib': 'Sakib'}
    a = 'Saqib made a data set at M I T.'
    b = 'Sakib made a dataset at MIT.'
    assert M.wer_ops(a, b, aliases)[0] == 0
    assert M.wer_ops(b, a, aliases)[0] == 0
    assert M.wer_ops(a, b)[0] > 0
    assert M.wer_ops('Saqib made a dataset', 'Sakib lost a dataset', aliases)[0] > 0
    assert M.toks('MITten', aliases) == ['mitten']
    assert M.toks('a b c', {'a b': 'x', 'a b c': 'y'}) == ['y']
    assert M.toks('alpha', {'alpha': 'beta', 'beta': 'gamma'}) == ['beta']
    path = tmp_path / 'script.txt'; path.write_text('# Intro\n\nA plain script.')
    assert load_script_document(path)['asr_equiv'] == {}


def test_aliases_retain_timestamps_across_whisper_entries():
    words = [('M', .1, .2), ('I', .3, .4), ('T', .5, .6), ('arrived', .7, .9)]
    assert M.transcript_tokens(words, {'M I T': 'MIT'}) == [('mit', .1, .6), ('arrived', .7, .9)]
    assert M.transcript_tokens([('MIT', .1, .6)], {'MIT': 'M I T'}) == [
        ('m', .1, .6), ('i', .1, .6), ('t', .1, .6)]


def test_alias_at_cut_boundary_is_not_lost():
    sr = 1000
    y = np.zeros(2000, np.float32)
    y[100:300] = .2; y[800:1200] = .2; y[1600:1800] = .2
    words = [('intro', .1, .3), ('M', .8, .9), ('I', .9, 1.), ('T', 1., 1.2), ('end', 1.6, 1.8)]
    assert cut_span(y, sr, 'intro', 'MIT', 'end', words)[0] is None
    audio, pause, bounds = cut_span(y, sr, 'intro', 'MIT', 'end', words, {'M I T': 'MIT'})
    assert audio is not None and pause > 0
    assert bounds[0] < 800 and bounds[1] >= 1200


def test_real_gate_accepts_alias_but_still_rejects_wrong_word(monkeypatch):
    monkeypatch.setattr(M, 'spk_embed', lambda y: np.array([1.]))
    monkeypatch.setattr(M, 'lufs', lambda y, sr: -20.)
    ref = {'emb': np.array([1.]), 'lufs': -20.}
    y = np.full(48000, .1, np.float32)
    text = 'M I T made this model'
    aliases = {'M I T': 'MIT'}
    hyp = {'text': 'MIT made this model'}
    assert not check(y, 24000, text, ref, res=hyp)['ok']
    assert check(y, 24000, text, ref, res=hyp, asr_equiv=aliases)['ok']
    assert not check(y, 24000, text, ref, res={'text': 'MIT broke this model'}, asr_equiv=aliases)['ok']


def test_punctuation_only_alias_is_rejected():
    with pytest.raises(ValueError, match='comparison tokens'):
        M.toks('', {'...': 'MIT'})
