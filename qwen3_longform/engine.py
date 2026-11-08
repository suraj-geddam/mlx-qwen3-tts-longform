"""Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0) in bf16 through mlx-audio, set up for cloning one reference."""
import numpy as np
import mlx.core as mx
from mlx_audio.tts.utils import load_model

from .metrics import load
from .qwen_fast import patch_prefix_decode

REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
SR = 24000


class Qwen:
    """m: the mlx-audio model, its ICL decode routed through the exact prefix-cached decoder.
    ref: the reference audio at 24 kHz (float32), ref_text: its transcript."""

    def __init__(self, ref_wav, ref_text, decode_step=50):
        self.m = load_model(REPO)
        patch_prefix_decode(self.m, step=decode_step)
        self.sr = SR
        self.ref, _ = load(ref_wav, SR)
        self.ref_text = ref_text

    def generate(self, texts, seed, max_tokens, temperature=0.7, top_p=0.9):
        """Clone the reference reading each of `texts`, all in one batch. Returns float32 arrays in order."""
        mx.random.seed(seed)
        ys = [None] * len(texts)
        for r in self.m.batch_generate(texts, ref_audio=mx.array(self.ref), ref_text=self.ref_text,
                                       lang_code="English", temperature=temperature, top_p=top_p,
                                       max_tokens=max_tokens):
            ys[r.sequence_idx] = np.asarray(r.audio, np.float32).reshape(-1)
        return [y if y is not None else np.zeros(0, np.float32) for y in ys]


def max_tokens_for(text):
    """Generation cap: about 2.2x the frames the text should take at ~15 characters per second."""
    return int(min(1500, max(60, len(text) / 15 * 12.5 * 2.2)))
