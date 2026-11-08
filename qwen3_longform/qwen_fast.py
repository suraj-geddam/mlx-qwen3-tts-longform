"""Speed-ups for Qwen3-TTS cloning that leave the output unchanged.

RefPrefixDecoder: every cloned chunk is decoded as [reference codes + new codes] so the new audio starts
with the right acoustic left context. The codec decoder is causal (causal convolutions, causal attention),
so the decoder state after the reference is the same for every chunk. Decode the reference once in
streaming mode, snapshot that state (conv buffers, transpose-conv overlap, attention KV cache), and resume
from the snapshot for each chunk: the result equals one unchunked pass over [reference + chunk], while only
the chunk's own frames are decoded.
"""
import copy
import mlx.core as mx


def _stateful(dec):
    for name, m in dec.named_modules():
        for attr in ("_buffer", "_overflow"):
            if hasattr(m, attr):
                yield m, attr


class RefPrefixDecoder:
    def __init__(self, speech_tokenizer, ref_codes, step=300):
        """ref_codes: [1, num_quantizers, R] (as mlx-audio's ICL prep returns them)."""
        # mlx-audio wraps the decoder in mx.compile; the wrapper hides underscore attributes, so reach the
        # module itself through a bound method
        self.dec = speech_tokenizer.decoder.streaming_step.__self__
        self.up = speech_tokenizer.decode_upsample_rate
        self.step = step
        self.dec.reset_streaming_state()
        for s in range(0, ref_codes.shape[2], step):
            mx.eval(self.dec.streaming_step(ref_codes[:, :, s:s + step]),
                    [getattr(m, a) for m, a in _stateful(self.dec)],
                    [(c.keys, c.values) for c in self.dec._transformer_cache])
        # arrays are rebound (never written in place) by the conv steps, so references are enough
        self.bufs = [(m, a, getattr(m, a)) for m, a in _stateful(self.dec)]
        self.kv = []
        for c in self.dec._transformer_cache:
            k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
            mx.eval(k, v)
            self.kv.append((c, k, v, c.offset))

    def _restore(self):
        for m, a, v in self.bufs:
            setattr(m, a, v)
        cache = []
        for c, k, v, off in self.kv:
            n = copy.copy(c)  # cache sized exactly to offset: the next update reallocates, never writes into k/v
            n.keys, n.values, n.offset = k, v, off
            cache.append(n)
        self.dec._transformer_cache = cache

    def decode(self, codes):
        """codes: [1, T, num_quantizers] or [T, num_quantizers] generated frames. Returns float32 [T*upsample]."""
        if codes.ndim == 2:
            codes = codes[None]
        x = mx.transpose(codes, (0, 2, 1))
        self._restore()
        parts = []
        for s in range(0, x.shape[2], self.step):
            parts.append(self.dec.streaming_step(x[:, :, s:s + self.step]))
            # evaluate the step and the state it hands on before building the next: left lazy, all steps run as
            # one graph and the step size no longer bounds memory (the carried state slices keep whole
            # activations alive)
            mx.eval(parts[-1], [getattr(m, a) for m, a in _stateful(self.dec)],
                    [(c.keys, c.values) for c in self.dec._transformer_cache])
        wav = mx.concatenate(parts, axis=-1)[0, 0] if len(parts) > 1 else parts[0][0, 0]
        return wav


def patch_prefix_decode(model, step=50):
    """Route mlx-audio's ICL decode (generate and batch_generate) through a RefPrefixDecoder, built on first use
    for each reference and rebuilt if the reference changes. `step` is how many frames the decoder takes at a time
    (12.5 frames per second). The output is the same for any step; the decoder's working memory grows with it
    (on an M3: ~1.6 GB at 25 frames, ~2.5 GB at 50, ~4 GB at 300) while small steps run a little slower."""
    cache = {}

    def dec(generated_codes, ref_codes):
        if not generated_codes:
            return mx.zeros((0,), dtype=mx.float32)
        key = (ref_codes.shape, float(ref_codes.sum()))
        if key not in cache:
            cache.clear(); cache[key] = RefPrefixDecoder(model.speech_tokenizer, ref_codes, step=step)
        out = cache[key].decode(mx.stack(generated_codes, axis=1))
        mx.async_eval(out)
        return out

    model._decode_icl_generated_codes = dec
