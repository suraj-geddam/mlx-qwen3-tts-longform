"""Optional: the script the author used to make a reference voice to clone, with Gemini TTS (bring your own key).

    GEMINI_API_KEY=... uv run python scripts/make_ref_gemini.py [Voice] [--takes 2] [--pace paced|brisk]

Writes refs/<voice>_<n>.wav (24 kHz mono) and refs/<voice>_<n>.txt (its transcript), ready for --ref.
Listen to the takes and keep the best one; any clean recording of a voice you have the rights to works as well.

- The default text (scripts/reference_text.txt) covers all 41 phonemes of British English, so the clone hears
  every sound at least once.
- Qwen3-TTS copies the reference's pacing along with its timbre: a brisk reference makes the clone run
  sentences together. The default direction asks for an unhurried read with clear pauses after full stops.
- Gemini voices: https://ai.google.dev/gemini-api/docs/speech-generation (the author used Alnilam). Check
  Google's terms for your use of the generated audio.
"""
import argparse, base64, io, json, os, urllib.error, urllib.request
import numpy as np, soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
DIRECTIONS = {
    "paced": ("Read this as a warm, engaging British documentary narrator: a man with a natural southern English "
              "accent, like a seasoned BBC presenter. Unhurried, measured pace, about 150 words per minute. After "
              "every full stop and question mark, take a clear pause of about half a second, letting the sentence "
              "ending fall and land before the next sentence begins. Take a shorter, audible pause at each comma, "
              "colon and dash. Conversational and curious, not theatrical."),
    "brisk": ("Read this as a warm, engaging British documentary narrator: a man with a natural southern English "
              "accent, like a seasoned BBC presenter. Moderately brisk pace, about 165 words per minute. Pause "
              "naturally at commas and colons, take a clear breath-length pause at every full stop, and let each "
              "sentence ending fall and land. Conversational and curious, not theatrical."),
}

ap = argparse.ArgumentParser()
ap.add_argument("voice", nargs="?", default="Alnilam", help="Gemini prebuilt voice name")
ap.add_argument("--text", default=f"{ROOT}/scripts/reference_text.txt")
ap.add_argument("--pace", choices=sorted(DIRECTIONS), default="paced")
ap.add_argument("--direction", help="your own reading direction (overrides --pace)")
ap.add_argument("--takes", type=int, default=2)
ap.add_argument("--out-dir", default=f"{ROOT}/refs")
a = ap.parse_args()

key = os.environ.get("GEMINI_API_KEY")
if not key:
    raise SystemExit("Set GEMINI_API_KEY (https://aistudio.google.com/apikey).")
text = open(a.text).read().strip()
prompt = f"### DIRECTOR'S NOTES\n{a.direction or DIRECTIONS[a.pace]}\n\n#### TRANSCRIPT\n{text}"
os.makedirs(a.out_dir, exist_ok=True)
for take in range(1, a.takes + 1):
    name = f"{a.voice.lower()}_{take}"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": a.voice}}}}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                                 data=json.dumps(body).encode(),
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Gemini API error {e.code}: {e.read()[:300]!r}")
    part = d["candidates"][0]["content"]["parts"][0]["inlineData"]
    raw = base64.b64decode(part["data"])
    if raw[:4] == b"RIFF":
        pcm, sr = sf.read(io.BytesIO(raw), dtype="float32")
    else:  # raw 16-bit PCM
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        mt = part.get("mimeType", "")
        sr = int(mt.split("rate=")[-1].split(";")[0]) if "rate=" in mt else 24000
    sf.write(f"{a.out_dir}/{name}.wav", pcm, sr)
    open(f"{a.out_dir}/{name}.txt", "w").write(text)
    print(f"{a.out_dir}/{name}.wav  {len(pcm)/sr:.1f}s", flush=True)
