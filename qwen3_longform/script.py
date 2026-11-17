"""Scripts and chunking.

A script is either
  * plain text (.txt, .md): paragraphs separated by blank lines; a line starting with "#" starts a new chapter
    (the heading itself is not read aloud), or
  * JSON: {"lines": [{"ch": "<chapter id>", "text": "...", "say": "<optional spoken form>"}, ...]}. "say" is
    what gets read and scored when it differs from the written text (e.g. a respelled name).
    An optional top-level asr_equiv maps equivalent spellings for checks only.

The text is normalized for speech first (normalize.py): "$1.87" -> "1 dollar and 87 cents", "Dr." -> "Doctor",
"50%" -> "50 percent", parentheses -> commas. Plain numbers are left for the model, which reads them in
context. normalize=False leaves the text as written.

Chunks are runs of whole sentences of about 90-150 characters (~6-10 s of speech), never spanning a chapter.
Each chunk is read with a lead-in (the end of the previous chunk) and a guard (the start of the next), which are
cut off afterwards so the kept audio starts and ends with natural prosody.
"""
import json, os, re


def load_script(path, normalize=True):
    """Returns the script's lines; use load_script_document to retain comparison metadata."""
    return load_script_document(path, normalize)["lines"]


def load_script_document(path, normalize=True):
    """Read lines and optional asr_equiv without applying equivalences to spoken text."""
    if os.path.splitext(path)[1].lower() == ".json":
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        lines = doc["lines"]
        equiv = doc.get("asr_equiv", {})
        if not isinstance(equiv, dict) or any(
                not isinstance(k, str) or not k.strip() or
                not isinstance(v, str) or not v.strip() for k, v in equiv.items()):
            raise ValueError("asr_equiv must be an object of nonempty spelling strings")
    else:
        lines, ch, equiv = [], 0, {}
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for block in re.split(r"\n\s*\n", text):
            rows = [r.strip() for r in block.strip().splitlines() if r.strip()]
            while rows and rows[0].startswith("#"):
                ch += 1; rows = rows[1:]
            if rows:
                lines.append({"ch": ch, "text": " ".join(rows)})
    if normalize:
        from .normalize import TextNormalizer
        n = TextNormalizer()
        lines = [dict(l, say=n.normalize(l.get("say", l["text"]))) for l in lines]
    return {"lines": lines, "asr_equiv": dict(equiv)}


def sentences(text):
    # split after . ! ? before a capital, but not after an initial ("J. Smith", "unassuming D. But")
    return [s for s in re.split(r'(?<=[.!?])(?<!\b[A-Z]\.)\s+(?=[A-Z"“‘\'])', text.strip()) if s]


def make_chunks(lines, lo=90, hi=150):
    """Pack the script's sentences into chunks of about lo..hi characters, never across a chapter. Long
    sentences are split at ; : , or a dash near hi. Returns [{text, ch, end_of_ch}]."""
    chunks = []
    for ch in dict.fromkeys(l["ch"] for l in lines):
        sents = []
        for l in (l for l in lines if l["ch"] == ch):
            for s in sentences(l.get("say", l["text"])):
                while len(s) > hi + 40:
                    cut = max((m.end() for m in re.finditer(r"[,;:—]\s", s[:hi + 20]) if m.end() > lo * 0.6), default=None)
                    if cut is None:
                        break
                    sents.append(s[:cut].strip()); s = s[cut:].strip()
                sents.append(s)
        cur = ""
        for s in sents:
            if cur and len(cur) + 1 + len(s) > hi and len(cur) >= lo * 0.5:
                chunks.append({"text": cur, "ch": ch}); cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            if chunks and chunks[-1]["ch"] == ch and len(cur) < 50 and len(chunks[-1]["text"]) + len(cur) < hi + 60:
                chunks[-1]["text"] += " " + cur
            else:
                chunks.append({"text": cur, "ch": ch})
    for i, c in enumerate(chunks):
        c["end_of_ch"] = i == len(chunks) - 1 or chunks[i + 1]["ch"] != c["ch"]
    return chunks


def clause_tail(text, max_chars=80):
    """Last sentence of text, or its last clauses if the sentence is long (the lead-in)."""
    last = sentences(text)[-1]
    if len(last) <= max_chars:
        return last
    parts = re.split(r"(?<=[,;:—])\s+", last)
    tail = parts[-1]
    for p in reversed(parts[:-1]):
        if len(p) + 1 + len(tail) > max_chars:
            break
        tail = p + " " + tail
    return tail


def clause_head(text, max_chars=80):
    """First sentence of text, or its first clause if long (the guard)."""
    first = sentences(text)[0]
    if len(first) <= max_chars:
        return first
    m = re.search(r"[,;:—]\s", first[20:max_chars + 20])
    return first[: 20 + m.end()].strip() if m else first


def overlap_jobs(chunks):
    """What to generate for each chunk: lead-in (none at a chapter start), the chunk, guard (none at a chapter
    end)."""
    jobs = []
    for i, c in enumerate(chunks):
        prev = chunks[i - 1] if i > 0 and chunks[i - 1]["ch"] == c["ch"] else None
        nxt = chunks[i + 1]["text"] if i + 1 < len(chunks) else None
        lead = clause_tail(prev["text"]) if prev else ""
        guard = clause_head(nxt) if (nxt and not c["end_of_ch"]) else ""
        jobs.append(dict(i=i, lead=lead, text=c["text"], guard=guard,
                         full=" ".join(x for x in (lead, c["text"], guard) if x)))
    return jobs
