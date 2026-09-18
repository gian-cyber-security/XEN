from __future__ import annotations
from pathlib import Path
import json

class XENTokenizer:
    """Small dependency-free byte tokenizer built from scratch.

    Every UTF-8 byte is representable, so XEN has no normal-language OOV problem.
    Token ids 4..259 map to byte values 0..255.
    """
    def __init__(self):
        self.vocab_size = 260
        self.special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def fit(self, texts=None, vocab_size=260):
        if vocab_size != 260:
            raise ValueError("XEN byte tokenizer requires vocab_size=260")

    def encode(self, text, max_length=None):
        ids = [1] + [4 + b for b in text.encode("utf-8")] + [2]
        if max_length is not None:
            ids = ids[:max_length]
            if ids and ids[-1] != 2:
                ids[-1] = 2
        return ids

    def decode(self, ids):
        raw = bytes(max(0, i - 4) for i in ids if 4 <= i <= 259)
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        Path(path).write_text(json.dumps({"type": "byte", "vocab_size": 260}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "byte":
            raise ValueError("Tokenizer file is not a XEN byte tokenizer")
        return cls()
