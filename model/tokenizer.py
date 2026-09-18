from __future__ import annotations
from pathlib import Path
import json

class XENTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
        self.inverse = {v:k for k,v in self.vocab.items()}

    def fit(self, texts, vocab_size=32000):
        counts = {}
        for text in texts:
            for token in text.split():
                counts[token] = counts.get(token, 0) + 1
        for token, _ in sorted(counts.items(), key=lambda x:(-x[1],x[0])):
            if token not in self.vocab and len(self.vocab) < vocab_size:
                self.vocab[token] = len(self.vocab)
        self.inverse = {v:k for k,v in self.vocab.items()}

    def encode(self, text, max_length=None):
        ids = [self.vocab["<bos>"]] + [self.vocab.get(t,3) for t in text.split()] + [self.vocab["<eos>"]]
        if max_length:
            ids = ids[:max_length]
            if ids and ids[-1] != self.vocab["<eos>"]:
                ids[-1] = self.vocab["<eos>"]
        return ids

    def decode(self, ids):
        return " ".join(self.inverse.get(i,"<unk>") for i in ids if self.inverse.get(i) not in {"<pad>","<bos>","<eos>"})

    def save(self, path):
        Path(path).write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
