from dataclasses import dataclass

@dataclass
class XENConfig:
    vocab_size: int = 32000
    max_seq_len: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
