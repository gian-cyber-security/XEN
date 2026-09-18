from dataclasses import dataclass

@dataclass
class XENConfig:
    vocab_size: int = 260  # 256 UTF-8 byte values + 4 special tokens
    max_seq_len: int = 512
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    ffn_mult: float = 2.6666666667
    dropout: float = 0.0
    rope_theta: float = 10000.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
