from __future__ import annotations
import torch
from .config import XENConfig
from .model import XENModel
from .tokenizer import XENTokenizer

def load_model(model_dir="outputs/xen"):
    checkpoint=torch.load(f"{model_dir}/model.pt",map_location="cpu")
    cfg=XENConfig(**checkpoint["config"])
    model=XENModel(cfg); model.load_state_dict(checkpoint["model"])
    tokenizer=XENTokenizer.load(f"{model_dir}/tokenizer.json")
    model.eval()
    return model,tokenizer
