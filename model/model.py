from __future__ import annotations
import math
import torch
from torch import nn
from .config import XENConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = nn.Linear(cfg.d_model, 3*cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.register_buffer("mask", torch.tril(torch.ones(cfg.max_seq_len,cfg.max_seq_len)).view(1,1,cfg.max_seq_len,cfg.max_seq_len), persistent=False)

    def forward(self,x):
        b,t,c=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        k=k.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        v=v.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        scores=scores.masked_fill(self.mask[:,:,:t,:t]==0,float("-inf"))
        return self.out((torch.softmax(scores,dim=-1)@v).transpose(1,2).contiguous().view(b,t,c))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg)
        self.ln2=nn.LayerNorm(cfg.d_model)
        self.mlp=nn.Sequential(nn.Linear(cfg.d_model,4*cfg.d_model),nn.GELU(),nn.Linear(4*cfg.d_model,cfg.d_model))
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        return x+self.mlp(self.ln2(x))

class XENModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.token_embedding=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.position_embedding=nn.Embedding(cfg.max_seq_len,cfg.d_model)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln=nn.LayerNorm(cfg.d_model)
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.lm_head.weight=self.token_embedding.weight
    def forward(self,input_ids,targets=None):
        b,t=input_ids.shape
        if t>self.cfg.max_seq_len: raise ValueError("Sequence exceeds XEN max_seq_len")
        pos=torch.arange(t,device=input_ids.device)
        x=self.token_embedding(input_ids)+self.position_embedding(pos)[None,:,:]
        for block in self.blocks: x=block(x)
        logits=self.lm_head(self.ln(x))
        loss=None
        if targets is not None:
            loss=nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-100)
        return logits,loss
