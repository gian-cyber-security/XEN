from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from model.config import XENConfig
from model.model import XENModel
from model.tokenizer import XENTokenizer

def load_examples(path):
    with open(path,encoding="utf-8") as f:
        return [(x["instruction"].strip(),x["response"].strip()) for line in f if line.strip() for x in [json.loads(line)]]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="datasets/train.jsonl")
    p.add_argument("--output",default="outputs/xen"); p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--lr",type=float,default=3e-4); a=p.parse_args()
    examples=load_examples(a.data)
    texts=[f"User: {q} Assistant: {r}" for q,r in examples]
    tokenizer=XENTokenizer(); tokenizer.fit(texts)
    cfg=XENConfig(vocab_size=max(32000,len(tokenizer.vocab))); model=XENModel(cfg)
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr); model.train()
    for epoch in range(a.epochs):
        total=0.0
        for text in texts:
            ids=tokenizer.encode(text,max_length=cfg.max_seq_len)
            x=torch.tensor([ids[:-1]],dtype=torch.long,device=device)
            y=torch.tensor([ids[1:]],dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True); _,loss=model(x,y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total+=float(loss)
        print(f"epoch={epoch+1} loss={total/max(1,len(texts)):.4f}")
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    torch.save({"config":cfg.__dict__,"model":model.state_dict()},out/"model.pt")
    tokenizer.save(out/"tokenizer.json")
    print(f"Saved XEN from-scratch model to {out}")

if __name__=="__main__": main()
