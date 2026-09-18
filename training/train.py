from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
from model.config import XENConfig
from model.model import XENModel
from model.tokenizer import XENTokenizer

SYSTEM_PROMPT_PATH = Path("configs/system_prompt_t.txt")

def load_system_prompt():
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

def load_examples(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            item=json.loads(line)
            if "instruction" not in item or "response" not in item:
                raise ValueError(f"Line {line_no}: expected instruction and response")
            rows.append((str(item["instruction"]).strip(),str(item["response"]).strip()))
    if not rows: raise ValueError("Dataset is empty")
    return rows

class XENDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_len):
        system_prompt = load_system_prompt()
        self.items=[]
        for q,r in examples:
            text=f"System: {system_prompt}\nUser: {q}\nAssistant: {r}"
            ids=tokenizer.encode(text,max_length=max_seq_len)
            if len(ids)>=2: self.items.append(torch.tensor(ids,dtype=torch.long))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        ids=self.items[i]
        return ids[:-1],ids[1:]

def collate(batch):
    max_len=max(x[0].numel() for x in batch)
    xs=[]; ys=[]
    for x,y in batch:
        pad=max_len-x.numel()
        xs.append(torch.nn.functional.pad(x,(0,pad),value=0))
        ys.append(torch.nn.functional.pad(y,(0,pad),value=-100))
    return torch.stack(xs),torch.stack(ys)

def load_cfg(path):
    with open(path,encoding="utf-8") as f: return yaml.safe_load(f)

def evaluate(model,loader,device,amp_dtype):
    model.eval(); total=0.0; count=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            with torch.autocast(device_type="cuda",dtype=amp_dtype,enabled=device.type=="cuda"):
                _,loss=model(x,y)
            total+=float(loss); count+=1
    model.train(); return total/max(1,count)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="datasets/train.jsonl")
    p.add_argument("--output",default="outputs/xen")
    p.add_argument("--config",default="configs/train.yaml")
    p.add_argument("--resume",default=None)
    a=p.parse_args()
    cfgd=load_cfg(a.config); seed=int(cfgd.get("seed",42)); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    examples=load_examples(a.data)
    tokenizer=XENTokenizer(); tokenizer.fit(vocab_size=int(cfgd["vocab_size"]))
    dataset=XENDataset(examples,tokenizer,int(cfgd["max_seq_len"]))
    val_n=max(1,int(len(dataset)*float(cfgd.get("validation_split",0.02)))) if len(dataset)>1 else 0
    train_n=len(dataset)-val_n
    train_set,val_set=random_split(dataset,[train_n,val_n],generator=torch.Generator().manual_seed(seed)) if val_n else (dataset,None)
    workers=min(4,max(0,(torch.get_num_threads()//2)))
    train_loader=DataLoader(train_set,batch_size=int(cfgd["batch_size"]),shuffle=True,collate_fn=collate,num_workers=workers,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val_set,batch_size=int(cfgd["batch_size"]),shuffle=False,collate_fn=collate,num_workers=workers,pin_memory=torch.cuda.is_available()) if val_set else None
    model_cfg=XENConfig(vocab_size=int(cfgd["vocab_size"]),max_seq_len=int(cfgd["max_seq_len"]),d_model=int(cfgd["d_model"]),n_heads=int(cfgd["n_heads"]),n_layers=int(cfgd["n_layers"]),ffn_mult=float(cfgd.get("ffn_mult",2.6666666667)),dropout=float(cfgd.get("dropout",0.0)),rope_theta=float(cfgd.get("rope_theta",10000.0)))
    model=XENModel(model_cfg)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfgd["learning_rate"]),weight_decay=float(cfgd.get("weight_decay",0.1)),betas=(0.9,0.95))
    total_steps=max(1,math.ceil(len(train_loader)*int(cfgd["epochs"])/int(cfgd["gradient_accumulation"])))
    warmup=int(cfgd.get("warmup_steps",200))
    def lr_lambda(step):
        if step<warmup: return max(1e-8,(step+1)/max(1,warmup))
        progress=(step-warmup)/max(1,total_steps-warmup)
        return 0.5*(1+math.cos(math.pi*min(1.0,progress)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    amp_dtype=torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); tokenizer.save(out/"tokenizer.json")
    start_epoch=0; global_step=0; best_val=float("inf")
    if a.resume:
        ckpt=torch.load(a.resume,map_location=device)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt.get("scaler",{})); start_epoch=int(ckpt.get("epoch",0)); global_step=int(ckpt.get("global_step",0)); best_val=float(ckpt.get("best_val",float("inf")))
        print(f"Resumed checkpoint: epoch={start_epoch} step={global_step}")
    model.train(); accum=int(cfgd["gradient_accumulation"]); log_every=int(cfgd.get("log_every",20)); save_every=int(cfgd.get("save_every",500)); running=0.0; seen=0; t0=time.time(); optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch,int(cfgd["epochs"])):
        for batch_idx,(x,y) in enumerate(train_loader):
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=amp_dtype,enabled=device.type=="cuda"):
                _,loss=model(x,y); scaled_loss=loss/accum
            scaler.scale(scaled_loss).backward(); running+=float(loss); seen+=1
            if (batch_idx+1)%accum==0 or batch_idx+1==len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfgd.get("max_grad_norm",1.0)))
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step+=1
                if global_step%log_every==0:
                    elapsed=max(1e-6,time.time()-t0); print(f"epoch={epoch+1} step={global_step} loss={running/seen:.4f} lr={scheduler.get_last_lr()[0]:.2e} samples/s={seen*int(cfgd['batch_size'])/elapsed:.1f}"); running=0.0; seen=0; t0=time.time()
                if global_step%save_every==0:
                    torch.save({"config":model_cfg.__dict__,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"global_step":global_step,"best_val":best_val},out/f"checkpoint-{global_step}.pt")
        val=evaluate(model,val_loader,device,amp_dtype) if val_loader else float("nan")
        print(f"epoch={epoch+1} validation_loss={val:.4f}")
        torch.save({"config":model_cfg.__dict__,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch+1,"global_step":global_step,"best_val":best_val},out/"latest.pt")
        if val_loader and val<best_val:
            best_val=val
            torch.save({"config":model_cfg.__dict__,"model":model.state_dict()},out/"model.pt")
            print("Saved new best XEN model")
    if not val_loader: torch.save({"config":model_cfg.__dict__,"model":model.state_dict()},out/"model.pt")
    print(f"Training complete. Device={device}. Output={out}")

if __name__=="__main__": main()
