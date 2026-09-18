from __future__ import annotations
from functools import lru_cache
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from inference.generate import generate
from model.loader import load_model

app=FastAPI(title="XEN Text-to-Text",version="0.2.0")

class GenerateRequest(BaseModel):
    prompt:str=Field(min_length=1)
    max_new_tokens:int=Field(default=100,ge=1,le=1024)
    temperature:float=Field(default=0.7,ge=0,le=2)

@lru_cache(maxsize=1)
def get_model(): return load_model()

@app.get("/health")
def health(): return {"status":"ok","model":"XEN-from-scratch"}

@app.post("/generate")
def generate_text(request:GenerateRequest):
    try:
        model,tokenizer=get_model()
        return {"text":generate(model,tokenizer,request.prompt,request.max_new_tokens,request.temperature)}
    except Exception as exc:
        raise HTTPException(status_code=500,detail=str(exc)) from exc
