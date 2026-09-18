from __future__ import annotations

import os
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from inference.generate import generate
from model.loader import load_model

app = FastAPI(title="XEN Text-to-Text", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)


@lru_cache(maxsize=1)
def get_bundle():
    return load_model(os.getenv("XEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("XEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")}


@app.post("/generate")
def generate_text(request: GenerateRequest):
    try:
        bundle = get_bundle()
        answer = generate(bundle.model, bundle.tokenizer, request.prompt, request.max_new_tokens, request.temperature, request.top_p)
        return {"text": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
