"""
api.py — HackBattle chatbot backend (FastAPI)
==============================================

Wraps the RAG engine in an HTTP API so any frontend can use it.

Endpoints:
    GET  /health   -> liveness + how many chunks are indexed
    POST /chat     -> {"message": "...", "history": [...]} -> answer + sources

Run locally:
    pip install fastapi uvicorn sentence-transformers bm25s numpy openai python-dotenv
    uvicorn api:app --reload --port 8000
    # then: http://localhost:8000/docs

Env:
    GROQ_API_KEY   free key from console.groq.com (optional)
    GROQ_MODEL     default llama-3.1-8b-instant
    VECTOR_STORE   memory | chroma | pgvector
    ALLOWED_ORIGINS  comma-separated site origins for CORS
"""

from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from engine import HackBattleEngine

ALLOWED_ORIGINS = [o.strip() for o in
                   os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="HackBattle chatbot", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,      # set to the real site origin in production
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Built once at import, reused for every request. Never load per-request.
engine = HackBattleEngine()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    grounded: bool          # False when the guardrail refused (off-topic)
    cached: bool


@app.get("/health")
def health():
    # report the backend that actually loaded, not merely whether a key is
    # present — a key can be set while the client failed to build.
    return {"status": "ok",
            "chunks": engine.store.count(),
            "store": os.environ.get("VECTOR_STORE", "memory"),
            "llm": getattr(engine.llm, "name", None)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = engine.answer(req.message, history=req.history)
    return ChatResponse(**result)
