"""
modal_app.py — HackBattle chatbot on Modal
===========================================

Serves the RAG engine as a FastAPI app, reading the embedding model and the
prebuilt vector index from a persistent Volume.

NOTE ON PATHS: every container path below is a plain forward-slash STRING,
never a pathlib.Path. These paths live inside the Linux container. Building them
with pathlib on Windows produces Windows-style separators, which makes the Modal
image build fail with "UnrecognizedEscape: unrecognized escape sequence".

NOTE ON CONFIG: your local .env does NOT travel to Modal. Everything the
deployed app needs is set in the .env({...}) block below, and the Groq key
comes from a Modal secret.

Deploy:
    pip install modal
    modal token new
    modal secret create hackbattle-secrets GROQ_API_KEY=gsk_...
    modal run setup_volume.py          # once: model + index into the Volume
    modal serve modal_app.py           # temporary URL, live reload
    modal deploy modal_app.py          # permanent URL

Then POST {"message": "..."} to  https://<app>.modal.run/chat
"""

import modal

VOLUME_NAME = "hackbattle-vol"

# Container paths — Linux style, plain strings. Do not use pathlib here.
MODEL_DIR = "/models"
EMBED_LOCAL = "/models/bge-large-en-v1.5"
CHROMA_DIR = "/models/chroma"
CHUNKS_FILE = "/models/chunks.json"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "sentence-transformers",
        "bm25s",
        "numpy",
        "openai",
        "chromadb",
        "rapidfuzz",
        "faiss-cpu",
    )
    .env({
        "EMBED_MODEL_PATH": EMBED_LOCAL,
        "VECTOR_STORE": "chroma",
        "CHROMA_DIR": CHROMA_DIR,
        "CHUNKS_PATH": CHUNKS_FILE,
        "LLM_BACKEND": "groq",
        # must be a model your Groq key can actually call
        "GROQ_MODEL": "openai/gpt-oss-20b",
        # tuned for bge-large: 0.35 let off-topic questions through
        "RELEVANCE_FLOOR": "0.55",
        "TOP_K": "3",
        "DEBUG_LLM": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # lock this to the real site origin before going live
        "ALLOWED_ORIGINS": "*",
    })
    .add_local_python_source("engine", "store")
)

app = modal.App("hackbattle-chatbot", image=image)


@app.cls(
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("hackbattle-secrets")],
    min_containers=0,        # scale to zero when idle (saves credits).
                             # Set to 1 shortly before the event goes live so
                             # visitors never wait on a cold start.
    scaledown_window=300,
    timeout=300,
)
@modal.concurrent(max_inputs=50)
class Chatbot:

    @modal.enter()
    def load(self):
        """Runs once per container, not per request."""
        from engine import HackBattleEngine
        self.engine = HackBattleEngine()
        print(f"Engine ready: {self.engine.store.count()} chunks, "
              f"llm={getattr(self.engine.llm, 'name', None)}")

    @modal.asgi_app()
    def web(self):
        import os
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field

        api = FastAPI(title="HackBattle FAQ chatbot", version="2.1")
        origins = [o.strip() for o in
                   os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
        api.add_middleware(CORSMiddleware, allow_origins=origins,
                           allow_credentials=False,
                           allow_methods=["GET", "POST"], allow_headers=["*"])
        engine = self.engine

        class ChatRequest(BaseModel):
            message: str = Field(..., min_length=1, max_length=1000)
            history: list[dict] = Field(default_factory=list)

        @api.get("/health")
        def health():
            # reports the backend that actually loaded, not just whether a key
            # is present — a key can be set while the client failed to build
            return {"status": "ok",
                    "chunks": engine.store.count(),
                    "llm": getattr(engine.llm, "name", None)}

        @api.post("/chat")
        def chat(req: ChatRequest):
            return engine.answer(req.message, history=req.history)

        return api
