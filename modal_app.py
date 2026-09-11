"""
modal_app.py — HackBattle chatbot on Modal (CHROMA variant)
============================================================

Every value below is written literally. Nothing is read from your shell.

This is deliberate. An earlier version selected the vector store from a STORE
environment variable, but Modal resolves the image definition at BUILD time and
caches the result, so a locally-set variable did not reliably reach the running
container: the app deployed under the right name while still running the wrong
store. Explicit files remove that whole class of failure.

The FAISS variant lives in modal_app_faiss.py. The two use separate app names
and separate Volumes, so they run side by side and can be benchmarked.

Deploy:
    modal secret create hackbattle-secrets GROQ_API_KEY=gsk_...
    modal run setup_volume.py          # once: model + chunks + index
    modal deploy modal_app.py

Then POST {"message": "..."} to  https://<app>.modal.run/chat
"""

import modal

APP_NAME = "hackbattle-chatbot"
VOLUME_NAME = "hackbattle-vol"

# Container paths — Linux style, plain strings. Never build these with pathlib
# on Windows: the backslashes break the Modal image build.
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
        "rapidfuzz",
        "numpy",
        "openai",
        "chromadb",
    )
    .env({
        "EMBED_MODEL_PATH": EMBED_LOCAL,
        "VECTOR_STORE": "chroma",           # literal, not from the shell
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
        # Locked to the frontend origins below. Any other site calling
        # this API will be blocked by the browser.
        "ALLOWED_ORIGINS": (
            # The frontend team's origins. Comma-separated, no spaces,
            # no trailing slashes — an origin is scheme + host + port only.
            "http://localhost:3000,"
            "https://hackbattle-26-frontend.vercel.app,"
            "https://hackbattle-26-frontend-rry2.vercel.app,"
            # production domain
            "https://hackbattle.ieeecsvit.com"
        ),
    })
    .add_local_python_source("engine", "store")
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("hackbattle-secrets")],
    min_containers=1,        # scale to zero when idle; set 1 before the event
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
        print(f"Engine ready [chroma]: {self.engine.store.count()} chunks, "
              f"llm={getattr(self.engine.llm, 'name', None)}")

    @modal.asgi_app()
    def web(self):
        import os
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field

        api = FastAPI(title="HackBattle FAQ chatbot (chroma)", version="2.2")
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
            return {"status": "ok",
                    "store": os.environ.get("VECTOR_STORE"),
                    "chunks": engine.store.count(),
                    "llm": getattr(engine.llm, "name", None)}

        @api.post("/chat")
        def chat(req: ChatRequest):
            return engine.answer(req.message, history=req.history)

        return api
