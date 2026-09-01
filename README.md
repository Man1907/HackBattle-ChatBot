# HackBattle FAQ Chatbot

A retrieval-augmented (RAG) chatbot that answers visitor questions about
**IEEE-CS VIT HackBattle** (graVITas 2026), grounded strictly in the official
event document. It refuses off-topic questions instead of guessing, resists
prompt injection, and stays up even when its language model is unavailable.

Deployed serverless on Modal as a FastAPI endpoint.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [API reference](#api-reference)
5. [Running locally](#running-locally)
6. [Deploying to Modal](#deploying-to-modal)
7. [Updating the knowledge base](#updating-the-knowledge-base)
8. [Configuration](#configuration)
9. [Design decisions](#design-decisions)
10. [Troubleshooting](#troubleshooting)
11. [Current status](#current-status)

---

## What it does

| Capability | Implementation |
|---|---|
| Semantic search | `BAAI/bge-large-en-v1.5` embeddings (1024-dim) |
| Keyword search | BM25 via `bm25s` |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) over both rankings |
| Relevance guardrail | cosine floor; below it the bot replies `I don't know` without calling the LLM |
| Grounded generation | open-weight LLM via Groq, constrained by a strict system prompt |
| Prompt-injection resistance | system prompt declines instruction-override attempts; client history is sanitised |
| Answer caching | repeated questions are served from memory, costing no API calls |
| Graceful fallback | if the LLM is rate-limited or down, returns the retrieved source passages |
| Pluggable vector store | in-memory, ChromaDB, or pgvector behind one interface |

Everything except the final phrasing works with **no LLM at all** — retrieval,
filtering, the guardrail, and cited sources are all local.

---

## Architecture

The system separates a one-off **offline build** from the **online query path**.

### Offline (run once, via `setup_volume.py`)

1. The event document is represented as **14 curated, self-contained chunks**
   in `engine.py`. Each begins with its topic label (`Prizes:`,
   `Team size and eligibility:`, …).
2. Each chunk is embedded with bge-large into a 1024-dimensional unit vector.
3. Vectors, chunk text, and topic metadata are written into a **ChromaDB**
   collection stored on a persistent **Modal Volume**.

### Online (per request)

```
question
   ├─► dense search  (bge embedding → ChromaDB, cosine)
   └─► sparse search (BM25 keyword match)
            │
            ▼
   Reciprocal Rank Fusion  →  top-K chunks
            │
            ▼
   relevance floor?  ── below ──►  "I don't know"  (no LLM call)
            │ above
            ▼
   strict prompt + chunks  →  Groq  →  grounded answer + sources
            │
            └─ on failure ──►  cited source passages (never an error)
```

The BM25 index is rebuilt in memory at container start (it is cheap); the
embedding model and vectors are read from the Volume, never recomputed.

---

## Project structure

```
.
├── engine.py         # RAG engine: chunks, hybrid retrieval, guardrail, generation
├── store.py          # pluggable vector store (memory | chroma | pgvector)
├── api.py            # FastAPI app for local development
├── modal_app.py      # Modal deployment (FastAPI on serverless containers)
├── setup_volume.py   # one-off: download model + build index into the Volume
├── requirements.txt
├── .env.example      # template for local configuration
└── .env              # your local secrets (gitignored)
```

**Modal Volume `hackbattle-vol`**, mounted at `/models` in the container:

```
/models/bge-large-en-v1.5/   # embedding model (~1.3 GB)
/models/chroma/              # ChromaDB collection: the 14 vectors + metadata
```

Note `/models` is the *mount point*; the Volume's root contains those two
directories directly.

---

## API reference

Base URL: your Modal deployment URL (`https://<workspace>--hackbattle-chatbot-chatbot-web.modal.run`).

### `POST /chat`

```json
{
  "message": "When is the event?",
  "history": [
    {"role": "user", "content": "Who is organising it?"},
    {"role": "assistant", "content": "IEEE CS, as part of graVITas 2026."}
  ]
}
```

`history` is optional. Malformed entries are dropped rather than rejected, and
only the last four turns are used.

**Response**

```json
{
  "reply": "The event is scheduled for 12 September 2026 (day 1) and 13 September 2026 (day 2).",
  "sources": ["schedule day 1", "dates and duration"],
  "grounded": true,
  "cached": false
}
```

| Field | Meaning |
|---|---|
| `reply` | the answer, or exactly `I don't know` |
| `sources` | topic labels of the chunks used; empty when not grounded |
| `grounded` | `false` when the question was refused as off-topic |
| `cached` | `true` when served from the answer cache |

### `GET /health`

```json
{"status": "ok", "chunks": 14, "llm": "groq"}
```

`llm` reports the backend that actually loaded — `null` means generation is
unavailable and the bot is serving cited passages.

### `GET /docs`

Interactive Swagger UI for testing without a frontend.

---

## Running locally

```bash
python -m pip install -r requirements.txt
cp .env.example .env        # then add your Groq API key
uvicorn api:app --reload --port 8000
```

Open <http://localhost:8000/docs> and try:

- `{"message": "What are the prizes?"}` → a generated answer with sources
- `{"message": "What is the capital of France?"}` → `I don't know`

The first run downloads bge-large (~1.3 GB) from Hugging Face and caches it, so
expect a few minutes of quiet before the server is ready.

A free Groq API key comes from <https://console.groq.com>. Without one the
retrieval still works and the bot returns cited passages instead of prose.

---

## Deploying to Modal

```bash
python -m pip install modal
modal token new
modal secret create hackbattle-secrets GROQ_API_KEY=gsk_...

modal run setup_volume.py     # once: model + index into the Volume
modal serve  modal_app.py     # temporary URL, live reload, for testing
modal deploy modal_app.py     # permanent URL
```

`setup_volume.py` takes several minutes (it downloads 1.3 GB into cloud
storage). You only run it again when the knowledge base changes.

`modal_app.py` sets its own environment in the `.env({...})` block — **your
local `.env` does not travel to Modal**. The Groq key comes from the Modal
secret.

---

## Updating the knowledge base

Editing `CHUNKS` in `engine.py` and redeploying is **not enough**. The vectors
live in the Volume, and the engine only re-embeds when the stored count differs
from the number of chunks. After any content edit:

```bash
modal run setup_volume.py::build_index
modal deploy modal_app.py
```

---

## Configuration

Set locally in `.env`; set for production in the `.env({...})` block of
`modal_app.py`.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | free key from console.groq.com |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | must be a model your key can call |
| `LLM_BACKEND` | `groq` | `groq`, `ollama`, or `none` |
| `VECTOR_STORE` | `memory` | `memory`, `chroma`, or `pgvector` |
| `EMBED_MODEL_PATH` | — | local model directory (set to the Volume path in production) |
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Hub fallback when no local path |
| `RELEVANCE_FLOOR` | `0.35` | cosine floor for the guardrail; **0.55 is tuned for bge-large** |
| `TOP_K` | `5` | chunks passed to the LLM; `3` is usually enough |
| `ALLOWED_ORIGINS` | `*` | CORS; set to the real site origin before launch |
| `DEBUG_LLM` | — | set to `1` to print LLM errors instead of failing silently |

Modal container settings live in `modal_app.py`:

- `min_containers` — `0` scales to zero when idle (saves credits, costs a few
  seconds on the first request); `1` keeps a container warm.
- `scaledown_window` — how long idle containers linger before stopping.
- `@modal.concurrent(max_inputs=50)` — one container serves many simultaneous
  requests.

---

## Design decisions

**Curated chunks rather than automatic PDF splitting.** The source document
contained contradictory dates (12–13 vs 5–6 September) and an internally
inconsistent timeline table. A RAG bot repeats its sources faithfully, so those
contradictions would have been served to users verbatim. Chunks were written by
hand: dates normalised to 12–13 September 2026, the "solo not permitted" versus
"registration is individual" tension reconciled explicitly, and the schedule
labelled provisional. Each chunk leads with its topic word, which gives keyword
search a strong signal.

**Hybrid retrieval rather than dense-only.** Visitors ask keyword-shaped
questions — "prizes", "team size", "GitHub deadline". Pure semantic search
blurs exact terms; BM25 matched the correct chunk on 10/10 such queries in
testing. RRF combines the two rankings without needing to calibrate their score
scales against each other.

**A relevance floor before the LLM, not after.** Off-topic questions are
rejected on retrieval score alone, so they cost no API call and cannot be
talked around. The strict prompt is a second line of defence for anything that
slips through.

**Generation isolated behind one interface.** Swapping Groq for Ollama, another
provider, or nothing at all touches a single class. The engine does not know or
care what produces the prose.

**Never cache a refusal.** `I don't know` is deliberately not cached, so a
question that fails today can succeed after the knowledge base improves.

**No vector store locally, ChromaDB in production.** Fourteen chunks embed in
under a second, so in-memory NumPy is the right local choice; the Volume-backed
Chroma collection is what makes container starts fast in production.

---

## Troubleshooting

**Answers start with "Here's the relevant event information:"**
That is the fallback — the LLM is not being called. Set `DEBUG_LLM=1` and check
the server log. Usual causes: `GROQ_MODEL` names a model your key cannot call,
or the key is not loaded. Verify with `GET /health` (`llm` should be `groq`)
and list your available models:
`curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`

**Off-topic questions get real answers**
`RELEVANCE_FLOOR` is too low for the embedding model. bge-large produces high
similarities even for weak matches; `0.55` works well. Calibrate by printing
`top_score` for a few on- and off-topic questions and setting the floor between
the two clusters.

**`Error code: 400 … does not contain the discriminator property 'role'`**
A malformed `history` entry reached the LLM. Current code sanitises history, so
this indicates an older `engine.py`. Note Swagger pre-fills `history` with a
placeholder object; it is now stripped automatically.

**`UnrecognizedEscape: unrecognized escape sequence` during image build**
A Windows-style path (or any backslash) reached the Modal image build. Container
paths must be plain forward-slash strings — never build them with `pathlib` on
Windows. Check with `modal image logs <image-id>` for the exact step.

**Edited the chunks but answers are unchanged**
The Volume still holds the old vectors. Run
`modal run setup_volume.py::build_index`.

**`{"detail":"Not Found"}` at the base URL**
Expected — no route is defined at `/`. Use `/docs`, `/chat`, or `/health`.

---

## Current status

**Working and deployed.** The backend runs on Modal at a permanent URL,
answering event questions correctly from the curated knowledge base and
refusing off-topic questions. The model and index are persisted in the Volume,
so containers start in seconds without re-downloading or re-embedding.

**Outstanding:**
- **CORS is open (`*`).** Lock `ALLOWED_ORIGINS` to the real site origin before
  launch.
- **`min_containers` is `0`.** Set to `1` shortly before the event so live
  visitors never wait on a cold start.
- **Follow-up questions are limited.** Conversation history shapes generation
  but not retrieval, so a vague follow-up ("and their numbers?") may retrieve
  poorly. Query rewriting would fix this at the cost of an extra LLM call per
  turn.

---
