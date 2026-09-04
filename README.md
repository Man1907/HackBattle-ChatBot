# HackBattle FAQ Chatbot

A retrieval-augmented (RAG) chatbot that answers visitor questions about
**IEEE-CS VIT HackBattle** (graVITas 2026), grounded strictly in the official
event document. It refuses off-topic questions instead of guessing, resists
prompt injection, and stays up even when its language model is unavailable.

The knowledge base is a plain JSON file, so event details can be updated
without touching any code. Deployed serverless on Modal as a FastAPI endpoint.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [The knowledge base](#the-knowledge-base)
5. [API reference](#api-reference)
6. [Running locally](#running-locally)
7. [Deploying to Modal](#deploying-to-modal)
8. [Updating event content](#updating-event-content)
9. [Configuration](#configuration)
10. [Design decisions](#design-decisions)
11. [Troubleshooting](#troubleshooting)
12. [Current status](#current-status)

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
| Editable knowledge base | content lives in `chunks.json`, versioned by a content fingerprint |
| Pluggable vector store | in-memory, ChromaDB, or pgvector behind one interface |

Everything except the final phrasing works with **no LLM at all** — retrieval,
filtering, the guardrail, and cited sources are all local.

---

## Architecture

The system separates a one-off **offline build** from the **online query path**.

### Offline (via `setup_volume.py`)

1. The event content is read from **`chunks.json`** — 14 curated,
   self-contained passages, each beginning with its topic label (`Prizes:`,
   `Team size and eligibility:`, …).
2. Each chunk is embedded with bge-large into a 1024-dimensional unit vector.
3. Vectors, chunk text, and topic metadata are written into a **ChromaDB**
   collection on a persistent **Modal Volume**, alongside a copy of
   `chunks.json` itself.

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

The engine loads once per **container**, not per request, and serves many
concurrent requests from that single load. The BM25 index is rebuilt in memory
at startup (it is cheap); the embedding model and vectors are read from the
Volume and never recomputed unless the content changed.

---

## Project structure

```
.
├── chunks.json       # the knowledge base — edit this to change answers
├── engine.py         # RAG engine: loader, hybrid retrieval, guardrail, generation
├── store.py          # pluggable vector store (memory | chroma | pgvector)
├── api.py            # FastAPI app for local development
├── modal_app.py      # Modal deployment (FastAPI on serverless containers)
├── setup_volume.py   # one-off: download model + publish chunks + build index
├── requirements.txt
├── .env.example      # template for local configuration
└── .env              # your local secrets (gitignored)
```

**Modal Volume `hackbattle-vol`**, mounted at `/models` in the container:

```
/models/bge-large-en-v1.5/   # embedding model (~1.3 GB)
/models/chroma/              # ChromaDB collection: vectors + metadata
/models/chunks.json          # the published knowledge base
```

`/models` is the *mount point*; the Volume's root contains those entries
directly.

---

## The knowledge base

All event content lives in **`chunks.json`**:

```json
{
  "version": "1.0",
  "chunks": [
    {
      "topic": "prizes",
      "text": "Prizes: HackBattle awards a 1st Prize, a 2nd Prize, a 3rd Prize, and a Best Freshers prize."
    }
  ]
}
```

Two rules make retrieval work well:

- **Start each chunk with its topic word.** A question containing "prizes"
  then lands directly on the chunk beginning `Prizes:`. This gives keyword
  search a strong signal.
- **Keep each chunk self-contained.** A retrieved chunk is shown without its
  neighbours, so it must answer on its own.

`[topic, text]` pairs and a bare top-level list are also accepted. The loader
validates the file and **fails loudly** on a missing file, empty content, or a
malformed entry — rather than starting a bot that answers "I don't know" to
everything.

> **Before deploying:** the `contacts` chunk contains a placeholder,
> `contact the respective host team at <email>`. Replace `<email>` with the
> real address, or the bot will tell visitors to email "\<email\>".
>
> Personal contact details were deliberately removed from this file. If you add
> names, phone numbers, or anything similar, remember the bot will hand them to
> any visitor who asks, and the file is in version control.

---

## API reference

Base URL: your Modal deployment URL
(`https://<workspace>--hackbattle-chatbot-chatbot-web.modal.run`).

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

Interactive Swagger UI for testing without a frontend. Note there is no route
at `/`; a 404 there is expected.

---

## Running locally

```bash
python -m pip install -r requirements.txt
cp .env.example .env        # then add your Groq API key
uvicorn api:app --reload --port 8000
```

`chunks.json` must sit beside `engine.py` (or set `CHUNKS_PATH`).

Open <http://localhost:8000/docs> and try:

- `{"message": "What are the prizes?"}` → a generated answer with sources
- `{"message": "What is the capital of France?"}` → `I don't know`

The startup log tells you what happened to the index:

- `Indexing 14 chunks (fingerprint ...)` — content is new or changed
- `Loaded 14 chunks from the store (fingerprint ...)` — unchanged, reused

The first run downloads bge-large (~1.3 GB) from Hugging Face and caches it, so
expect a few quiet minutes before the server is ready.

A free Groq API key comes from <https://console.groq.com>. Without one the
retrieval still works and the bot returns cited passages instead of prose.

---

## Deploying to Modal

```bash
python -m pip install modal
modal token new
modal secret create hackbattle-secrets GROQ_API_KEY=gsk_...

modal run setup_volume.py     # once: model + chunks + index into the Volume
modal serve  modal_app.py     # temporary URL, live reload, for testing
modal deploy modal_app.py     # permanent URL
```

`setup_volume.py` takes several minutes on the first run (it downloads 1.3 GB
into cloud storage).

`modal_app.py` sets its own environment in the `.env({...})` block — **your
local `.env` does not travel to Modal**. The Groq key comes from the Modal
secret.

---

## Updating event content

Edit `chunks.json`, then publish it:

```bash
modal run setup_volume.py::build_index
```

That copies the file onto the Volume and rebuilds the vectors. **No code change
and no redeploy needed.**

The engine stores a **content fingerprint** (a hash of every chunk) alongside
the vectors, so it re-embeds whenever the content differs — including when a
chunk is reworded and the chunk *count* stays the same. A count-only check
would miss that and keep serving stale answers.

A container that is already warm keeps serving the old vectors until it
recycles. To switch immediately, redeploy or stop the app so fresh containers
start.

---

## Configuration

Set locally in `.env`; set for production in the `.env({...})` block of
`modal_app.py`.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | free key from console.groq.com |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | must be a model your key can call |
| `LLM_BACKEND` | `groq` | `groq`, `ollama`, or `none` |
| `CHUNKS_PATH` | `./chunks.json` | knowledge base location (the Volume path in production) |
| `VECTOR_STORE` | `memory` | `memory`, `chroma`, or `pgvector` |
| `EMBED_MODEL_PATH` | — | local model directory (the Volume path in production) |
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

**Curated chunks in a data file, not automatic PDF splitting.** The source
document contained contradictory dates (12–13 vs 5–6 September) and an
internally inconsistent timeline table. A RAG bot repeats its sources
faithfully, so those contradictions would have been served to users verbatim.
Chunks were written by hand — dates normalised to 12–13 September 2026, the
"solo not permitted" versus "registration is individual" tension reconciled
explicitly, and the schedule labelled provisional. Keeping them in JSON rather
than in code means non-developers can review and edit the content, and updates
do not require a deploy.

**Hybrid retrieval rather than dense-only.** Visitors ask keyword-shaped
questions — "prizes", "team size", "GitHub deadline". Pure semantic search
blurs exact terms; BM25 matched the correct chunk on 10/10 such queries in
testing. RRF combines the two rankings without needing to calibrate their score
scales against each other.

**A relevance floor before the LLM, not after.** Off-topic questions are
rejected on retrieval score alone, so they cost no API call and cannot be
talked around. The strict prompt is a second line of defence for anything that
slips through.

**A content fingerprint, not a count check.** Detecting stale vectors by chunk
count alone silently misses edits that preserve the count, which is the most
common kind of edit.

**Generation isolated behind one interface.** Swapping Groq for Ollama, another
provider, or nothing at all touches a single class. The engine does not know or
care what produces the prose.

**Never cache a refusal.** `I don't know` is deliberately not cached, so a
question that fails today can succeed after the knowledge base improves.

---

## Troubleshooting

**`FileNotFoundError: Knowledge base not found at ...`**
`chunks.json` is missing. Place it beside `engine.py` or set `CHUNKS_PATH`.

**Edited `chunks.json` but answers are unchanged**
Locally, restart the server — the check runs at startup. On Modal, run
`modal run setup_volume.py::build_index`, and remember a warm container serves
old vectors until it recycles. The startup log distinguishes `Indexing …` from
`Loaded …`.

**Answers start with "Here's the relevant event information:"**
That is the fallback — the LLM is not being called. Set `DEBUG_LLM=1` and check
the server log. Usual causes: `GROQ_MODEL` names a model your key cannot call,
or the key is not loaded. Verify with `GET /health` (`llm` should be `groq`)
and list your models:
`curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`

**Off-topic questions get real answers**
`RELEVANCE_FLOOR` is too low for the embedding model. bge-large produces high
similarities even for weak matches; `0.55` works well. Calibrate by printing
`top_score` for a few on- and off-topic questions and setting the floor between
the two clusters.

**`Error code: 400 … does not contain the discriminator property 'role'`**
A malformed `history` entry reached the LLM. Current code sanitises history, so
this indicates an older `engine.py`. Swagger pre-fills `history` with a
placeholder object; it is now stripped automatically.

**`UnrecognizedEscape: unrecognized escape sequence` during image build**
A backslash reached the Modal image build. Container paths must be plain
forward-slash strings — never build them with `pathlib` on Windows. Use
`modal image logs <image-id>` for the exact failing step.

**`{"detail":"Not Found"}` at the base URL**
Expected — no route is defined at `/`. Use `/docs`, `/chat`, or `/health`.

---

## Current status

**Working and deployed.** The backend runs on Modal at a permanent URL,
answering event questions correctly from the knowledge base and refusing
off-topic questions. The model, content file, and index are persisted in the
Volume, so containers start in seconds without re-downloading or re-embedding.

**Outstanding:**

- **Replace `<email>` in the `contacts` chunk** with the real address.
- **Frontend widget not built.** The API is ready for it (`POST /chat`).
- **CORS is open (`*`).** Lock `ALLOWED_ORIGINS` to the real site origin before
  launch.
- **`min_containers` is `0`.** Set to `1` shortly before the event so live
  visitors never wait on a cold start.
- **Source document needs correcting.** The timeline table has durations that
  do not match their time ranges and a misaligned activities column. The bot
  serves the schedule as *provisional*; once the document is fixed, update
  `chunks.json` and rebuild.
- **Follow-up questions are limited.** Conversation history shapes generation
  but not retrieval, so a vague follow-up ("and their numbers?") may retrieve
  poorly. Query rewriting would fix this at the cost of an extra LLM call per
  turn.
