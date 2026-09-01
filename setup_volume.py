"""
setup_volume.py — one-off Modal setup
======================================

Fills the persistent Volume with everything the serving app needs, so a
starting container never downloads or rebuilds anything:

  1. download_model — pulls the embedding model into the Volume
  2. build_index    — embeds the chunks and writes the vector index into the
                      Volume (the earlier version loaded an index that nothing
                      ever created, which always failed at serve time)

NOTE ON PATHS: every path below is a plain forward-slash STRING, never a
pathlib.Path. These are paths *inside the Linux container*. Building them with
pathlib on Windows produces Windows-style separators, which makes the Modal
image build fail with "UnrecognizedEscape: unrecognized escape sequence".

Run once, before deploying:
    modal run setup_volume.py

Re-run after editing CHUNKS in engine.py:
    modal run setup_volume.py::build_index
"""

import modal

VOLUME_NAME = "hackbattle-vol"

# Container paths — Linux style, plain strings. Do not use pathlib here.
MODEL_DIR = "/models"
EMBED_LOCAL = "/models/bge-large-en-v1.5"
CHROMA_DIR = "/models/chroma"

EMBED_REPO = "BAAI/bge-large-en-v1.5"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})       # fast Rust downloader
)

index_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentence-transformers", "bm25s", "numpy", "chromadb")
    .add_local_python_source("engine", "store")
)

app = modal.App("hackbattle-setup")


@app.function(image=download_image, volumes={MODEL_DIR: volume}, timeout=1800)
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=EMBED_REPO, local_dir=EMBED_LOCAL)
    volume.commit()                # persist writes back to the Volume
    print(f"Embedding model downloaded to {EMBED_LOCAL}")


@app.function(image=index_image, volumes={MODEL_DIR: volume}, timeout=1800)
def build_index():
    """Embed the chunks and persist a Chroma collection into the Volume."""
    import os
    os.environ["EMBED_MODEL_PATH"] = EMBED_LOCAL
    os.environ["VECTOR_STORE"] = "chroma"
    os.environ["CHROMA_DIR"] = CHROMA_DIR
    os.environ["LLM_BACKEND"] = "none"      # no LLM needed just to index

    from engine import HackBattleEngine, CHUNKS
    engine = HackBattleEngine()             # embeds + upserts on construction
    count = engine.store.count()
    volume.commit()
    print(f"Index built: {count} chunks (expected {len(CHUNKS)}) at {CHROMA_DIR}")


@app.local_entrypoint()
def main():
    """modal run setup_volume.py — runs both steps in order."""
    download_model.remote()
    build_index.remote()
    print("Setup complete. Next: modal deploy modal_app.py")
