"""
setup_volume.py — one-off Modal setup (CHROMA variant)
=======================================================

Fills the Chroma Volume with everything the serving app needs:
  1. download_model — the embedding model
  2. build_index    — publishes chunks.json and builds the vector index

Every value is literal; nothing is read from your shell. The FAISS equivalent
is setup_volume_faiss.py.

Run once before deploying:
    modal run setup_volume.py

Re-run after editing chunks.json:
    modal run setup_volume.py::build_index
"""

import modal

APP_NAME = "hackbattle-setup"
VOLUME_NAME = "hackbattle-vol"

# Container paths — Linux style, plain strings. Never build with pathlib.
MODEL_DIR = "/models"
EMBED_LOCAL = "/models/bge-large-en-v1.5"
CHROMA_DIR = "/models/chroma"
CHUNKS_FILE = "/models/chunks.json"

EMBED_REPO = "BAAI/bge-large-en-v1.5"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})       # fast Rust downloader
)

index_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentence-transformers", "bm25s", "rapidfuzz",
                 "numpy", "chromadb")
    .add_local_python_source("engine", "store")
    .add_local_file("chunks.json", "/tmp/chunks.json")
)

app = modal.App(APP_NAME)


@app.function(image=download_image, volumes={MODEL_DIR: volume}, timeout=1800)
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=EMBED_REPO, local_dir=EMBED_LOCAL)
    volume.commit()
    print(f"Embedding model downloaded to {EMBED_LOCAL}")


@app.function(image=index_image, volumes={MODEL_DIR: volume}, timeout=1800)
def build_index():
    import os, shutil
    shutil.copyfile("/tmp/chunks.json", CHUNKS_FILE)

    os.environ["EMBED_MODEL_PATH"] = EMBED_LOCAL
    os.environ["VECTOR_STORE"] = "chroma"        # literal
    os.environ["CHROMA_DIR"] = CHROMA_DIR
    os.environ["CHUNKS_PATH"] = CHUNKS_FILE
    os.environ["LLM_BACKEND"] = "none"           # no LLM needed to index

    from engine import HackBattleEngine, load_chunks
    chunks = load_chunks(CHUNKS_FILE)
    engine = HackBattleEngine()
    count = engine.store.count()
    volume.commit()
    print(f"Index built [chroma]: {count} chunks (expected {len(chunks)})")
    print(f"Knowledge base published to {CHUNKS_FILE}")


@app.local_entrypoint()
def main():
    download_model.remote()
    build_index.remote()
    print("Setup complete. Next: modal deploy modal_app.py")
