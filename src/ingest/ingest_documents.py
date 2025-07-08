"""
firegpt/src/ingest/ingest_documents.py

Document ingestion pipeline: load, split and store in Chroma.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# ----------------------------------------------------------------------------
# Configuration and Logging
# ----------------------------------------------------------------------------

# Reduce noisy warnings from transformers
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logger = logging.getLogger("firegpt.ingest")
if not logger.handlers:
    _stream = logging.StreamHandler()
    _file = logging.FileHandler("ingest.log")
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in (_stream, _file):
        h.setFormatter(_fmt)
        logger.addHandler(h)
logger.setLevel(logging.INFO)

# Chunking parameters (in **tokens**, not characters)
CHUNK_SIZE = int(os.getenv("FGPT_CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("FGPT_CHUNK_OVERLAP", "128"))
COLL_NAME = os.getenv("FGPT_COLLECTION", "fire_docs")

# Embedding model path (local if available)
_EMBED_MODEL = os.getenv("FGPT_EMBED_MODEL", "/root/fp/AMI/FireGPT/models/bge-base-en-v1.5")
_embedder = HuggingFaceEmbeddings(
    model_name=_EMBED_MODEL,
    model_kwargs={"local_files_only": True},
)


# ----------------------------------------------------------------------------
# Document Loading
# ----------------------------------------------------------------------------


def _load_documents(source_dir: Path) -> List[Document]:
    """Recursively load supported file types into LangChain ``Document``s."""
    docs: List[Document] = []
    paths = list(source_dir.rglob("*"))

    with logging_redirect_tqdm():
        for path in tqdm(paths, desc="Loading documents", unit="file"):
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                loader = PDFPlumberLoader(str(path))
            elif suffix in {".txt", ".md"}:
                loader = TextLoader(str(path))
            else:
                loader = UnstructuredFileLoader(str(path))

            try:
                docs.extend(loader.load())
            except Exception as exc:
                logger.warning("Skipping %s - loader failed (%s)", path.name, exc)

    return docs


# ----------------------------------------------------------------------------
# Chunking & Processing
# ----------------------------------------------------------------------------


def _token_splitter() -> RecursiveCharacterTextSplitter:
    """Return a token-aware splitter configured for Qwen/Q-style BPE."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2",  # any HF-tokenizer name is acceptable here
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
    )


def _process_documents(docs: Iterable[Document]) -> List[Document]:
    """Split documents into token-bounded chunks ready for embedding."""
    splitter = _token_splitter()
    processed: List[Document] = []

    with logging_redirect_tqdm():
        for doc in tqdm(docs, desc="Processing documents", unit="doc"):
            chunks = splitter.split_documents([doc])
            for chunk in tqdm(chunks, desc="Chunks", leave=False, unit="chunk"):
                processed.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata={
                            "source": chunk.metadata.get("source", ""),
                            "pages": chunk.metadata.get("page", None),
                        },
                    )
                )

    logger.info("Prepared %d chunks", len(processed))
    return processed


# ----------------------------------------------------------------------------
# Chroma Store Builders
# ----------------------------------------------------------------------------

def build_persistent_store(source_dir: str | Path, target_dir: str | Path) -> Chroma:
    """Create or update a **persistent** Chroma database from a directory of docs."""
    src = Path(source_dir).expanduser().resolve()
    dst = Path(target_dir).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    logger.info("Loading documents from %s …", src)
    raw_docs = _load_documents(src)
    if not raw_docs:
        raise FileNotFoundError(f"No supported documents found in {src}.")

    processed = _process_documents(raw_docs)

    logger.info("Upserting into Chroma at %s…", dst)
    store = Chroma(
        collection_name=COLL_NAME,
        persist_directory=str(dst),
        embedding_function=_embedder,
    )
    store.add_documents(processed)

    count = store._collection.count()
    logger.info("Store ready with %d vectors", count)
    return store


def build_session_store(files: Iterable[Path]) -> Chroma:
    """Build an **in-memory** Chroma store for a one-off interactive session."""
    docs: List[Document] = []
    for path in files:
        docs.extend(_load_documents(path.parent / path.name))

    return Chroma.from_documents(docs=_process_documents(docs), embedding=_embedder)


# ----------------------------------------------------------------------------
# CLI Helper
# ----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """CLI: define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma store.")
    parser.add_argument("--source", default="docs/global", help="Directory with PDFs/docs")
    parser.add_argument("--target", default="stores/global", help="Directory to hold Chroma DB")
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete the target directory before writing"
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point."""
    args = _parse_args()
    target = Path(args.target)

    if args.overwrite and target.exists():
        import shutil

        shutil.rmtree(target)
        logger.info("Removed existing store at %s", target)

    build_persistent_store(args.source, args.target)


if __name__ == "__main__":
    main()
