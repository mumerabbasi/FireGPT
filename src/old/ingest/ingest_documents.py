"""
firegpt/src/ingest/ingest_documents.py

Document ingestion pipeline: load, split, summarize, and store in Chroma.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ----------------------------------------------------------------------------
# Configuration and Logging
# ----------------------------------------------------------------------------

# Set transformers verbosity to error to suppress invalid flag warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logger = logging.getLogger("firegpt.ingest")
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("ingest.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (stream_handler, file_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
logger.setLevel(logging.INFO)

CHUNK_SIZE = int(os.getenv("FGPT_CHUNK_SIZE", "3000"))
CHUNK_OVERLAP = int(os.getenv("FGPT_CHUNK_OVERLAP", "200"))
COLL_NAME = os.getenv("FGPT_COLLECTION", "fire_docs")

# Embedding model path (local if available)
_EMBED_MODEL = os.getenv("FGPT_EMBED_MODEL", "models/bge-base-en-v1.5")
_embedder = HuggingFaceEmbeddings(
    model_name=_EMBED_MODEL,
    model_kwargs={"local_files_only": True},
)

# Directory containing Mistral weights
_MISTRAL_DIR = Path(os.getenv("FGPT_MISTRAL_DIR", "models/mistral")).expanduser()


# ----------------------------------------------------------------------------
# LLM and Summarization Chain
# ----------------------------------------------------------------------------

def _build_mistral_llm() -> HuggingFacePipeline:
    """
    Construct a text-generation pipeline for Mistral-7B-Instruct.
    """
    return HuggingFacePipeline.from_model_id(
        model_id=str(_MISTRAL_DIR),
        task="text-generation",
        model_kwargs={"torch_dtype": "auto", "local_files_only": True},
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
        },
    )


_mistral_llm = _build_mistral_llm()
_summary_chain = load_summarize_chain(llm=_mistral_llm, chain_type="stuff")


# ----------------------------------------------------------------------------
# Document Loading and Splitting
# ----------------------------------------------------------------------------

def _load_documents(source_dir: Path) -> List[Document]:
    """
    Recursively load supported file types into LangChain Documents,
    with a progress bar.
    """
    docs: List[Document] = []
    paths = list(source_dir.rglob("*"))
    with logging_redirect_tqdm():
        for path in tqdm(paths, desc="Loading documents", unit="file"):
            suffix = path.suffix.lower()
            if suffix == '.pdf':
                # TODO: check how to prevent it from splitting pages
                loader = PDFPlumberLoader(str(path))
            elif suffix in {'.txt', '.md'}:
                loader = TextLoader(str(path))
            else:
                loader = UnstructuredFileLoader(str(path))

            try:
                docs.extend(loader.load())
            except Exception as exc:
                logger.warning("Skipping %s - loader failed (%s)", path.name, exc)
    return docs


def _summarise_chunk(chunk: Document) -> str:
    """
    Summarize a document chunk using the Mistral pipeline.
    """
    result = _summary_chain.invoke([chunk])
    text = result[_summary_chain.output_key]
    return text.strip()


def _process_documents(docs: Iterable[Document]) -> List[Document]:
    """
    Split, summarize, and wrap chunks for embedding and storage,
    with progress feedback.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
    )
    processed: List[Document] = []

    with logging_redirect_tqdm():
        for doc in tqdm(docs, desc="Processing documents", unit="doc"):
            chunks = splitter.split_documents([doc])
            for chunk in tqdm(chunks, desc="Chunks", leave=False, unit="chunk"):
                summary = _summarise_chunk(chunk)
                processed.append(
                    Document(
                        page_content=summary,
                        metadata={
                            "full_text": chunk.page_content,
                            "source": chunk.metadata.get("source", ""),
                            "pages": chunk.metadata.get("page", "?"),
                        },
                    )
                )
    logger.info("Prepared %d summarised chunks", len(processed))
    return processed


# ----------------------------------------------------------------------------
# Chroma Store Builders
# ----------------------------------------------------------------------------

def build_persistent_store(
    source_dir: str | Path, target_dir: str | Path
) -> Chroma:
    """
    Create or update a persistent Chroma database from a directory of docs.
    """
    src = Path(source_dir).expanduser().resolve()
    dst = Path(target_dir).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    logger.info("Loading documents from %s …", src)
    raw_docs = _load_documents(src)
    if not raw_docs:
        raise FileNotFoundError(
            f"No supported documents found in {src}."
        )

    processed = _process_documents(raw_docs)

    logger.info("Upserting into Chroma at %s…", dst)
    store = Chroma(
        collection_name=COLL_NAME,
        persist_directory=str(dst),
        embedding_function=_embedder,
    )
    store.add_documents(processed)
    store.persist()

    count = store._collection.count()
    logger.info("Store ready with %d vectors", count)
    return store


def build_session_store(files: Iterable[Path]) -> Chroma:
    """
    Build an in-memory Chroma store for a one-off session.
    """
    docs: List[Document] = []
    for path in files:
        docs.extend(_load_documents(path.parent / path.name))
    return Chroma.from_documents(
        docs=_process_documents(docs), embedding=_embedder
    )


# ----------------------------------------------------------------------------
# CLI Helper
# ----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents into Chroma store."
    )
    parser.add_argument(
        "--source",
        default="docs",
        help="Directory with PDFs/docs",
    )
    parser.add_argument(
        "--target",
        default="stores",
        help="Directory to hold Chroma DB",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the target directory before writing",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point: optionally wipe and then build the store.
    """
    args = _parse_args()
    if args.overwrite and Path(args.target).exists():
        import shutil

        shutil.rmtree(args.target)
        logger.info("Removed existing store at %s", args.target)

    build_persistent_store(args.source, args.target)


if __name__ == "__main__":
    main()
