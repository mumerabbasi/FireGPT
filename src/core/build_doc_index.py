#!/usr/bin/env python
"""
firegpt/ingest/doc_ingest.py
Ingests all PDFs under a given directory into a ChromaDB vector store.
It uses a Mistral-7B-Instruct-v0.3 model for summarisation and an all-MiniLM-L6-v2 model for embeddings.
"""
from __future__ import annotations

import hashlib
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pdfplumber
import typer
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    pipeline,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class IngestConfig:
    root_dir: Path
    db_path: Path = Path("data/db/chroma")
    model_path: str = "models/mistral"  # Mistral-7B-Instruct-v0.3
    embedder_path: str = "models/minilm"  # all-MiniLM-L6-v2
    chunk_window_pages: int = 2
    summary_words: int = 120
    collection_name: str = "fire_docs"


# -----------------------------------------------------------------------------
# Section extraction & summarisation helpers
# -----------------------------------------------------------------------------
def extract_sections(
    pdf_path: Path,
    page_window: int = 10,
) -> Iterable[Tuple[str, str]]:
    """
    Yield (page_range, text) tuples for overlapping, fixed-size page windows.
    """
    if page_window <= 0:
        raise ValueError("page_window must be a positive integer")

    overlap = page_window // 4  # e.g. 4-page window → 1-page overlap
    step = max(1, page_window - overlap)

    with pdfplumber.open(pdf_path) as pdf:
        pages_text: List[str] = [page.extract_text() or "" for page in pdf.pages]

    total_pages = len(pages_text)
    for start in range(0, total_pages, step):
        end = min(start + page_window, total_pages)  # inclusive upper bound
        yield (
            f"{start + 1}-{end}",
            "\n".join(pages_text[start:end]),
        )
        if end == total_pages:
            break


def build_summariser(model_path: str) -> TextGenerationPipeline:
    LOGGER.info("Loading summariser model [%s] …", model_path)
    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,  # use local files only
        )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        # device_map="auto",
        device_map=None,  # force use single GPU
        torch_dtype="auto",
        local_files_only=True,  # use local files only
    )
    return pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok
    )


def summarise(
    pipe: TextGenerationPipeline,
    text: str,
    max_words: int,
) -> str:
    prompt = textwrap.dedent(
        f"""
        You are a technical summariser. Reduce the following passage
        to ≤ {max_words} words, preserving numeric thresholds and procedural rules.
        ### PASSAGE
        {text}
        ### SUMMARY
        """
    ).strip()

    # TODO: inspect which other library sets this
    # without temperature=None, generation gives a warning
    out = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=1000,
        temperature=None,
        pad_token_id=pipe.tokenizer.eos_token_id
    )[0]["generated_text"]
    summary = out.split("### SUMMARY")[-1].strip()
    return " ".join(summary.split()[: max_words + 5])  # safety trim


# -----------------------------------------------------------------------------
# Core ingestion routine
# -----------------------------------------------------------------------------
def ingest_directory(cfg: IngestConfig) -> None:
    embedder = SentenceTransformer(
        cfg.embedder_path,
        local_files_only=True,  # use local files only
    )
    client = PersistentClient(path=str(cfg.db_path))
    collection = client.get_or_create_collection(cfg.collection_name)
    summariser = build_summariser(cfg.model_path)

    pdf_files = sorted(cfg.root_dir.rglob("*.pdf"))
    LOGGER.info("Discovered %d PDFs under %s", len(pdf_files), cfg.root_dir)

    for pdf in pdf_files:
        LOGGER.info("Processing %s", pdf.relative_to(cfg.root_dir))
        for page_range, section_text in extract_sections(
            pdf, cfg.chunk_window_pages
        ):
            summary = summarise(summariser, section_text, cfg.summary_words)
            embedding = embedder.encode(
                summary,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            section_id = hashlib.sha1(
                f"{pdf}:{page_range}".encode()
            ).hexdigest()[:16]

            full_text_path = (
                cfg.db_path.parent / "fulltext" / f"{section_id}.txt"
            )
            full_text_path.parent.mkdir(parents=True, exist_ok=True)
            full_text_path.write_text(section_text, encoding="utf-8")

            collection.upsert(
                ids=[section_id],
                embeddings=[embedding.tolist()],
                documents=[summary],
                metadatas=[
                    {
                        "pdf": pdf.relative_to(cfg.root_dir).as_posix(),
                        "pages": page_range,
                        "full_text": str(full_text_path),
                    }
                ],
            )
            LOGGER.debug(
                "Section %s (%s pages %s) ingested.",
                section_id,
                pdf.name,
                page_range,
            )


# -----------------------------------------------------------------------------
# Typer CLI
# -----------------------------------------------------------------------------
cli = typer.Typer(add_completion=False)


@cli.command()
def main(
    root_dir: Path = typer.Option(
        "data/docs",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Root directory containing PDFs (scanned recursively).",
    ),
    db_path: Path = typer.Option(
        "data/db/chroma",
        help="Directory for ChromaDB persistent storage.",
    ),
    model_path: str = typer.Option(
        "models/mistral",
        help="Local path to the Mistral-7B-Instruct-v0.3 model directory, used for summarisation.",
    ),
    embedder_path: str = typer.Option(
        "models/minilm",
        help="Local path to the all-MiniLM-L6-v2 model directory, used for embeddings.",
    ),
    window: int = typer.Option(
        4,
        help="Window size for chunks (number of pages per chunk).",
    ),
    summary_words: int = typer.Option(
        250,   # around 1/2 page of summary for 4 pages
        help="Maximum words per generated summary.",
    ),
):
    """Ingest all PDFs under *ROOT_DIR* into a ChromaDB vector store."""
    cfg = IngestConfig(
        root_dir=root_dir,
        db_path=db_path,
        model_path=model_path,
        embedder_path=embedder_path,
        chunk_window_pages=window,
        summary_words=summary_words,
    )
    ingest_directory(cfg)
    LOGGER.info("Finished ingestion of PDFs in %s", root_dir)


if __name__ == "__main__":
    cli()
