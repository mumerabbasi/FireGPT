#!/usr/bin/env python
"""
firegpt/src/mcp_server/mcp_server.py

FastMCP server exposing document retrieval (vector search) and
fire-danger assessment.
"""

import json
import logging
import os
from pathlib import Path
from typing import List

from fastmcp import FastMCP
from pydantic import BaseModel
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from forest_fire_gee import Client as FireGEE
from forest_fire_gee.models import (
    FireDangerRequest,
    FireDangerResponse,
    BoundingBox,
)
from forest_fire_gee.api.default import assess_fire_danger_assess_fire_danger_post
from forest_fire_gee.types import Response

# ----------------------------------------------------------------------------
# Configuration & Logging
# ----------------------------------------------------------------------------

# DB Paths and Configuration
DB_PATH_SESSION = Path(os.getenv("FGPT_DB_PATH_SESSION", "stores/session"))
DB_PATH_LOCAL = Path(os.getenv("FGPT_DB_PATH_LOCAL", "stores/local"))
DB_PATH_GLOBAL = Path(os.getenv("FGPT_DB_PATH_GLOBAL", "stores/global"))

EMB_MODEL = Path(os.getenv("FGPT_EMBED_MODEL", "models/bge-m3"))
RERANK_MODEL = Path(os.getenv("FGPT_RERANK_MODEL", "models/bge-reranker-v2-m3"))
COLL_NAME = os.getenv("FGPT_COLLECTION", "fire_docs")
CANDIDATE_K = int(os.getenv("FGPT_CANDIDATE_K", "50"))
TOP_K = int(os.getenv("FGPT_TOP_K", "5"))

# MCP Server Configuration
HOST = os.getenv("FGPT_HOST", "0.0.0.0")
PORT = int(os.getenv("FGPT_PORT", "7790"))

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
LOG = logging.getLogger("firegpt.fastmcp")

# ----------------------------------------------------------------------------
# External Clients & Vector Store
# ----------------------------------------------------------------------------

# FireGEE API client
client = FireGEE(base_url="https://api.firefirefire.lol")

# Embedding model
LOG.info("Loading embeddings from %s …", EMB_MODEL)
LOG.info("Loading reranker from %s …", RERANK_MODEL)
_embedder = HuggingFaceEmbeddings(model_name=str(EMB_MODEL), model_kwargs={"local_files_only": True})

# Reranker model
_cross_encoder = HuggingFaceCrossEncoder(
    model_name=str(RERANK_MODEL),
    model_kwargs={
        "device": "cuda",
        "local_files_only": True,
    },
)

_reranker = CrossEncoderReranker(model=_cross_encoder, top_n=TOP_K)

# Initialize Chroma vector stores
LOG.info("Connecting to ChromaDB at %s, collection '%s'…", DB_PATH_SESSION, COLL_NAME)
_store_session = Chroma(collection_name=COLL_NAME, persist_directory=str(DB_PATH_SESSION), embedding_function=_embedder)

LOG.info("Connecting to ChromaDB at %s, collection '%s'…", DB_PATH_LOCAL, COLL_NAME)
_store_local = Chroma(collection_name=COLL_NAME, persist_directory=str(DB_PATH_LOCAL), embedding_function=_embedder)

LOG.info("Connecting to ChromaDB at %s, collection '%s'…", DB_PATH_GLOBAL, COLL_NAME)
_store_global = Chroma(collection_name=COLL_NAME, persist_directory=str(DB_PATH_GLOBAL), embedding_function=_embedder)

# Wrap each vector store in a ContextualCompressionRetriever
_retriever_session = ContextualCompressionRetriever(
    base_retriever=_store_session.as_retriever(search_kwargs={"k": CANDIDATE_K}),
    base_compressor=_reranker,
)

_retriever_local = ContextualCompressionRetriever(
    base_retriever=_store_local.as_retriever(search_kwargs={"k": CANDIDATE_K}),
    base_compressor=_reranker,
)

_retriever_global = ContextualCompressionRetriever(
    base_retriever=_store_global.as_retriever(search_kwargs={"k": CANDIDATE_K}),
    base_compressor=_reranker,
)


# ----------------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------------

class DocHit(BaseModel):
    """Single vector-search hit."""

    id: str
    pdf: str
    pages: int
    text: str
    score: float


class MetaEntry(BaseModel):
    """Metadata entry for `docs/metadata` resource."""

    id: str
    pdf: str
    pages: str


# ----------------------------------------------------------------------------
# FastMCP Resources & Tools
# ----------------------------------------------------------------------------

mcp = FastMCP("FireGPT-All-In-One")


@mcp.resource("data://docs/metadata")
def docs_metadata() -> List[MetaEntry]:
    """Return id, source PDF path and pages for every stored chunk."""

    data = _store_local.get(include=["metadatas", "ids"])
    entries: List[MetaEntry] = []
    for idx, meta in zip(data["ids"], data["metadatas"]):
        entries.append(MetaEntry(id=idx, pdf=meta.get("source", ""), pages=meta.get("pages", "?")))
    return entries


@mcp.tool
def retrieve_chunks_session(
    query: str,
) -> List[DocHit]:
    """Semantic retrieval with relevance scores of FireFighting SOPs from the user-provided docs.

    Parameters
    ----------
    query : str
        Natural-language search string.

    Returns
    -------
    List[DocHit]
        Vector hits including their similarity score ∈ [0, 1].
    """
    docs = _retriever_session.invoke(query)
    hits: List[DocHit] = []
    logging.info(f"Retrieved {len(docs)} chunks from session store.")
    for doc in docs:
        meta = doc.metadata or {}
        hits.append(
            DocHit(
                id=meta.get("id", ""),
                pdf=meta.get("source", ""),
                pages=meta.get("pages", -1),
                text=doc.page_content,
                score=float(doc.metadata.get("rerank_score", 1.0)),
            )
        )
    return hits


@mcp.tool
def retrieve_chunks_local(
    query: str,
) -> List[DocHit]:
    """Semantic retrieval with relevance scores of FireFighting SOPs implemented in my region.

    Parameters
    ----------
    query : str
        Natural-language search string.

    Returns
    -------
    List[DocHit]
        Vector hits including their similarity score ∈ [0, 1].
    """
    docs = _retriever_local.invoke(query)
    hits: List[DocHit] = []
    logging.info(f"Retrieved {len(docs)} chunks from local store.")
    for doc in docs:
        meta = doc.metadata or {}
        hits.append(
            DocHit(
                id=meta.get("id", ""),
                pdf=meta.get("source", ""),
                pages=meta.get("pages", -1),
                text=doc.page_content,
                score=float(doc.metadata.get("rerank_score", 1.0)),
            )
        )
    return hits


@mcp.tool
def retrieve_chunks_global(
    query: str,
) -> List[DocHit]:
    """Semantic retrieval with relevance scores of FireFighting SOPs implemented in my region.

    Parameters
    ----------
    query : str
        Natural-language search string.

    Returns
    -------
    List[DocHit]
        Vector hits including their similarity score ∈ [0, 1].
    """
    docs = _retriever_global.invoke(query)
    hits: List[DocHit] = []
    logging.info(f"Retrieved {len(docs)} chunks from global store.")
    for doc in docs:
        meta = doc.metadata or {}
        hits.append(
            DocHit(
                id=meta.get("id", ""),
                pdf=meta.get("source", ""),
                pages=meta.get("pages", -1),
                text=doc.page_content,
                score=float(doc.metadata.get("rerank_score", 1.0)),
            )
        )
    return hits


@mcp.tool
def assess_fire_danger(
    top_left_lat: float = 47.6969,
    top_left_lon: float = 7.9468,
    bottom_right_lat: float = 47.7024,
    bottom_right_lon: float = 7.9901,
) -> FireDangerResponse | None:
    """
    Assess fire danger at a given bounding box using the FireGEE API. Required parameters:
    - top_left_lat: Latitude of the top-left corner of the bounding box.
    - top_left_lon: Longitude of the top-left corner of the bounding box.
    - bottom_right_lat: Latitude of the bottom-right corner of the bounding box.
    - bottom_right_lon: Longitude of the bottom-right corner of the bounding box.
    Uses the FireGEE API to get the fire danger assessment.
    It divides the given bbox into a grid. At the center of each grid cell, it calcualtes params like vegetation type,
    vegetation amount, slope, weather, wind speed and direction etc. This could tell in which direction, fire is most
    likely to spread. Also, it checks if there are any critical buildings near or inside the bounding box. Based on
    that, it calculates a risk score for each of those buildings.

    Returns a json.
    """
    subgrid_size_m: int = 250
    forecast_hours: int = 1
    poi_search_buffer_m: int = 1000
    bbox = BoundingBox(
        top_left_lat=top_left_lat,
        top_left_lon=top_left_lon,
        bottom_right_lat=bottom_right_lat,
        bottom_right_lon=bottom_right_lon,
    )
    payload = FireDangerRequest(
        bbox=bbox,
        subgrid_size_m=subgrid_size_m,
        forecast_hours=forecast_hours,
        poi_search_buffer_m=poi_search_buffer_m,
    )

    response: Response[FireDangerResponse] = assess_fire_danger_assess_fire_danger_post.sync_detailed(
        client=client, body=payload
    )
    if response.status_code == 200:
        body = json.loads(response.content)
        return FireDangerResponse(**body)

    LOG.error("FireGEE API error: %s", response.status_code)
    return None


# ----------------------------------------------------------------------------
# Server Entry Point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    LOG.info("Starting FireGPT FastMCP server on %s:%d …", HOST, PORT)
    mcp.run(transport="streamable-http", host=HOST, port=PORT)
