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

DB_PATH_LOCAL = Path(os.getenv("FGPT_DB_PATH_LOCAL", "stores/local"))
DB_PATH_GLOBAL = Path(os.getenv("FGPT_DB_PATH_LOCAL", "stores/global"))
EMB_MODEL = Path(os.getenv("FGPT_EMBED_MODEL", "models/bge-base-en-v1.5"))
COLL_NAME = os.getenv("FGPT_COLLECTION", "fire_docs")
TOP_K = int(os.getenv("FGPT_TOP_K", "5"))

HOST = os.getenv("FGPT_HOST", "0.0.0.0")
PORT = int(os.getenv("FGPT_PORT", "7790"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
LOG = logging.getLogger("firegpt.fastmcp")

# ----------------------------------------------------------------------------
# External Clients & Vector Store
# ----------------------------------------------------------------------------

# FireGEE API client
client = FireGEE(base_url="https://api.firefirefire.lol")

# Embedding model - *still needed* at retrieval time!
LOG.info("Loading embeddings from %s …", EMB_MODEL)
_embedder = HuggingFaceEmbeddings(model_name=str(EMB_MODEL), model_kwargs={"local_files_only": True})

# LangChain → Chroma wrapper
LOG.info("Connecting to ChromaDB at %s, collection '%s'…", DB_PATH_LOCAL, COLL_NAME)
_store_local = Chroma(collection_name=COLL_NAME, persist_directory=str(DB_PATH_LOCAL), embedding_function=_embedder)

# LangChain → Chroma wrapper
LOG.info("Connecting to ChromaDB at %s, collection '%s'…", DB_PATH_GLOBAL, COLL_NAME)
_store_global = Chroma(collection_name=COLL_NAME, persist_directory=str(DB_PATH_GLOBAL), embedding_function=_embedder)


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
def retrieve_chunks_local(
    query: str,
    k: int = TOP_K,
    score_threshold: float | None = 0.2,
) -> List[DocHit]:
    """Semantic retrieval with relevance scores of FireFighting SOPs implemented in my region.

    Parameters
    ----------
    query : str
        Natural-language search string.
    k : int, default *(env FGPT_TOP_K)*
        Maximum number of hits to return **before** filtering.
    score_threshold : float | None, default 0.2
        Drop results whose relevance score is below this value.
        Set to *None* to disable filtering.

    Returns
    -------
    List[DocHit]
        Vector hits including their similarity score ∈ [0, 1].
    """

    # Wrapper method embeds the query for us
    raw_hits = _store_local.similarity_search_with_relevance_scores(query, k=k)
    filtered_hits: List[DocHit] = []
    for doc, score in raw_hits:
        if score_threshold is not None and score < score_threshold:
            continue
        meta = doc.metadata or {}
        filtered_hits.append(
            DocHit(
                id=meta.get("id", ""),
                pdf=meta.get("source", ""),
                pages=meta.get("pages", -1),
                text=doc.page_content,
                score=score,
            )
        )
    return filtered_hits


@mcp.tool
def retrieve_chunks_global(
    query: str,
    k: int = TOP_K,
    score_threshold: float | None = 0.2,
) -> List[DocHit]:
    """Semantic retrieval with relevance scores of FireFighting SOPs implemented in my region.

    Parameters
    ----------
    query : str
        Natural-language search string.
    k : int, default *(env FGPT_TOP_K)*
        Maximum number of hits to return **before** filtering.
    score_threshold : float | None, default 0.2
        Drop results whose relevance score is below this value.
        Set to *None* to disable filtering.

    Returns
    -------
    List[DocHit]
        Vector hits including their similarity score ∈ [0, 1].
    """

    # Wrapper method embeds the query for us
    raw_hits = _store_global.similarity_search_with_relevance_scores(query, k=k)
    filtered_hits: List[DocHit] = []
    for doc, score in raw_hits:
        if score_threshold is not None and score < score_threshold:
            continue
        meta = doc.metadata or {}
        filtered_hits.append(
            DocHit(
                id=meta.get("id", ""),
                pdf=meta.get("source", ""),
                pages=meta.get("pages", -1),
                text=doc.page_content,
                score=score,
            )
        )
    return filtered_hits


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
