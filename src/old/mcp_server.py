#!/usr/bin/env python
"""
firegpt/mcp_server/main.py
TODO: Implement graceful shutdown
"""
import json
import logging
import os
from pathlib import Path

import chromadb
from fastmcp import FastMCP
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


from forest_fire_gee import Client as FireGEE
from forest_fire_gee.models import FireDangerRequest, FireDangerResponse, BoundingBox
from forest_fire_gee.api.default import assess_fire_danger_assess_fire_danger_post
from forest_fire_gee.types import Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = Path(os.getenv("FGPT_DB_PATH", "data/db/chroma"))
EMB_PATH = Path(os.getenv("FGPT_EMBED_PATH", "models/minilm"))
COLL_NAME = os.getenv("FGPT_COLLECTION", "fire_docs")
TOP_K = int(os.getenv("FGPT_TOP_K", "5"))

HOST = os.getenv("FGPT_HOST", "0.0.0.0")
PORT = int(os.getenv("FGPT_PORT", "7790"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
LOG = logging.getLogger("firegpt.fastmcp")


# ---------------------------------------------------------------------------
# Instantiations
# ---------------------------------------------------------------------------
client = FireGEE(base_url="https://api.firefirefire.lol")


# ---------------------------------------------------------------------------
# Load vector store & embedder once at startup
# ---------------------------------------------------------------------------
LOG.info("Opening ChromaDB at %s …", DB_PATH)
_coll = chromadb.PersistentClient(str(DB_PATH)).get_collection(COLL_NAME)
LOG.info("Collection “%s” loaded with %d chunks", COLL_NAME, _coll.count())

LOG.info("Loading MiniLM embedder from %s …", EMB_PATH)
_EMB = SentenceTransformer(str(EMB_PATH), local_files_only=True)


# ---------------------------------------------------------------------------
# Pydantic schemas (tool / resource I-O)
# ---------------------------------------------------------------------------
class DocHit(BaseModel):
    id: str
    pdf: str
    pages: str
    summary: str
    full_text: str


class MetaEntry(BaseModel):
    id: str
    pdf: str
    pages: str


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP("FireGPT-All-In-One")


# ---------------------------------------------------------------------------
# Document retrieval tools
# ---------------------------------------------------------------------------
@mcp.resource("data://docs/metadata")
def docs_metadata() -> list[MetaEntry]:
    """Metadata for every ingested section (id, pdf, pages)."""
    meta = _coll.get(include=["metadatas", "ids"])
    return [
        MetaEntry(id=i, pdf=m["pdf"], pages=m["pages"])
        for i, m in zip(meta["ids"], meta["metadatas"])
    ]


@mcp.tool
def retrieve_chunks(query: str, k: int = TOP_K) -> list[DocHit]:
    """Semantic top-k search over section summaries."""
    qvec = _EMB.encode(query, normalize_embeddings=True).tolist()
    res = _coll.query(query_embeddings=[qvec], n_results=k)  # list of lists for batch computation
    return [
        DocHit(
            id=i,
            pdf=m["pdf"],
            pages=m["pages"],
            summary=doc,
            full_text=m.get("full_text", ""),
        )
        for i, doc, m in zip(
            res["ids"][0],  # because we are using a single query
            res["documents"][0],
            res["metadatas"][0],
        )
    ]


# ---------------------------------------------------------------------------
# GEOSPATIAL PLACE-HOLDERS  (wire-up later)
# ---------------------------------------------------------------------------
@mcp.tool
def assess_fire_danger(
    top_left_lat: float = 47.6969,
    top_left_lon: float = 7.9468,
    bottom_right_lat: float = 47.7524,
    bottom_right_lon: float = 8.0347,
    subgrid_size_m=100,
    forecast_hours=3,
    poi_search_buffer_m=0,
) -> FireDangerResponse:
    """
    Assess fire danger at a given bounding box using the FireGEE API. Required parameters:
    - top_left_lat: Latitude of the top-left corner of the bounding box.
    - top_left_lon: Longitude of the top-left corner of the bounding box.
    - bottom_right_lat: Latitude of the bottom-right corner of the bounding box.
    - bottom_right_lon: Longitude of the bottom-right corner of the bounding box.
    - subgrid_size_m: Size of the subgrid in meters. Default is 100.
    - forecast_hours: Number of hours into the future for the GFS forecast. Default is 3.
    - poi_search_buffer_m: Buffer distance in meters outside the main bounding box to search for Points of Interest.
                           Default is 0.
    Uses the FireGEE API to get the fire danger assessment.
    It divides the given bbox into a grid. At the center of each grid cell, it calcualtes params like vegetation type,
    vegetation amount, slope, weather, wind speed and direction etc. This could tell in which direction, fire is most
    likely to spread. Also, it checks if there are any critical buildings near or inside the bounding box. Based on
    that, it calculates a risk score for each of those buildings.

    Returns a json.
    """
    bbox = BoundingBox(
        top_left_lat=top_left_lat,
        top_left_lon=top_left_lon,
        bottom_right_lat=bottom_right_lat,
        bottom_right_lon=bottom_right_lon
    )
    payload = FireDangerRequest(
        bbox=bbox,  # Not used in this context
        subgrid_size_m=subgrid_size_m,  # Default value
        forecast_hours=forecast_hours,  # Default value
        poi_search_buffer_m=poi_search_buffer_m,  # Default value
    )
    response: Response[FireDangerResponse] = assess_fire_danger_assess_fire_danger_post.sync_detailed(
            client=client,
            body=payload,
    )
    print(response)
    return json.loads(response.content) if response.content == 200 else None


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    LOG.info("Starting FireGPT FastMCP server ...")
    d = DocHit.model_json_schema()
    mcp.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
    )
