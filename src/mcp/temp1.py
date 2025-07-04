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
from forest_fire_gee.models import *
from forest_fire_gee.api.default import assess_fire_danger_assess_fire_danger_post
from forest_fire_gee.types import *
import json

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


def assess_fire_danger(
    top_left_lat: float = 47.6969,
    top_left_lon: float = 7.9468,
    bottom_right_lat: float = 47.7524,
    bottom_right_lon: float = 8.0347,
    subgrid_size_m=100,  # Default value
    forecast_hours=3,  # Default value
    poi_search_buffer_m=0,  # Default value
) -> FireDangerResponse:
    """
    Assess fire danger at a given bounding box using the FireGEE API. Required parameters:
    - top_left_lat: Latitude of the top-left corner of the bounding box.
    - top_left_lon: Longitude of the top-left corner of the bounding box.
    - bottom_right_lat: Latitude of the bottom-right corner of the bounding box.
    - bottom_right_lon: Longitude of the bottom-right corner of the bounding box.
    - subgrid_size_m: Size of the subgrid in meters. Default is 100.
    - forecast_hours: Number of hours into the future for the GFS forecast. Default is 3.
    - poi_search_buffer_m: Buffer distance in meters outside the main bounding box to search for Points of Interest. Default is 0.
    Uses the FireGEE API to get the fire danger assessment.
    """
    bbox = BoundingBox(
        top_left_lat=top_left_lat,
        top_left_lon=top_left_lon,
        bottom_right_lat=bottom_right_lat,
        bottom_right_lon=bottom_right_lon,)
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
    return json.loads(response.content) if response.content == 200 else None

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    resposne = assess_fire_danger()
    print(resposne)