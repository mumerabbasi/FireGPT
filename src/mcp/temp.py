#!/usr/bin/env python
"""
firegpt/src/mcp_server/mcp_server.py

FastMCP server exposing document retrieval (vector search) and
fire-danger assessment.
"""

import json
import logging

from forest_fire_gee import Client as FireGEE
from forest_fire_gee.models import (
    FireDangerRequest,
    FireDangerResponse,
    BoundingBox,
)
from forest_fire_gee.api.default import assess_fire_danger_assess_fire_danger_post
from forest_fire_gee.types import Response


LOG = logging.getLogger("firegpt.fastmcp")

# ----------------------------------------------------------------------------
# External Clients & Vector Store
# ----------------------------------------------------------------------------

# FireGEE API client
client = FireGEE(base_url="https://api.firefirefire.lol")


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
    subgrid_size_m: int = 500
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
    print("PAYLOAD")
    print(payload)
    print("PAYLOAD ENDED")
    response: Response[FireDangerResponse] = assess_fire_danger_assess_fire_danger_post.sync_detailed(
        client=client, body=payload
    )
    print(response)
    if response.status_code == 200:
        body = json.loads(response.content)
        return FireDangerResponse(**body)

    LOG.error("FireGEE API error: %s", response.status_code)
    return None


# ----------------------------------------------------------------------------
# Server Entry Point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    out = assess_fire_danger()
    print(out)
