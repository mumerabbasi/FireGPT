from __future__ import annotations

import logging
import os
from typing import List
import json
import asyncio
from src.mcp.forest_fire_gee.models import FireDangerResponse
from fastmcp import Client

# ---------------------------------------------------------------------------
# Config (env vars or sane defaults)
# ---------------------------------------------------------------------------
MCP_EP = os.getenv("MCP_EP", "http://localhost:7790/mcp")
TOP_K = int(os.getenv("FGPT_TOP_K", "1"))
MAX_NEW = int(os.getenv("FGPT_MAX_NEW", "600"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
LOG = logging.getLogger("firegpt.doc-agent")

# ---------------------------------------------------------------------------
# Fast-MCP client  (sync interface is fine for CLI)
# ---------------------------------------------------------------------------
cli = Client(
    transport=MCP_EP
)


async def fetch_chunks(question: str, k: int = TOP_K) -> List[dict]:
    """
    Call Fast-MCP tool `retrieve_chunks`.

    FastMCPClient lets you pass kwargs directly.
    """
    res = await cli.call_tool("retrieve_chunks_local", {"query": question, "k": k})
    hits_json = res[0].text
    return json.loads(hits_json)


async def fetch_fire_danger(
    top_left_lat: float = 47.6969,
    top_left_lon: float = 7.9468,
    bottom_right_lat: float = 47.7524,
    bottom_right_lon: float = 8.0347,
    subgrid_size_m: int = 100,
    forecast_hours: int = 3,
    poi_search_buffer_m: int = 0,
) -> FireDangerResponse:
    """
    Call the Fast-MCP tool `assess_fire_danger` and parse its response
    into a Pydantic FireDangerResponse.
    """
    payload = {
        "top_left_lat": top_left_lat,
        "top_left_lon": top_left_lon,
        "bottom_right_lat": bottom_right_lat,
        "bottom_right_lon": bottom_right_lon,
    }
    # Invoke the MCP tool
    res = await cli.call_tool("assess_fire_danger", payload)
    return res


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
async def main() -> None:
    print("FireGPT (PDF-only)  -  type a question, Ctrl-C to quit\n")
    try:
        async with cli:
            while True:
                q = input("You: ").strip()
                if not q:
                    continue

                chunks = await fetch_fire_danger()
                if not chunks:
                    print("No relevant document sections found.\n")
                    continue

                print(chunks)

    except KeyboardInterrupt:
        print("\nBye")


if __name__ == "__main__":
    asyncio.run(main())
