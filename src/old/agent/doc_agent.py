#!/usr/bin/env python
"""
firegpt/agent/doc_agent.py
--------------------------
CLI agent that answers questions *only* from the PDFs ingested in ChromaDB,
talking to the **FastMCP 2.x** server you now run with

    $ fastmcp run firegpt/mcp_server/main.py

Workflow
-----------
1. Calls the Fast-MCP tool **`retrieve_chunks`** to obtain top-k summaries.
2. Builds a prompt and streams an answer from a *local*
   **Mistral-7B-Instruct-v0.3** model.
"""
from __future__ import annotations

import logging
import os
from typing import List
import json
import asyncio
from src.mcp.forest_fire_gee.models import FireDangerResponse


import transformers
from fastmcp import Client

# ---------------------------------------------------------------------------
# Config (env vars or sane defaults)
# ---------------------------------------------------------------------------
MCP_EP = os.getenv("MCP_EP", "http://localhost:7790/mcp")
MISTRAL_PATH = os.getenv("FGPT_MISTRAL_PATH", "models/mistral")
TOP_K = int(os.getenv("FGPT_TOP_K", "5"))
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

# ---------------------------------------------------------------------------
# Local Mistral model (fp16; single GPU)
# ---------------------------------------------------------------------------
LOG.info("Loading Mistral model from %s …", MISTRAL_PATH)
pipe = transformers.pipeline(
    "text-generation",
    model=MISTRAL_PATH,
    tokenizer=MISTRAL_PATH,
    device_map=None,  # -- force single GPU, adjust if multi-GPU-auto desired
    torch_dtype="auto",
    temperature=None,
    do_sample=True,
    max_new_tokens=MAX_NEW,
    pad_token_id=transformers.AutoTokenizer.from_pretrained(
        MISTRAL_PATH, local_files_only=True
    ).eos_token_id,
)

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------
SYS_PROMPT = (
    "You are FireGPT, an assistant for wildfire drone operators. "
    "Answer ONLY using the context below and cite the PDF name + page range "
    "in square brackets after each fact.\n"
)


async def fetch_chunks(question: str, k: int = TOP_K) -> List[dict]:
    """
    Call Fast-MCP tool `retrieve_chunks`.

    FastMCPClient lets you pass kwargs directly.
    """
    res = await cli.call_tool("retrieve_chunks", {"query": question, "k": k})
    hits_json = res[0].text  # TextContent → the JSON string
    return json.loads(hits_json)  # -> list of dicts


async def fetch_fire_danger(
    top_left_lat: float,
    top_left_lon: float,
    bottom_right_lat: float,
    bottom_right_lon: float,
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
        "subgrid_size_m": subgrid_size_m,
        "forecast_hours": forecast_hours,
        "poi_search_buffer_m": poi_search_buffer_m,
    }
    # Invoke the MCP tool
    res = await cli.call_tool("assess_fire_danger", payload)
    return res


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = [
        f"[{c['pdf']} {c['pages']}]\n{c['text']}".strip()
        for c in chunks
    ]
    context = "\n\n".join(context_parts)

    return (
        SYS_PROMPT
        + "\nContext:\n"
        + context
        + f"\n\nUser: {question}\nAssistant:"
    )


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

                chunks = await fetch_chunks(q)
                if not chunks:
                    print("No relevant document sections found.\n")
                    continue

                prompt = build_prompt(q, chunks)
                generation = pipe(prompt, return_full_text=False)[0]["generated_text"]
                print("\nAssistant:", generation.strip(), "\n")

    except KeyboardInterrupt:
        print("\nBye")


if __name__ == "__main__":
    asyncio.run(main())
