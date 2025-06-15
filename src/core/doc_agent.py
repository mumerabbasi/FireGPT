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

Quick-start
-----------
$ python -m firegpt.agent.doc_agent
You: why must drops be 50 m high?

Environment overrides (optional)
--------------------------------
FGPT_MISTRAL_PATH   models/mistral          # local dir with v0.3 weights
FGPT_TOP_K          5                       # chunks per query
FGPT_MAX_NEW        350                     # max generated tokens
"""
from __future__ import annotations

import logging
import os
from typing import List
from pathlib import Path
import json

import transformers
from fastmcp import Client

# ---------------------------------------------------------------------------
# Config (env vars or sane defaults)
# ---------------------------------------------------------------------------
MCP_EP = os.getenv("MCP_EP", "http://localhost:7790/mcp")
MISTRAL_PATH = os.getenv("FGPT_MISTRAL_PATH", "models/mistral")
TOP_K = int(os.getenv("FGPT_TOP_K", "2"))
MAX_NEW = int(os.getenv("FGPT_MAX_NEW", "500"))

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
    device_map=None,          # -- force single GPU, adjust if multi-GPU-auto desired
    torch_dtype="auto",
    temperature=None,
    do_sample=False,
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


def build_prompt(question: str, chunks: List[dict]) -> str:
    """
    Construct the full prompt for the LLM by stitching together:
    1) a fixed system instruction,
    2) the raw text of each retrieved document chunk (with PDF name & pages as citation),
    and 3) the user's question.
    """
    # Assemble context from each chunk’s full-text file
    context_parts = []
    for c in chunks:
        text = Path(c["full_text"]).read_text(encoding="utf-8").strip()
        citation = f"[{c['pdf']} {c['pages']}]"
        context_parts.append(f"{citation}\n{text}")

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
                generated = ""
                for token in pipe(prompt):
                    part = token["generated_text"]
                    if not generated and part.startswith(prompt):
                        part = part[len(prompt):]
                    print(part, end="", flush=True)
                    generated += part
                print("\n")

    except KeyboardInterrupt:
        print("\nBye")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
