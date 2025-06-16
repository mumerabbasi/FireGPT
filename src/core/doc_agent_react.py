#!/usr/bin/env python
"""
firegpt/agent/doc_agent.py
--------------------------
CLI agent with step-by-step reasoning, tool-use and long-term memory.

* Talks to a local FastMCP 2.x server for tools:
    - retrieve_chunks
    - gee_grid
    - risk_score
    - plan_fire_mission   (extend as you add tools)

* Runs a local Mistral-7B-Instruct-v0.3 model for the LLM brain.
* Stores past “FinalAnswer” plans in `data/memory/memory.jsonl`
  and retrieves the top-K similar ones as long-term context.

Usage
-----
$ fastmcp run firegpt/mcp_server/main.py      # in another shell
$ python -m firegpt.agent.doc_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import transformers
from fastmcp import Client
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Config (env vars or sane defaults)
# ---------------------------------------------------------------------------
MCP_EP = os.getenv("MCP_EP", "http://localhost:7790/mcp")
MODEL_PATH = os.getenv("FGPT_MISTRAL_PATH", "models/mistral")
TOP_K_DOCS = int(os.getenv("FGPT_TOP_K", "3"))
MAX_NEW = int(os.getenv("FGPT_MAX_NEW", "600"))
MEMORY_DIR = Path(os.getenv("FGPT_MEM_DIR", "data/memory"))
MEMORY_TOP_K = int(os.getenv("FGPT_MEM_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
LOG = logging.getLogger("firegpt.doc_agent")

# ---------------------------------------------------------------------------
# FastMCP client
# ---------------------------------------------------------------------------
cli = Client(transport=MCP_EP)

# ---------------------------------------------------------------------------
# Local Mistral model
# ---------------------------------------------------------------------------
LOG.info("Loading Mistral model from %s …", MODEL_PATH)
pipe = transformers.pipeline(
    "text-generation",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
    temperature=None,
    do_sample=False,
    max_new_tokens=MAX_NEW,
    pad_token_id=transformers.AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True
    ).eos_token_id,
)

# ---------------------------------------------------------------------------
# Embedding model for memory similarity search
# ---------------------------------------------------------------------------
LOG.info("Loading MiniLM embedder for memory search …")
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=True
)

# ---------------------------------------------------------------------------
# Tool registry (name → schema → call-lambda)
# ---------------------------------------------------------------------------
ToolCall = Dict[str, Any]
ToolResult = Any


@dataclass
class ToolSpec:
    """Metadata + executor for each available tool."""

    name: str
    description: str
    schema: Dict[str, Any]
    fn: Any  # coroutine or normal callable


def _tool(
    name: str,
    description: str,
    schema: Dict[str, Any] | None = None,
):
    """Decorator to register a tool easily."""

    def registrar(func):
        TOOL_REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            schema=schema or {},
            fn=func,
        )
        return func

    return registrar


TOOL_REGISTRY: Dict[str, ToolSpec] = {}


@_tool(
    name="retrieve_chunks",
    description="Semantic search over PDF summaries. "
    "Args: query (str), k (int, default 3). Returns list of hits.",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer"},
        },
        "required": ["query"],
    },
)
async def tool_retrieve_chunks(query: str, k: int = TOP_K_DOCS) -> ToolResult:
    res = await cli.call_tool("retrieve_chunks", {"query": query, "k": k})
    return json.loads(res[0].text)


@_tool(
    name="gee_grid",
    description="Return the cached geo grid (cells with slope, fuel, etc.) for "
    "a region of interest. Args: roi (str, bbox id). Returns GeoJSON.",
)
async def tool_gee_grid(roi: str) -> ToolResult:
    res = await cli.call_tool("gee_grid", {"roi": roi})
    return json.loads(res[0].text)


@_tool(
    name="risk_score",
    description="Get the wildfire risk score for a cell. "
    "Args: cell_id (str). Returns float 0-1.",
    schema={
        "type": "object",
        "properties": {"cell_id": {"type": "string"}},
        "required": ["cell_id"],
    },
)
async def tool_risk_score(cell_id: str) -> ToolResult:
    res = await cli.call_tool("risk_score", {"cell_id": cell_id})
    return json.loads(res[0].text)


@_tool(
    name="plan_fire_mission",
    description="Generate way-points for each drone to fight the given fire. "
    "Args: bbox (str, lat1,lon1,lat2,lon2). Returns JSON plan.",
)
async def tool_plan_fire_mission(bbox: str) -> ToolResult:
    res = await cli.call_tool("plan_fire_mission", {"bbox": bbox})
    return json.loads(res[0].text)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _ensure_mem_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


MEM_FILE = MEMORY_DIR / "memory.jsonl"


def _append_memory(text: str) -> None:
    """Store a line of text in JSONL with its embedding vector."""
    _ensure_mem_dir()
    vec = embedder.encode(text).tolist()
    with MEM_FILE.open("a", encoding="utf-8") as fh:
        json.dump({"t": datetime.utcnow().isoformat(), "text": text, "vec": vec}, fh)
        fh.write("\n")


def _retrieve_memories(query: str, top_k: int = MEMORY_TOP_K) -> List[str]:
    """Return up to top_k memory snippets most similar to the query."""
    if not MEM_FILE.exists():
        return []

    query_vec = embedder.encode(query)
    texts, vecs = [], []
    with MEM_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            texts.append(rec["text"])
            vecs.append(rec["vec"])
    if not vecs:
        return []

    dists = cdist([query_vec], np.array(vecs), metric="cosine")[0]
    idx = np.argsort(dists)[:top_k]
    return [texts[i] for i in idx]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
TOOL_DESCRIPTIONS = "\n".join(
    f"- **{spec.name}**: {spec.description}" for spec in TOOL_REGISTRY.values()
)

SYSTEM_PROMPT = (
    "You are FireMission-GPT, an expert wildfire mission planner.\n"
    "You have access to the following TOOLS:\n"
    f"{TOOL_DESCRIPTIONS}\n\n"
    "Follow the rules:\n"
    "1. Think step-by-step. After each Thought, decide whether you need a tool.\n"
    "2. If yes, output exactly two lines:\n"
    "   Action: <tool_name>\n"
    "   Arguments: <JSON matching the tool's schema>\n"
    "3. I will execute the tool and return the Observation.\n"
    "4. Repeat Thought/Action/Observation as needed.\n"
    "5. Finish with:\n"
    "   FinalAnswer: <concise mission plan (bullets), with PDF citations>\n"
    "Do NOT invent tool names. Only choose from the list above.\n"
    "Chain-of-thought will be hidden from the operator.\n"
)


def _format_messages(msgs: List[Dict[str, str]]) -> str:
    """Simple prepend-role format compatible with vanilla generation."""
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs) + "\nAssistant:"


# ---------------------------------------------------------------------------
# Controller loop
# ---------------------------------------------------------------------------
MAX_STEPS = 8
OBS_TRUNC = 800  # chars


async def run_agent(question: str) -> str:
    """Main ReAct controller."""
    memories = _retrieve_memories(question)
    if memories:
        mem_block = "Long-term memory snippets:\n" + "\n".join(f"- {m}" for m in memories)
    else:
        mem_block = ""

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if mem_block:
        messages.append({"role": "system", "content": mem_block})

    messages.append({"role": "user", "content": question})

    for step in range(MAX_STEPS):
        prompt = _format_messages(messages)
        generation = pipe(prompt, return_full_text=False)[0]["generated_text"]
        assistant_reply = generation.strip()
        messages.append({"role": "assistant", "content": assistant_reply})

        # --- Check for FinalAnswer
        if assistant_reply.startswith("FinalAnswer:"):
            final = assistant_reply[len("FinalAnswer:"):].strip()
            _append_memory(final)  # save to long-term store
            return final

        # --- Parse Action
        if assistant_reply.startswith("Action:"):
            try:
                lines = assistant_reply.splitlines()
                action_name = lines[0].split("Action:", 1)[1].strip()
                args_json = "\n".join(lines[1:]).split("Arguments:", 1)[1].strip()
                args = json.loads(args_json)
            except (IndexError, json.JSONDecodeError) as exc:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Observation: ERROR parsing tool call: {exc}",
                    }
                )
                continue

            spec = TOOL_REGISTRY.get(action_name)
            if spec is None:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Observation: ERROR unknown tool {action_name}",
                    }
                )
                continue

            try:
                result = await spec.fn(**args)  # type: ignore[arg-type]
                result_str = json.dumps(result)[:OBS_TRUNC]
            except Exception as exc:  # pylint: disable=broad-except
                LOG.exception("Tool %s failed", action_name)
                result_str = f"ERROR running tool: {exc}"

            messages.append({"role": "system", "content": f"Observation: {result_str}"})
            continue

        # If neither FinalAnswer nor Action ⇒ model forgot rules
        messages.append(
            {
                "role": "system",
                "content": "Observation: You must output either an Action or FinalAnswer.",
            }
        )

    return "Error: reached tool-use step limit without FinalAnswer."


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------
async def main() -> None:
    LOG.info("Connected to MCP %s. Ready.", MCP_EP)
    print("FireMission-GPT  –  type a question, Ctrl-C to quit\n")
    try:
        async with cli:
            while True:
                user_q = input("You: ").strip()
                if not user_q:
                    continue
                answer = await run_agent(user_q)
                print(f"\nAssistant: {answer}\n")
    except KeyboardInterrupt:
        print("\nBye")


if __name__ == "__main__":
    asyncio.run(main())
