#!/usr/bin/env python
"""
firegpt/langgraph_agent.py
==========================
LangGraph retrieval-augmented agent for **FireGPT** *with store-selection routing*.

Key additions
~~~~~~~~~~~~~
* **decide_retrieval** node - an LLM-powered router that returns *one* of
  ``{"none", "local", "global", "both"}``.
* Two vector stores: **local** and **global** SOP collections.
* Conditional edges that branch the graph based on the router’s decision.
"""

from __future__ import annotations

# ──────────────────────────────
# Standard Library
import asyncio
import operator
import os
import re
from pathlib import Path
from typing import List, Literal, Optional

# ──────────────────────────────
# Third-Party Libraries
from pydantic import BaseModel
from typing_extensions import Annotated
import json

from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient


# ──────────────────────────────
# Configuration
LOCAL_DB_PATH = Path(os.getenv("FGPT_DB_PATH", "stores/local"))
GLOBAL_DB_PATH = Path(os.getenv("FGPT_GLOBAL_DB_PATH", "stores/global"))

EMB_MODEL = Path(os.getenv("FGPT_EMBED_MODEL", "models/bge-base-en-v1.5"))
LOCAL_COLL = os.getenv("FGPT_COLLECTION", "fire_docs")
GLOBAL_COLL = os.getenv("FGPT_GLOBAL_COLLECTION", "global_docs")
TOP_K = int(os.getenv("FGPT_TOP_K", "5"))

MODEL_NAME = os.getenv("FGPT_MODEL_NAME", "qwen3:8b")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")

MCP_EP = os.getenv("FGPT_MCP_ENDPOINT", "http://localhost:7790")

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


# ──────────────────────────────
# Vector stores
_embedder = HuggingFaceEmbeddings(
    model_name=str(EMB_MODEL),
    model_kwargs={"local_files_only": True},
)
_store_local = Chroma(
    collection_name=LOCAL_COLL,
    persist_directory=str(LOCAL_DB_PATH),
    embedding_function=_embedder,
)
_store_global = Chroma(
    collection_name=GLOBAL_COLL,
    persist_directory=str(GLOBAL_DB_PATH),
    embedding_function=_embedder,
)


# ──────────────────────────────
# State model
class AgentState(BaseModel):
    """Data carried between nodes."""

    query: str = ""
    context_store: str = ""
    context: List[str] = []
    history: Annotated[List[dict], operator.add] = []
    tool_plan: Optional[List[dict]] = None


# ──────────────────────────────
# Helpers
def _strip_think(text: str) -> str:
    """Remove `<think>…</think>` sections (model’s private thoughts)."""
    return _THINK_RE.sub("", text).strip()


def _retrieve_from(store: Chroma, query: str) -> list[str]:
    """Search *store* and return page contents of top-k hits."""
    hits = store.similarity_search_with_relevance_scores(query, k=TOP_K)
    return [doc.page_content for doc, _ in hits]


# ──────────────────────────────
# Router node

#  Literal type alias for router output
Decision = Literal["none", "local", "global", "both"]


def decide_retrieval(state: AgentState) -> Decision:  # noqa: D401
    """
    Ask the LLM whether to consult the local and/or global SOP stores.

    Returns
    -------
    Decision
        One of ``"none"``, ``"local"``, ``"global"``, or ``"both"``.
    """
    system = {
        "role": "system",
        "content": (
            "You are a routing assistant. "
            "For the user question, decide which knowledge base is needed.\n"
            "Respond with **one** token, lowercase, chosen from:\n"
            "  none   - no retrieval needed (you already know the answer)\n"
            "  local  - consult the *local* SOP store only\n"
            "  global - consult the *global* SOP store only\n"
            "  both   - consult *both* stores\n"
            "Return just the word; no punctuation or explanation."
        ),
    }
    user = {"role": "user", "content": state.query}

    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        openai_api_base=OPENAI_API_BASE,
        openai_api_key="unused",
        temperature=0.0,
    )
    raw = llm.invoke([system, user]).content.strip().lower()

    # Fall back to conservative default
    decision: Decision
    if raw in {"none", "local", "global", "both"}:
        decision = raw  # type: ignore[assignment]
    else:
        decision = "both"  # safest

    return {"context_store": decision}


# ──────────────────────────────
# Retrieval nodes
def retrieve_local(state: AgentState) -> dict:  # noqa: D401
    context = _retrieve_from(_store_local, state.query)
    return {"context": context}


def retrieve_global(state: AgentState) -> dict:  # noqa: D401
    context = _retrieve_from(_store_global, state.query)
    return {"context": context}


def retrieve_both(state: AgentState) -> dict:  # noqa: D401
    ctx_local = _retrieve_from(_store_local, state.query)
    ctx_global = _retrieve_from(_store_global, state.query)
    return {"context": ctx_local + ctx_global}


# ──────────────────────────────
# MCP client & tool catalogue
mcp_client = MultiServerMCPClient(
        {
            "fire_mcp": {
                "url": "http://localhost:7790/mcp/",
                "transport": "streamable_http",
            }
        }
    )


async def load_mcp_tools() -> List[Tool]:
    """Fetch MCP tools asynchronously and tweak descriptions."""
    tools = await mcp_client.get_tools()
    for tool in tools:
        if tool.name == "retrieve_chunks":
            tool.description += (
                " Example queries: 'ICU evacuation with drones', "
                "'hospital indoor fire drone operations'."
            )
    return tools


async def plan_mcp(state: AgentState) -> dict:
    tools_list = await load_mcp_tools()                  # ← real list
    tools = {t.name: t for t in tools_list}              # name → tool obj

    catalogue_str = "\n".join(
        f"- {name}({', '.join(t.args_schema['properties'].keys())}) : {t.description}"
        for name, t in tools.items()
    ) or "(no tools)"
    print(catalogue_str)
    system = {
        "role": "system",
        "content": (
            "You are a planning assistant.\n"
            "Available MCP tools:\n"
            f"{catalogue_str}\n\n"
            'Return JSON: {"calls":[{"name":<tool>,"args":{...}}, …]} '
            'or {"calls":[]} if none.'
        ),
    }
    user = {"role": "user", "content": state.query}

    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        openai_api_base=OPENAI_API_BASE,
        openai_api_key="unused",
        temperature=0.0,
    )

    raw = llm.invoke([system, user]).content
    try:
        plan = json.loads(raw)
        assert isinstance(plan.get("calls"), list)
    except Exception:
        plan = {"calls": []}

    return {
        "tool_plan": plan["calls"],
        "tool_catalogue": tools,          # pass the dict forward
    }


async def exec_mcp(state: AgentState) -> dict:
    tools_list = await load_mcp_tools()                  # ← real list
    tools = {t.name: t for t in tools_list}      # dict from plan_mcp
    ctx_additions: list[str] = []

    for call in state.tool_plan or []:
        name = call.get("name")
        args = call.get("args", {})

        tool = tools.get(name)
        if tool is None:
            ctx_additions.append(f"(unknown tool '{name}')")
            continue

        try:
            # StructuredTool exposes .run() sync and .arun() async
            res = await tool.arun(**args) if tool.is_async else tool.run(**args)
            ctx_additions.append(f"### {name} output\n```json\n{res}\n```")
        except Exception as exc:
            ctx_additions.append(f"(tool {name} failed: {exc})")

    return {"context": state.context + ctx_additions}


# ──────────────────────────────
# Answer-synthesis node (unchanged)
async def synthesize(state: AgentState) -> dict:  # noqa: D401
    """Stream an answer and extend conversation history (excluding thoughts)."""
    system_msg = {
        "role": "system",
        "content": (
            "You are FireGPT, an expert assistant for wildfire prevention. "
            "Answer only from the provided context; if unsure, say you don't know."
        ),
    }

    context_block = "\n\n".join(f"- {chunk}" for chunk in state.context)
    user_msg = {
        "role": "user",
        "content": (
            f"### Context\n{context_block}\n\n"
            f"### Question\n{state.query}\n\n"
            "### Answer:"
        ),
    }

    prompt_msgs = [system_msg] + state.history + [user_msg]

    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_base=OPENAI_API_BASE,
        openai_api_key="unused",
        temperature=0.0,
    )

    print("🤖  FireGPT: ", end="", flush=True)
    reply_parts: List[str] = []

    async for chunk in llm.astream(prompt_msgs):
        token: str | None
        if isinstance(chunk, AIMessageChunk):
            token = getattr(chunk, "content", None)
        else:
            token = str(chunk)
        if token:
            reply_parts.append(token)
    print()

    assistant_reply_raw = "".join(reply_parts)
    assistant_reply_clean = _strip_think(assistant_reply_raw)

    new_history = state.history + [
        {"role": "user", "content": state.query},
        {"role": "assistant", "content": assistant_reply_clean},
    ]
    return {"history": new_history}


# ──────────────────────────────
# Graph definition
def build_graph():
    graph = StateGraph(state_schema=AgentState)

    # nodes
    graph.add_node("decide_store", decide_retrieval)
    graph.add_node("retrieve_local", retrieve_local)
    graph.add_node("retrieve_global", retrieve_global)
    graph.add_node("retrieve_both", retrieve_both)
    graph.add_node("plan_mcp", plan_mcp)
    graph.add_node("exec_mcp", exec_mcp)
    graph.add_node("synthesize", synthesize)

    # conditional routing
    graph.add_conditional_edges(
        "decide_store",
        lambda out: out.context_store,
        {
            "none": "synthesize",
            "local": "retrieve_local",
            "global": "retrieve_global",
            "both": "retrieve_both",
        },
    )

    # connect retrieval to plan_mcp
    for retr in ("retrieve_local", "retrieve_global", "retrieve_both"):
        graph.add_edge(retr, "plan_mcp")

    # if plan is not empty go to exec_mcp otherwise go to synthesize
    graph.add_conditional_edges(
        "plan_mcp",
        lambda out: "exec" if out.tool_plan else "skip",
        {"exec": "exec_mcp", "skip": "synthesize"},
    )

    graph.add_edge("exec_mcp", "synthesize")
    graph.add_edge("synthesize", END)

    graph.set_entry_point("decide_store")
    graph.set_finish_point("synthesize")

    return graph.compile()


# ──────────────────────────────
# CLI loop (unchanged)
async def main() -> None:  # noqa: D401
    agent = build_graph()
    state = AgentState()

    png_data = agent.get_graph().draw_mermaid_png()

    # Save it to a PNG file
    with open("agent_graph.png", "wb") as f:
        f.write(png_data)

    print("\n🔥  FireGPT (LangGraph) - type 'exit' to quit.\n")

    while True:
        user_input = input("👨‍🚒  You : ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋  Stay safe out there!")
            break

        state.query = user_input
        state = AgentState(**await agent.ainvoke(state))  # re-validate


if __name__ == "__main__":
    asyncio.run(main())
