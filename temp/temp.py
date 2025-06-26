#!/usr/bin/env python
"""
firegpt/src/agent/agent_react_full.py – simplified graph v2 (PEP‑8)

LangGraph Agentic‑RAG pipeline for drone‑assisted fire‑fighting.
Safety and geo‑risk checks are now delegated to the ReAct agent itself, so the
explicit ``safety_guard`` and ``fire_assess`` nodes were removed. The agent
still has access to the ``assess_fire_danger`` tool whenever it needs it.

**Fixes in this version**
1. Prevents infinite rewrite/grade loop that triggered a
   ``GraphRecursionError`` by adding a boolean ``rewritten`` flag in
   :class:`AgentState`. Once a query has been rewritten, ``grade_decider`` will
   always route to ``react_agent``.
2. **Type‑safe message history** – ``AgentState.messages`` now stores real
   LangChain ``BaseMessage`` objects instead of plain dictionaries. This avoids
   the ``ValidationError`` that occurred when downstream nodes appended
   ``HumanMessage``/``AIMessage``/``ToolMessage`` instances.
3. Helper utilities ``last_user_content`` and ``append_human`` centralise
   message handling.
4. Small docstring / prompt tweaks and PEP‑8 formatting.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path
from typing import List, Literal

from langchain_chroma import Chroma
from langchain_core.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import Tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
EMB_MODEL = Path(os.getenv("FGPT_EMBED_MODEL", "models/bge-base-en-v1.5"))
TOP_K = int(os.getenv("FGPT_TOP_K", "5"))
OPENAI_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")

# ──────────────────────────────────────────────────────────────
# Embeddings & retriever factory
# ──────────────────────────────────────────────────────────────
_EMBEDDER = HuggingFaceEmbeddings(
    model_name=str(EMB_MODEL), model_kwargs={"local_files_only": True}
)


def build_retriever(store_path: str, collection: str):
    """Return a Chroma retriever with similarity + score‑threshold search."""
    store = Chroma(
        collection_name=collection,
        persist_directory=store_path,
        embedding_function=_EMBEDDER,
    )
    return store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": TOP_K, "score_threshold": 0.2},
    )


RETRIEVERS = {
    "regional": build_retriever("stores/local", "fire_docs"),
    # add global/session stores if/when available
}


# ──────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────

def last_user_content(messages: List[BaseMessage]) -> str:
    """Return the content of the most recent *human* message."""
    for msg in reversed(messages):
        if msg.type == "human":
            return msg.content
    return ""


def append_human(messages: List[BaseMessage], text: str) -> None:
    """Append a :class:`HumanMessage` to the history."""
    messages.append(HumanMessage(content=text))


# ──────────────────────────────────────────────────────────────
# Agent state schema (with rewrite flag)
# ──────────────────────────────────────────────────────────────


class AgentState(BaseModel):
    """Shared state passed along the LangGraph edges."""

    messages: List[BaseMessage] = Field(default_factory=list)
    store: Literal["regional", "global", "session"] | None = None
    retrieved_docs: List[Document] = Field(default_factory=list)
    rewritten: bool = False  # prevent recursion


# ──────────────────────────────────────────────────────────────
# Tool loader (Fast‑MCP)
# ──────────────────────────────────────────────────────────────


async def load_mcp_tools() -> List[Tool]:
    """Fetch MCP tools asynchronously and tweak descriptions."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "fire_mcp": {
                "url": "http://localhost:7790/mcp/",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    for tool in tools:
        if tool.name == "retrieve_chunks":
            tool.description += (
                " Example queries: 'ICU evacuation with drones', "
                "'hospital indoor fire drone operations'."
            )
    return tools


# ──────────────────────────────────────────────────────────────
# Router node – choose retriever store
# ──────────────────────────────────────────────────────────────

_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Choose the best knowledge store for the user question. "
            "Return ONLY `regional`, `global`, or `session`.",
        ),
        MessagesPlaceholder("messages"),
    ]
)
_ROUTER_LLM = ChatOpenAI(
    model_name="qwen3:8b",
    temperature=0.0,
    openai_api_base=OPENAI_BASE,
    openai_api_key="unused",
)
_ROUTER_CHAIN: Runnable = _ROUTER_PROMPT | _ROUTER_LLM | (
    lambda x: x.content.strip()
)


def router_node(state: AgentState) -> AgentState:
    choice = _ROUTER_CHAIN.invoke({"messages": state.messages})
    state.store = choice if choice in RETRIEVERS else "regional"
    return state


# ──────────────────────────────────────────────────────────────
# Retrieval node
# ──────────────────────────────────────────────────────────────

def retrieval_node(state: AgentState) -> AgentState:
    query = last_user_content(state.messages)
    state.retrieved_docs = RETRIEVERS[state.store].invoke(query)
    return state


# ──────────────────────────────────────────────────────────────
# Document quality grader & branch decider
# ──────────────────────────────────────────────────────────────

_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Return true if the provided documents are irrelevant or "
            "low quality, otherwise return false.",
        ),
        ("user", "Question: {question}\n\nDocuments: {docs}"),
    ]
)
_GRADE_LLM = ChatOpenAI(
    model_name="qwen3:8b",
    temperature=0.0,
    openai_api_base=OPENAI_BASE,
    openai_api_key="unused",
)
_GRADE_CHAIN: Runnable = _GRADE_PROMPT | _GRADE_LLM


def grade_decider(state: AgentState) -> str:
    """Return the next node label based on document quality."""
    if state.rewritten:
        return "react_agent"  # avoid loops
    doc_snip = " ".join(doc.page_content[:500] for doc in state.retrieved_docs)
    resp = _GRADE_CHAIN.invoke(
        {
            "question": last_user_content(state.messages),
            "docs": doc_snip,
        }
    )
    is_bad = "true" in resp.content.lower()
    return "rewrite_question" if is_bad else "react_agent"


def grade_passthrough(state: AgentState) -> AgentState:  # noqa: D401
    """No‑op passthrough so we can attach conditional edges."""
    return state


# ──────────────────────────────────────────────────────────────
# Rewrite‑question node
# ──────────────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's question for better search. "
            "Respond with the improved query text only.",
        ),
        MessagesPlaceholder("messages"),
    ]
)
_REWRITE_LLM = ChatOpenAI(
    model_name="qwen3:8b",
    temperature=0.2,
    openai_api_base=OPENAI_BASE,
    openai_api_key="unused",
)
_REWRITE_CHAIN: Runnable = _REWRITE_PROMPT | _REWRITE_LLM


def rewrite_node(state: AgentState) -> AgentState:
    new_query = _REWRITE_CHAIN.invoke({"messages": state.messages}).content.strip()
    append_human(state.messages, new_query)
    state.rewritten = True
    return state


# ──────────────────────────────────────────────────────────────
# ReAct planner node – agent reasons about safety & risk
# ──────────────────────────────────────────────────────────────


async def build_react_agent(tools: List[Tool]) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are FireGPT. Use `retrieve_chunks` for SOPs and "
                "`assess_fire_danger` for geo‑risk when relevant. Always confirm "
                "requested payloads do not exceed **50 L per drone** and refuse "
                "if unsafe. Think step‑by‑step and cite any tools you call.",
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    llm = ChatOpenAI(
        model_name="qwen3:8b",
        streaming=True,
        temperature=0.1,
        openai_api_base=OPENAI_BASE,
        openai_api_key="unused",
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={"tool_choice": "any"},
    )
    return create_react_agent(llm, tools, prompt=prompt)


# ──────────────────────────────────────────────────────────────
# ReAct wrapper node (ensures AgentState round‑trip)
# ──────────────────────────────────────────────────────────────


def make_react_node(react_core: Runnable) -> RunnableLambda:  # noqa: D401
    """Wrap the raw runnable so LangGraph always gets back an AgentState."""

    async def _node(state: AgentState) -> AgentState:  # noqa: D401
        outputs = await react_core.ainvoke({"messages": state.messages})
        # create_react_agent returns either a single BaseMessage or a list
        if isinstance(outputs, BaseMessage):
            state.messages.append(outputs)
        elif isinstance(outputs, list):
            state.messages.extend(outputs)
        else:  # pragma: no cover – unexpected types
            raise TypeError("Unexpected return from ReAct agent")
        return state

    return RunnableLambda(_node)


# ──────────────────────────────────────────────────────────────
# Synthesiser node
# ──────────────────────────────────────────────────────────────

_SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Draft a concise SOP answer. Include Regulatory, Medical‑gas, "
            "Battery‑swap, Decontamination, Comms sections.",
        ),
        ("user", "{question}"),
        ("assistant", "{docs}"),
    ]
)
_SYNTH_LLM = ChatOpenAI(
    model_name="qwen3:8b",
    temperature=0.2,
    openai_api_base=OPENAI_BASE,
    openai_api_key="unused",
)


def synth_node(state: AgentState) -> str:
    docs_txt = "\n\n".join(
        doc.page_content[:800] for doc in state.retrieved_docs[:3]
    )
    return _SYNTH_LLM.invoke(
        _SYNTH_PROMPT.format_prompt(
            question=last_user_content(state.messages),
            docs=docs_txt,
        )
    ).content


# ──────────────────────────────────────────────────────────────
# Build StateGraph
# ──────────────────────────────────────────────────────────────


async def build_graph() -> Runnable:
    tools = await load_mcp_tools()
    react_core = await build_react_agent(tools)

    graph = StateGraph(AgentState)

    # nodes
    graph.add_node("router", RunnableLambda(router_node))
    graph.add_node("retriever", RunnableLambda(retrieval_node))
    graph.add_node("grade_documents", RunnableLambda(grade_passthrough))
    graph.add_node("rewrite_question", RunnableLambda(rewrite_node))
    graph.add_node("react_agent", make_react_node(react_core))
    graph.add_node("synthesiser", RunnableLambda(synth_node))

    # linear edges
    graph.add_edge("router", "retriever")
    graph.add_edge("rewrite_question", "retriever")
    graph.add_edge("react_agent", "synthesiser")

    # grading branch
    graph.add_edge("retriever", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        RunnableLambda(grade_decider),
        {
            "rewrite_question": "rewrite_question",
            "react_agent": "react_agent",
        },
    )

    graph.set_entry_point("router")
    graph.set_finish_point("synthesiser")
    return graph.compile()


# ──────────────────────────────────────────────────────────────
# CLI helper – optional Mermaid PNG
# ──────────────────────────────────────────────────────────────

def save_graph_png(graph: Runnable, path: str = "agent_graph.png") -> None:
    """Save a Mermaid diagram of the LangGraph to *path* if PIL is present."""
    try:
        from PIL import Image

        png_bytes = graph.get_graph().draw_mermaid_png()
        Image.open(io.BytesIO(png_bytes)).save(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not save graph image: {exc}")


# ──────────────────────────────────────────────────────────────
# Async CLI loop
# ──────────────────────────────────────────────────────────────


async def main_cli() -> None:
    """Interactive CLI – type 'exit' to quit."""
    graph = await build_graph()
    save_graph_png(graph)

    state = AgentState()
    print("FireGPT (async) — type 'exit' to quit.")
    while True:
        try:
            user = input("You : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("Goodbye!")
            break
        if user.lower() in {"exit", "quit"}:
            break
        append_human(state.messages, user)

        async for chunk in graph.astream(state):
            if isinstance(chunk, str):
                sys.stdout.write(chunk)
                sys.stdout.flush()
        print()  # newline after each answer


if __name__ == "__main__":
    asyncio.run(main_cli())
