"""fire_agent_async.py
================================

Async CLI wrapper around a ReAct-style LangGraph agent for **wild-fire
incident response**.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai.chat_models import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

__all__ = ["run_cli"]

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------

MCP_SERVERS = {
    "fire_mcp": {
        "url": os.getenv("FGPT_MCP_URL", "http://localhost:7790/mcp/"),
        "transport": "streamable_http",
    }
}

OPENAI_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
OPENAI_MODEL = os.getenv("FGPT_MODEL", "qwen3:8b")

# -----------------------------------------------------------
# Demo geometry (will be injected by GUI later)
# -----------------------------------------------------------

HARDCODED_POIS: List[dict[str, Any]] = [
    {"lat": 47.70, "lon": 7.95, "note": "Critical power sub-station - protect"},
    {"lat": 47.71, "lon": 7.99, "note": "Regional hospital - protect"},
]
HARDCODED_BBOX: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (47.6969, 7.9468),  # top-left  (lat, lon)
    (47.7024, 7.9901),  # bottom-right
)

# -----------------------------------------------------------
# Dataclass for context (kept for future GUI integration)
# -----------------------------------------------------------


@dataclass(eq=True, frozen=True)
class IncidentContext:
    """Structured spatial context for the current turn."""

    bbox: Tuple[Tuple[float, float], Tuple[float, float]] | None = None
    pois: List[dict[str, Any]] | None = None

    @property
    def is_empty(self) -> bool:
        return self.bbox is None and (not self.pois)


# -----------------------------------------------------------
# Prompts
# -----------------------------------------------------------
BASE_RULES = (
    """
    You are **FireGPT**, a professional wildfire-incident assistant. Accuracy and
    safety are paramount.

    ### General principles
    * Respond only with information that is justified by
    - tool outputs,
    - the conversation so far, **or**
    - your own background knowledge **when the question is *not* about the current
        incident.** If you are uncertain, say so instead of inventing facts.

    ### Tools
    * `assess_fire_danger(bbox)` — returns per-cell danger metrics.
    * `retrieve_chunks_local(query)` — jurisdiction-specific SOPs (**use first**).
    * `retrieve_chunks_global(query)` — global best-practice docs (fallback).

    ### When the user supplies a bounding box (`bbox`)
    1. Call `assess_fire_danger` **exactly once**. Call `assess_fire_danger` **exactly once**.
    2. **Drone way-points requested?**
    * Return **only** a JSON array with **6-12** `[lat, lon]` pairs (no keys,
        comments, or prose).
    * Waypoints must be safe for drone to operate in.
    3. **SOPs/guidance requested?**
    * Query `retrieve_chunks_local` first.
    * Use `retrieve_chunks_global` only if relevant local guidance is missing.

    ### Additional rules
    * Distinguish between **action** (way-points) and **information** (SOPs, status)
    requests and answer accordingly.
    """
).strip()


def build_context_msg(ctx: IncidentContext) -> HumanMessage | None:
    """Return a fenced JSON block with bbox/POIs or *None* if both absent."""

    if ctx.is_empty:
        return None

    context_dict: dict[str, Any] = {}
    if ctx.bbox is not None and len(ctx.bbox) == 2:
        context_dict["bbox"] = ctx.bbox
    if ctx.pois is not None and len(ctx.pois) > 0:
        context_dict["pois"] = ctx.pois

    content = "CONTEXT\n```json\n" + json.dumps(context_dict, indent=2) + "\n```"
    return HumanMessage(content=content)


# -----------------------------------------------------------
# LangGraph helpers
# -----------------------------------------------------------


def load_tools() -> list:  # -> list[BaseTool]
    client = MultiServerMCPClient(MCP_SERVERS)
    return asyncio.run(client.get_tools())


async def build_react_agent(tools: list):
    llm = ChatOpenAI(
        model_name=OPENAI_MODEL,
        # temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_base=OPENAI_BASE,
        openai_api_key="unused",
        model_kwargs={"tool_choice": "any"},
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=AgentState,
        prompt=BASE_RULES
    )
    return agent, llm


async def compile_graph() -> StateGraph:
    tools = await asyncio.to_thread(load_tools)
    react_agent, _ = await build_react_agent(tools)

    checkpointer = InMemorySaver()

    graph = StateGraph(AgentState)
    graph.add_node("agent", react_agent)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")

    return graph.compile(checkpointer=checkpointer)


# -----------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------


def last_ai_message(messages: Iterable[BaseMessage]) -> AIMessage:
    for msg in reversed(list(messages)):
        if isinstance(msg, AIMessage):
            return msg
    raise RuntimeError("No AIMessage found in state")


async def run_turn(
    graph: StateGraph,
    user_prompt: str,
    ctx: IncidentContext,
    thread_id: str,
) -> tuple[str, IncidentContext]:
    """Execute one conversational turn and return (assistant_reply, ctx)."""

    messages: List[BaseMessage] = []

    # 1. System prompt
    messages.append(SystemMessage(content=BASE_RULES))

    # 2. Context message if we have geometry
    if not ctx.is_empty:
        context_msg = build_context_msg(ctx)
        if context_msg:
            messages.append(context_msg)

    # 3. User prompt
    messages.append(HumanMessage(content=user_prompt))

    initial_state = {"messages": messages, "ctx": ctx}

    out: AgentState = await graph.ainvoke(
        initial_state,
        config={
            "recursion_limit": 15,
            "configurable": {"thread_id": thread_id},
        },
    )
    '''print("STATE----------------------")
    print(out)
    print("STATE----------------------")'''
    reply = last_ai_message(out["messages"]).content
    return reply


# -----------------------------------------------------------
# Public CLI
# -----------------------------------------------------------


async def run_cli() -> None:  # pragma: no cover – interactive
    graph = await compile_graph()
    thread_id = f"fire-session-{uuid.uuid4()}"

    # Initial geometry
    current_ctx = IncidentContext(bbox=HARDCODED_BBOX, pois=HARDCODED_POIS)

    print("\nFireGPT - async CLI (type 'exit' to quit)\n")

    while True:
        try:
            user_prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting …")
            break

        if user_prompt.lower() in {"exit", "quit"}:
            break
        if not user_prompt:
            continue

        try:
            reply = await run_turn(
                graph,
                user_prompt,
                current_ctx,
                thread_id,
            )
        except Exception as exc:
            print(f"\nAgent error: {exc}\n", file=sys.stderr)
            continue

        print("\nAssistant:\n" + reply + "\n")


# -----------------------------------------------------------
# Launcher
# -----------------------------------------------------------


async def run_chat(graph, thread_id, user_prompt, bbox, pois):
    graph = graph
    thread_id = thread_id

    # Initial geometry
    current_ctx = IncidentContext(bbox=bbox, pois=pois)

    print("\nFireGPT - async CLI (type 'exit' to quit)\n")
    reply = await run_turn(
        graph,
        user_prompt,
        current_ctx,
        thread_id,
    )
    return reply


if __name__ == "__main__":
    asyncio.run(run_cli())
