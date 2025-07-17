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
print("MCP_SERVERS:", MCP_SERVERS)
OPENAI_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
print("OPENAI_BASE:", OPENAI_BASE)
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
        You are **FireGPT**, a professional wildfire-incident assistant.
        Accuracy, safety, and transparency are paramount.

        -------------------------------------------------------------------------------
        GENERAL BEHAVIOUR
        -------------------------------------------------------------------------------
        • *Conversation memory* - you CAN and SHOULD reference the full message list
        provided in the conversation state. Treat it as your short-term memory.
        • *Scope*
        - **Fire-related or SOP questions:** follow the domain-specific rules below.
        - **Other questions:** rely on your own background knowledge.

        -------------------------------------------------------------------------------
        TOOLS AVAILABLE
        -------------------------------------------------------------------------------
        1. `assess_fire_danger(bbox)`
            -> returns per-cell metrics like landcover percentage, landcover type,
            weather, wind speed and direction, fire risk level, and some other
            parameters. Use this to assess fire danger, before guiding the user
            or generating waypoints.

        2. `retrieve_chunks_local(query)`
            -> returns passages from jurisdiction-specific Standard Operating
            Procedures (SOPs). **Always query this first**.

        3. `retrieve_chunks_global(query)`
            -> returns passages from global best-practice documents. Use only when
            local retrieval returns nothing relevant.

        -------------------------------------------------------------------------------
        RULES FOR FIRE-RELATED TASKS
        -------------------------------------------------------------------------------
        A. **Fire Bounding box handling**
        • If the user supplies a **new** fire bounding box (`bbox`), call
            `assess_fire_danger(bbox)` **exactly once** for that bbox.
        • If the user does **not** supply a bbox in a later turn, assume the most
            recently provided bbox is still valid.
        • If the user provides a different bbox later, treat it as new and call the
            tool again (once).

        B. **SOP retrieval & advice**
        • When the user asks for SOPs, guidance, or strategy:
            1. Query **`retrieve_chunks_local`** with a concise description of the
                scenario (fuel type, wind, terrain, etc.).
            2. If no pertinent local passages are returned, fall back to
                `retrieve_chunks_global`.
        • **Summarise** the retrieved guidance in your own words. If you cite
            authority, reference it briefly, e.g.
            > *“Per Local SOP §4.3 (mixed-forest containment) …”*
            Do **not** dump raw passages.

        C. **Drone waypoint generation**
        • When the user requests way-points to perform a specific task with drones, follow
            the following instructions to generate waypoints to perform the task effectively:
            * If you have a bounding box:
                - Call `assess_fire_danger()` with bbox to get the context of fire region.
            * Retrieve relevant SOPs and operational guides using `retrieve_chunks_local()` and/or
              `retrieve_chunks_global()`.
            * Reason about choosing waypoints based on the fire danger assessment (if fire bbox is given),
              and retrieved SOPs.
            * Provide a JSON **array** of **4-8** `[lat, lon]` pairs. Do not put **any comments** in the json.
            * If the user asks for the explanation of the waypoints, do not put the explanation
              in the comments of the json, but rather provide them in the text as plain sentences.

        -------------------------------------------------------------------------------
        FAIL-SAFE
        -------------------------------------------------------------------------------
        If you are not confident in your answer, say so clearly rather than inventing
        facts or ignoring the rules above.
    """
).strip()


def build_context_msg(fire_bbox, pois) -> str | None:
    """Return a context message with bbox/POIs or *None* if both absent."""
    context_dict: dict[str, Any] = {}
    if fire_bbox is not None:
        context_dict["fire_bbox"] = fire_bbox
    if pois is not None and len(pois) > 0:
        context_dict["points_of_interest"] = pois

    content = "FIRE REGION CONTEXT:\n```json\n" + json.dumps(context_dict, indent=2) + "\n```"
    return content if context_dict else None


# -----------------------------------------------------------
# LangGraph helpers
# -----------------------------------------------------------


def load_tools() -> list:
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
    thread_id: str,
    user_prompt: str,
    context_msg: str | None = None
) -> tuple[str, IncidentContext]:
    """Execute one conversational turn and return (assistant_reply, ctx)."""

    messages: List[BaseMessage] = []

    # 1. System prompt
    messages.append(SystemMessage(content=BASE_RULES))

    # 2. Context message if we have geometry
    if context_msg is not None:
        messages.append(HumanMessage(content=context_msg))

    # 3. User prompt
    messages.append(HumanMessage(content=user_prompt))

    initial_state = {"messages": messages}

    out: AgentState = await graph.ainvoke(
        initial_state,
        config={
            "recursion_limit": 15,
            "configurable": {"thread_id": thread_id},
        },
    )
    print("STATE----------------------")
    print(out)
    print("STATE END----------------------")
    reply = last_ai_message(out["messages"]).content
    return reply


# -----------------------------------------------------------
# Public CLI
# -----------------------------------------------------------


async def run_cli() -> None:  # pragma: no cover - interactive
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


async def run_chat(graph, thread_id, user_prompt, fire_bbox, pois):
    graph = graph
    thread_id = thread_id

    # Initial geometry
    context_msg = build_context_msg(fire_bbox, pois)

    print("\nFireGPT - async CLI (type 'exit' to quit)\n")
    reply = await run_turn(
        graph,
        thread_id,
        user_prompt,
        context_msg,
    )
    return reply


if __name__ == "__main__":
    asyncio.run(run_cli())
