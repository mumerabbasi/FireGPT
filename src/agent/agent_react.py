"""fire_agent_async.py
====================

Async CLI wrapper around a ReAct-style LangGraph agent for **wild-fire incident
response**.

Key features (v3)
-----------------
1. **Structured spatial context** - operator may supply a fire bounding-box and a
   list of POIs; both are embedded in the **system prompt**.  The agent *may* call
   `assess_fire_danger` when a bbox exists, but it is not forced to.
2. **Three MCP tools**
   * `retrieve_chunks_local`   - jurisdiction-specific SOPs (always override global)
   * `retrieve_chunks_global`  - wider best-practice repository
   * `assess_fire_danger`      - per-cell danger profile for the bbox (wind, slope …)
3. **Lightweight QA editor** - validates the assistant’s reply.  When no fixes are
   required, it signals *no-edit* so the original response is returned unchanged;
   otherwise it replaces the answer with a corrected version.
4. **Waypoint output contract** - if the user asks for a drone route, the final
   answer **must** be a plain JSON list of 6-12 `[lat, lon]` pairs that avoid high-
   danger cells (score ≤ 30) unless within 150 m of a POI.

This module strives for **PEP 8** compliance and clear separation of concerns.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from langchain_openai.chat_models import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = [
    "run_cli",
]

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

# ---------------------------------------------------------------------------
# Default demo context (used when the operator leaves inputs empty)
# ---------------------------------------------------------------------------
DEFAULT_POIS = [
    {"lat": 47.70, "lon": 8.00, "note": "Critical power sub-station"},
    {"lat": 47.73, "lon": 8.05, "note": "Regional hospital - must protect"},
]
DEFAULT_BBOX = ((47.6969, 7.9468), (47.7524, 8.0347))

# -----------------------------------------------------------
# Dataclasses - strongly-typed context passed through the LangGraph state
# -----------------------------------------------------------


@dataclass
class IncidentContext:
    """Spatial context for the current user turn."""

    bbox: Tuple[Tuple[float, float], Tuple[float, float]] | None = None
    pois: List[dict[str, Any]] | None = None

    @property
    def has_bbox(self) -> bool:  # noqa: D401 (property, read like attribute)
        """Whether a bounding-box is defined."""
        return self.bbox is not None


# -----------------------------------------------------------
# System-prompt builder
# -----------------------------------------------------------


def _format_system_msg(ctx: IncidentContext) -> SystemMessage:
    """Return a concise system message embedding spatial context.

    The wording *encourages* (but does not force) use of the danger-assessment
    tool when a bbox is present.
    """

    lines: list[str] = ["INCIDENT CONTEXT"]

    # -- Fire bounding-box ----------------------------------------------------
    if ctx.bbox is not None:
        (lat_tl, lon_tl), (lat_br, lon_br) = ctx.bbox
        lines.append(
            f"Active-fire bbox: [ {lat_tl:.5f}, {lon_tl:.5f} ]  (top left) -> "
            f"[ {lat_br:.5f}, {lon_br:.5f} ] (bottom right)"
        )

    # -- Points of interest ---------------------------------------------------
    if ctx.pois:
        poi_block = "\n".join(
            f"• ( {p['lat']:.5f}, {p['lon']:.5f} ) — {p['note']}" for p in ctx.pois
        )
        lines.extend(["", "Points-of-interest:", poi_block])

    # -- Tool head-up ---------------------------------------------------------
    lines.extend(
        [
            "",
            """If active-fire box is present, always call the assess_fire_danger tool exactly once
            with parameter `bbox=[[lat_tl, lon_tl], [lat_br, lon_br]]` using the
            active-fire box coords *before* producing your final answer.

            `assess_fire_danger` gives you several parameters and statistics of the region.
            Based on these statistics, you should consult user in doing the right actions and 
            if the user asks, give them waypoints in form of coordinates for the user's drone.

            When the user requests way-points, return them as a plain JSON list"
            of [lat, lon] pairs that AVOID high-danger zones. Do NOT invent"
            coordinates — only derive them logically from `assess_fire_danger`
            output plus POIs and retrieved documents.

            FINAL-ANSWER FORMAT (MANDATORY)
            Return **only** a JSON list of [lat, lon] pairs, e.g.:
            '''json
            [[47.7010, 7.9586],
            [47.7050, 7.9602],
            [47.7100, 7.9621]]
            '''
            Inside the json block, never put any comments. Only parsable json.
            6-12 points max.
            Each point must lie either
            inside a sub-grid whose fire_danger.score ≤ 30, or
            within 150 m of a POI.
            Do NOT add any prose or keys — just the JSON array.

            If the user asks for standard operating procedures (SOPs) or guidance during fire-fighting,
            you should retrieve relevant documents using the `retrieve_chunks_local` tool first.
            Local SOPs always take priority. Only consult global SOPs (via `retrieve_chunks_global`)
            if local documents do not provide sufficient guidance or are absent.

            If the user does **not** ask for waypoints, do **not** return coordinates. In such cases,
            respond normally in helpful prose based on retrieved documents and relevant observations.

            Always distinguish clearly between requests for action (e.g. waypoints) and requests for
            information (e.g. SOPs or protocols)."""
        ]
    )

    return SystemMessage(content="\n".join(lines))


# -----------------------------------------------------------
# LangGraph construction helpers
# -----------------------------------------------------------


async def _load_tools() -> list:  # -> list[BaseTool]
    """Fetch the tool manifests from the MCP server(s)."""

    client = MultiServerMCPClient(MCP_SERVERS)
    return await client.get_tools()


async def _build_react_agent(tools: list) -> Tuple[Runnable, ChatOpenAI]:
    """Return (react_agent, llm)."""

    llm = ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=0,
        streaming=True,
        openai_api_base=OPENAI_BASE,
        openai_api_key="unused",
        model_kwargs={"tool_choice": "any"},
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=AgentState,
    )
    return agent, llm


# -----------------------------------------------------------
# QA editor (critic)
# -----------------------------------------------------------

CRITIC_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a wildfire-response QA editor. You need to see if the response
            generated by another AI is correct, faithful, and helpful as per the user
            prompt and the previous history. You will have access to the user messge,
            AI system prompt, AI reponse and the history using which the AI generated the response.
            History will contain LLM thoughts inside <think> <\think> tags, its tool
            calls and its reponses. Taking all this into account you will have to
            decide if you want to improve the response or not. If you want to improve
            response then give the improved response. Otherwise just reproduce the same response.
            """,
        ),
        (
            "human",
            """\n\nAI System Prompt:
            \n{agent_system_prompt}
            \n\nUser prompt:
            \n{user_prompt}
            Full turn transcript (messages + tool calls):\n\n{history}""",
        ),
    ]
)


def _build_critic(llm: ChatOpenAI) -> Runnable:
    """Return a runnable that either passes or rewrites the answer."""

    prompt_chain = CRITIC_TEMPLATE | llm | StrOutputParser()

    def _editor_node(state: AgentState, config: RunnableConfig):
        # Build a plain-text transcript for the critic prompt
        def _msg_to_line(m: BaseMessage) -> str:
            role = (
                "Human" if isinstance(m, HumanMessage)
                else "Assistant" if isinstance(m, AIMessage)
                else "System"
            )
            return f"{role}: {m.content}"

        transcript = "\n".join(_msg_to_line(m) for m in state["messages"])
        edited = prompt_chain.invoke({"history": transcript}, config=config).strip()

        if edited.lower() == "no_edit":
            # No changes - terminate without injecting a new message
            return {}

        return {"messages": AIMessage(content=edited)}

    return _editor_node


# -----------------------------------------------------------
# Build full LangGraph
# -----------------------------------------------------------


async def build_agent() -> Runnable:
    """Assemble the full (agent  -> critic) graph and return it."""

    tools = await _load_tools()
    react_agent, llm = await _build_react_agent(tools)

    checkpointer = InMemorySaver()
    # critic_node = _build_critic(llm)

    graph = StateGraph(AgentState)
    graph.add_node("agent", react_agent)
    # graph.add_node("critic", critic_node)

    graph.set_entry_point("agent")
    # graph.add_edge("agent", "critic")
    graph.set_finish_point("agent")

    return graph.compile(checkpointer=checkpointer)


# -----------------------------------------------------------
# CLI driver helpers
# -----------------------------------------------------------


def _load_json_or_default(label: str, default: Any) -> Any:
    """Prompt the operator for JSON; fall back to *default* on blank input."""

    raw = input(label).strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"\nInvalid JSON: {exc}\n", file=sys.stderr)
        return _load_json_or_default(label, default)


def _iter_until_ai(messages: Iterable[BaseMessage]):
    """Return the last AI message from an iterable."""

    for msg in reversed(list(messages)):
        if isinstance(msg, AIMessage):
            return msg
    raise RuntimeError("No AIMessage found in state")


async def _run_turn(
    graph: Runnable,
    user_prompt: str,
    ctx: IncidentContext,
    thread_id: str,
) -> str:
    """Run a single turn and return the assistant's final text."""

    system_msg = _format_system_msg(ctx)
    init_messages: list[BaseMessage] = [system_msg, HumanMessage(content=user_prompt)]

    initial_state = {
        "messages": init_messages,
        "ctx": ctx,
    }

    out: AgentState = await graph.ainvoke(
        initial_state,
        config={
            "recursion_limit": 15,
            "configurable": {"thread_id": thread_id},
        },
    )
    print("RAWWWWWW")
    print(out)
    print("RAWWWWWW")
    return _iter_until_ai(out["messages"]).content


# -----------------------------------------------------------
# Public entry
# -----------------------------------------------------------


async def run_cli() -> None:  # pragma: no cover - interactive
    """Interactive async CLI."""

    graph = await build_agent()
    thread_id: str = f"fire-session-{uuid.uuid4()}"

    print("\nFireGPT - async CLI (type 'exit' to quit)\n")

    while True:
        try:
            user_prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break

        if user_prompt.lower() in {"exit", "quit"}:
            break
        if not user_prompt:
            continue

        # ---- spatial context ------------------------------------------------
        pois = _load_json_or_default("POIs JSON    (blank = demo) > ", DEFAULT_POIS)
        bbox_list = _load_json_or_default("BBox JSON    (blank = demo) > ", DEFAULT_BBOX)

        # ensure bbox is either None or proper tuple of tuples
        bbox_typed: Tuple[Tuple[float, float], Tuple[float, float]] | None = None
        if bbox_list is not None:
            try:
                bbox_typed = ((bbox_list[0][0], bbox_list[0][1]), (bbox_list[1][0], bbox_list[1][1]))
            except (TypeError, IndexError):
                print("BBox must be [[lat_tl, lon_tl], [lat_br, lon_br]] - ignoring.")

        ctx = IncidentContext(bbox=bbox_typed, pois=pois)

        try:
            reply = await _run_turn(graph, user_prompt, ctx, thread_id)
        except Exception as exc:  # pragma: no cover - interactive diagnostics
            print(f"\nAgent error: {exc}\n", file=sys.stderr)
            continue

        print("\nAssistant:\n" + reply + "\n")


# -----------------------------------------------------------
# Script launcher
# -----------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover - interactive
    asyncio.run(run_cli())
