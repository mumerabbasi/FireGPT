"""
cli_fire_agent_async.py -v2
===========================

Asynchronous CLI wrapper around a ReAct-style LangGraph agent that is
specifically tuned for **wild-fire incident response**. New features:

1. **Structured context intake** -the operator now supplies:
   • Natural-language *prompt*
   • A **POI list** (5-10 objects with lat, lon, note)
   • A **fire bounding box** (top-left & bottom-right lat/lon pairs)
   Those three elements are normalised into a *system message* so the LLM
   can reliably see them and is reminded to call `assess_fire_danger`.
2. **Mandatory tool usage guard** -the system prompt tells the agent it
   *must* call `assess_fire_danger(bbox=…)` before it answers.
3. **Critic/editor upgrade** -the QA pass now double-checks that the
   danger assessor *was* invoked and that no hallucinated coordinates are
   present.
4. **Waypoint output format** -if the user asks for a drone path the
   model is instructed to return `[[lat, lon], …]` pairs.

The rest of the file keeps the same high-level structure: build base
agent → wrap with critic → async CLI loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, List, Tuple, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig

from langchain_openai.chat_models import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
POIS = [
    {"lat": 47.70, "lon": 8.00, "note": "Critical power sub-station"},
    {"lat": 47.73, "lon": 8.05, "note": "Regional hospital - must protect"},
]

D_BBOX = [[47.6969, 7.9468], [47.7524, 8.0347]]


# ---------------------------------------------------------------------------
# Helper: format the operator-supplied spatial context as a compact string
# ---------------------------------------------------------------------------


def _context_system_message(pois: List[Dict[str, Any]],
                            bbox: Tuple[Tuple[float, float], Tuple[float, float]]) -> SystemMessage:
    """Return a SystemMessage embedding POIs & bounding box.

    Parameters
    ----------
    pois : list of dicts with keys lat, lon, note
    bbox : ((lat_tl, lon_tl), (lat_br, lon_br)) - top-left & bottom-right
    """
    poi_lines = [f"• ({p['lat']:.6f}, {p['lon']:.6f}) — {p['note']}" for p in pois]
    poi_block = "\n".join(poi_lines) if poi_lines else "(none)"

    context = (
        "INCIDENT CONTEXT\n"
        "Bounding-box of active fire: "
        f"[{bbox[0][0]:.6f}, {bbox[0][1]:.6f}] (top-left) → "
        f"[{bbox[1][0]:.6f}, {bbox[1][1]:.6f}] (bottom-right)\n\n"
        "Points-of-interest (POIs):\n"
        f"{poi_block}\n\n"
        "MANDATORY: Always call the `assess_fire_danger` tool exactly once "
        "with parameter `bbox=[[lat_tl, lon_tl], [lat_br, lon_br]]` using the "
        "coords given above *before* producing your final answer.\n"
        "`assess_fire_danger` gives you several parameters and statistics of the region."
        "Based on these statistics, you should give waypoints in form of coordinates for"
        "the user's drone."
        "When the user requests way-points, return them as a plain JSON list "
        "of [lat, lon] pairs that AVOID high-danger zones. Do NOT invent "
        "coordinates - only derive them logically from `assess_fire_danger` "
        "output plus POIs and retrieved documents."
        "FINAL-ANSWER FORMAT (MANDATORY)"
        "Return **only** a JSON list of [lat, lon] pairs, e.g.:"

        "[[47.7010, 7.9586],"
        "[47.7050, 7.9602],"
        "[47.7100, 7.9621]]"
        "• 6-12 points max."
        "• Each point must lie either"
        "   - inside a sub-grid whose fire_danger.score ≤ 30, or"
        "   - within 150 m of a POI."
        "• Do NOT add any prose or keys — just the JSON array."
    )
    return SystemMessage(content=context)


# ---------------------------------------------------------------------------
# 1. Build the base ReAct agent (unchanged except for streaming defaults)
# ---------------------------------------------------------------------------


async def build_agent() -> Tuple[Runnable, InMemorySaver, ChatOpenAI]:
    llm = ChatOpenAI(
        model_name="qwen3:8b",
        temperature=0,
        streaming=True,
        openai_api_base=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        openai_api_key="unused",
        model_kwargs={"tool_choice": "any"},
    )

    mcp = MultiServerMCPClient({
        "fire_mcp": {
            "url": "http://localhost:7790/mcp/",
            "transport": "streamable_http",
        }
    })
    tools = await mcp.get_tools()

    checkpointer = InMemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=AgentState,
        checkpointer=checkpointer,
    )
    return agent, checkpointer, llm


# ---------------------------------------------------------------------------
# 2. Wrap with critic / self-editor that enforces safety & correctness
# ---------------------------------------------------------------------------


def build_agent_with_editor(base_agent: Runnable,
                            llm: ChatOpenAI,
                            checkpointer: InMemorySaver) -> Runnable:

    critic_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a wildfire-response QA editor. Ensure the assistant's "
            "final answer is *truthful*, *faithful* to retrieved evidence, "
            "and *operationally safe*.\n"
            "MANDATORY checks:\n"
            "1. The `assess_fire_danger` tool must have been called *once* and with the correct bbox.\n"
            "2. No invented coordinates or facilities.\n"
            "3. Way-points (if present) are a valid list of [lat, lon] pairs and avoid HIGH danger zones reported by \
                the tool.\n"
            "4. Language is clear for first responders.\n\n"
            "If any issue is found, rewrite the answer to fix it."
        ),
        (
            "human",
            "Full turn transcript (messages & tool calls):\n\n{history}\n"
        ),
    ])

    critic_chain = critic_prompt | llm | StrOutputParser()

    def edit_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        # Build transcript string
        lines: List[str] = []
        for m in state["messages"]:
            speaker = (
                "Human" if isinstance(m, HumanMessage)
                else "Assistant" if isinstance(m, AIMessage)
                else "System"
            )
            lines.append(f"{speaker}: {m.content}")
        transcript = "\n".join(lines)

        edited_response = critic_chain.invoke({"history": transcript}, config=config)

        return {"messages": AIMessage(content=edited_response)}

    g = StateGraph(AgentState)
    g.add_node("agent", base_agent)
    g.add_node("editor", edit_node)

    g.set_entry_point("agent")
    # g.set_finish_point("agent")
    g.add_edge("agent", "editor")
    g.set_finish_point("editor")

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 3. Async CLI loop with structured context intake
# ---------------------------------------------------------------------------


async def chat() -> None:
    base_agent, cp, llm = await build_agent()
    agent = build_agent_with_editor(base_agent, llm, cp)

    # Visualise the graph once for debugging
    try:
        with open("agent_graph_react.png", "wb") as f:
            f.write(agent.get_graph().draw_mermaid_png())
    except Exception:
        pass

    thread_id = f"fire-session-{uuid.uuid4()}"
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 17,
    }

    print("\nFireGPT (Async) — type 'exit' to quit.\n")

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

        # ---- Gather spatial context ----
        def _inp(label: str) -> str:
            return input(label).strip()

        try:
            # To convert them to a string,
            # coz we will recieve a string from front end
            pois_raw = json.dumps(POIS)
            bbox_raw = json.dumps(D_BBOX)
            pois: List[Dict[str, Any]] = json.loads(pois_raw)
            bbox_list = json.loads(bbox_raw)
            bbox: Tuple[Tuple[float, float], Tuple[float, float]] = (
                (bbox_list[0][0], bbox_list[0][1]),
                (bbox_list[1][0], bbox_list[1][1]),
            )
        except (json.JSONDecodeError, IndexError, TypeError, ValueError):
            print("\nInvalid JSON. Please retry.\n")
            continue

        system_ctx = _context_system_message(pois, bbox)
        messages = [system_ctx, HumanMessage(content=user_prompt)]

        try:
            state = await agent.ainvoke({"messages": messages}, config=cfg)
        except GraphRecursionError:
            print("\nReached step limit without a final answer. Try refining the query.\n")
            continue

        # Print edited assistant reply
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print("\nAssistant:\n" + msg.content + "\n")
                break

        # print(state)

# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(chat())
