#!/usr/bin/env python
"""
cli_fire_agent_async.py
Asynchronous ReAct agent for FireGPT – CLI version
with short-term memory + summarisation + self-critic.
"""

from __future__ import annotations

import os
import uuid
import asyncio
from typing import List, Any, Dict
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_openai.chat_models import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError

from langmem.short_term import SummarizationNode


# ───────────────────────────────────────────────────────────
# 1. Build the ReAct agent with memory + summarisation
# ───────────────────────────────────────────────────────────
async def build_agent() -> tuple["Runnable", InMemorySaver, ChatOpenAI]:
    # 1a – shared LLM
    llm = ChatOpenAI(
        model_name="qwen3:8b",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_base=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        openai_api_key="unused",
        model_kwargs={"tool_choice": "any"},
    )

    # 1b – MCP tools
    mcp = MultiServerMCPClient(
        {"fire_mcp": {"url": "http://localhost:7790/mcp/", "transport": "streamable_http"}}
    )
    tools: List = await mcp.get_tools()
    tool_names = ", ".join(t.name for t in tools)

    # 1c – ReAct prompt
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are **FireGPT**, a drone-enabled wildfire-fighting assistant.

                Use the *ReAct* format **exactly**:

                Thought: <your private reasoning>
                Action: <one of {tools}[<tool-input>]>
                Observation: <tool result>

                … (repeat Thought→Action→Observation) …

                Final Answer: <concise answer for the firefighter – no private thoughts>.

                *Never reveal the raw “Thought:” text to the human.*
                If you cannot answer, say “I don’t know based on the information available.”"""
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    prompt = prompt_template.partial(tools=tool_names)

    # 1d – summarisation node keeps context small
    summarisation = SummarizationNode(
        model=llm.bind(max_tokens=128),
        max_tokens=384,
        max_summary_tokens=128,
        output_messages_key="llm_input_messages",
    )

    # 1e – optional custom state schema
    class FireState(AgentState):
        context: dict[str, Any]

    # 1f – memory persistence
    checkpointer = InMemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        # pre_model_hook=summarisation,
        state_schema=FireState,
        checkpointer=checkpointer,
    )
    return agent, checkpointer, llm


# ────────────────────────────────────────────────────────────────────
# Helper: wrap the inner ReAct agent so it consumes & produces a
# root-level "messages" list
# ────────────────────────────────────────────────────────────────────
def make_agent_node(inner_agent):
    async def _call(state: Dict[str, Any], config: Dict[str, Any] | None = None):
        # pass the current transcript to the inner agent
        result = await inner_agent.ainvoke({"messages": state["messages"]}, config=config)
        # overwrite the transcript in *outer* state
        return {"messages": result["messages"]}
    return _call


# ───────────────────────────────────────────────────────────
# 2. Wrap with self-critic loop
# ───────────────────────────────────────────────────────────
def build_agent_with_critic(base_agent: "Runnable", llm, checkpointer) -> "Runnable":
    # critic chain (same as before)
    critic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a wildfire-safety QA reviewer.\n"
             "Reply 'OK' if the answer is correct & complete; "
             "otherwise reply 'NEEDS' and one short reason."),
            ("human", "{answer}"),
        ]
    )
    critic_chain = (
        critic_prompt
        | llm
        | StrOutputParser()
        | (lambda v: {"verdict": v})          # put verdict into state
    )

    # node that runs the inner agent and FLATTENS the result
    agent_node = make_agent_node(base_agent)

    # extract the latest assistant message to feed the critic
    def get_final_answer(state):
        last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        return {"answer": last_ai.content}

    # decide whether to retry
    def needs_retry(state):
        return "retry" if state["verdict"].startswith("NEEDS") else "done"

    # build graph
    g = StateGraph(AgentState)         # state schema: just a message list
    g.add_node("agent", agent_node)
    g.add_node("get_answer", get_final_answer)
    g.add_node("critic", critic_chain)

    g.set_entry_point("agent")
    g.add_edge("agent", "get_answer")
    g.add_edge("get_answer", "critic")
    g.add_conditional_edges("critic", needs_retry,
                            {"retry": "agent", "done": END})

    return g.compile(checkpointer=checkpointer)


# ───────────────────────────────────────────────────────────
# 3. Async CLI loop
# ───────────────────────────────────────────────────────────
async def chat():
    base_agent, cp, llm = await build_agent()
    agent = build_agent_with_critic(base_agent, llm, cp)

    thread_id = f"fire-session-{uuid.uuid4()}"
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 13,        # hard stop
    }

    # optional: draw graph to PNG
    with open("agent_graph_react.png", "wb") as f:
        f.write(agent.get_graph().draw_mermaid_png())

    print("\nFireGPT (Async) — type 'exit' to quit.\n")

    # Initialise state for FIRST turn
    state = {"messages": []}
    while True:
        user_text = input("👨‍🚒  You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        state["messages"].append({"role": "user", "content": user_text})

        state = await agent.ainvoke(state,   # <─ pass whole state
                                    config={"configurable": {"thread_id": "demo"}})

        # print only the assistant's visible answer
        last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        final = last_ai.content.split("Final Answer:")[-1].strip()
        print(f"🤖  {final}\n")


# ───────────────────────────────────────────────────────────
# 4. Entry-point
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(chat())
