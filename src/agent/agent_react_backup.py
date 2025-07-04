#!/usr/bin/env python
"""
cli_fire_agent_async.py
Asynchronous ReAct agent for FireGPT - CLI version
with short-term memory + summarisation + self-critic.
"""

from __future__ import annotations
import os
import uuid
import asyncio
from typing import Any

from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_openai.chat_models import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import StateGraph
from langgraph.errors import GraphRecursionError


# -----------------------------------------------------------
# 0. Custom state schema
# -----------------------------------------------------------
class FireState(AgentState):
    """Extend the base AgentState with any extra fields you need."""
    context: dict[str, Any]
    answer: str
    verdict: str
    feedback: str


# -----------------------------------------------------------
# 1. Build the ReAct agent with memory + summarisation
# -----------------------------------------------------------
async def build_agent() -> tuple[Runnable, InMemorySaver, ChatOpenAI]:
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
            "transport": "streamable_http"
        }
    })
    tools = await mcp.get_tools()
    checkpointer = InMemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=FireState,
        checkpointer=checkpointer,
    )
    return agent, checkpointer, llm


# -----------------------------------------------------------
# 2. Wrap with self-critic loop
# -----------------------------------------------------------
def build_agent_with_editor(
    base_agent: Runnable,
    llm: ChatOpenAI,
    checkpointer: InMemorySaver,
) -> Runnable:

    critic_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a wildfire-safety QA editor."
            "Think deeply before you output anything."
            "You will be given a transcript where an LLM Agent has called some tools and articulated a response."
            "Based on that, you need to assess the response for truthfulness, faithfulness and helpfulness."
            "Based on your assessment edit the response to improve it in the aformentioned aspects."
            "Also state what did you improve in the original text. If you didnt improve anything then also tell."
            "Then give your version of the response."
        ),
        ("human",
         "Here is the entire transcript of the conversation, including tool calls and observations:\n\n"
         "{history}\n\n"),
    ])
    critic_chain = critic_prompt | llm | StrOutputParser()

    def edit_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\n\n\n\nORIGNINAL Agent Response: {msg}\n")
                break
        # Build the history string
        history_lines = []
        for msg in state["messages"]:
            speaker = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            history_lines.append(f"{speaker}: {msg.content}")
        history_text = "\n".join(history_lines)

        # Call the critic to get the edited response
        edited_content = critic_chain.invoke({"history": history_text}, config=config)
        print("EDITED CONTENT")
        print(edited_content)
        print("EDITED CONTENT END")
        return {
            "messages": AIMessage(content=edited_content)
        }

    g = StateGraph(FireState)
    g.add_node("agent", base_agent)
    g.add_node("editor", edit_node)

    g.set_entry_point("agent")
    g.add_edge("agent", "editor")
    g.set_finish_point("editor")

    return g.compile(checkpointer=checkpointer)


# -----------------------------------------------------------
# 3. Async CLI loop
# -----------------------------------------------------------
async def chat():
    base_agent, cp, llm = await build_agent()
    agent = build_agent_with_editor(base_agent, llm, cp)

    with open("agent_graph_react.png", "wb") as f:
        f.write(agent.get_graph().draw_mermaid_png())

    thread_id = f"fire-session-{uuid.uuid4()}"
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 13,
    }

    print("\nFireGPT (Async) — type 'exit' to quit.\n")
    while True:
        user = input("You : ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        try:
            state = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user}]},
                config=cfg
            )
        except GraphRecursionError:
            print("\nReached step limit without final answer.\n")
            continue
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\n\n\n\nAgent Response: {msg}\n")
                break

# -----------------------------------------------------------
# 4. Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(chat())
