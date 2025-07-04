#!/usr/bin/env python
"""
cli_fire_agent_async.py
Asynchronous ReAct agent for FireGPT – CLI version
with short-term memory + summarisation + self-critic.
"""

from __future__ import annotations
import os
import asyncio

from langchain_core.runnables.base import Runnable

from langchain_openai.chat_models import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


# -----------------------------------------------------------
# 1. Build the ReAct agent with memory + summarisation
# -----------------------------------------------------------
async def build_agent() -> tuple[Runnable, InMemorySaver, ChatOpenAI]:
    llm = ChatOpenAI(
        model_name="qwen3:8b",
        temperature=0,
        streaming=True,
        # callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_base=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        openai_api_key="unused",
        model_kwargs={"tool_choice": "any"},
    )

    client = MultiServerMCPClient({
        "fire_mcp": {
            "url": "http://localhost:7790/mcp/",
            "transport": "streamable_http"
        }
    })
    tools = await client.get_tools()

    agent = create_react_agent(llm, tools)
    return agent


# -----------------------------------------------------------
# 3. Async CLI loop
# -----------------------------------------------------------
async def chat():
    agent = await build_agent()

    print("\nFireGPT (Async) — type 'exit' to quit.\n")
    while True:
        user = input("You : ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        state = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user}]},
        )

        print(f"\nAgent Response: {state}")

# -----------------------------------------------------------
# 4. Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(chat())
