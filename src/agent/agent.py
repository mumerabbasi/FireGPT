import os
import asyncio
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_mcp_adapters.client import MultiServerMCPClient


# ----------------------------------------------------------------------------
# In-memory history implementation for RunnableWithMessageHistory
# ----------------------------------------------------------------------------
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: list = Field(default_factory=list)

    def add_messages(self, messages: list) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# Global store for session histories
_history_store: dict[str, InMemoryHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _history_store:
        _history_store[session_id] = InMemoryHistory()
    return _history_store[session_id]


# ----------------------------------------------------------------------------
# Simple Streaming Chat App with MCP tool calling
# ----------------------------------------------------------------------------
def main():
    # 1) Configure OpenAI-compatible endpoint for Ollama
    os.environ["OPENAI_API_BASE"] = os.getenv(
        "OPENAI_API_BASE", "http://localhost:11434/v1"
    )
    os.environ["OPENAI_API_KEY"] = os.getenv(
        "OPENAI_API_KEY", "unused"
    )

    # 2) Connect to MCP server and retrieve tool definitions
    mcp_client = MultiServerMCPClient({
        "mcp": {"url": "http://localhost:7790/mcp/", "transport": "streamable_http"}
    })
    tools = asyncio.run(mcp_client.get_tools())

    # 3) Define chat prompt with history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use external tools."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{user_input}"),
    ])

    # 4) Initialize streaming ChatOpenAI LLM with tool definitions
    llm = ChatOpenAI(
        model_name="qwen3:8b",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
    ).bind_tools(
        tools=tools,
        tool_choice="auto"
    )

    # 5) Wrap prompt | llm in RunnableWithMessageHistory
    chain = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="user_input",
        history_messages_key="history",
    )

    session_id = "default_session"
    print("Streaming Qwen3-8B via Ollama with MCP tool calling (type 'quit' to exit)\n")

    # 6) Simple REPL loop
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        print("Assistant: ", end="", flush=True)
        # Stream response fragments and print as they arrive
        for fragment in chain.stream(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}},
        ):
            continue
            # text = getattr(fragment, "content", str(fragment))
            # print(text, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
