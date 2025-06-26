import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field


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
# Simple Streaming Chat App using the RunnableSequence pattern
# ----------------------------------------------------------------------------
def main():
    # 1) Configure OpenAI-compatible endpoint for Ollama
    os.environ["OPENAI_API_BASE"] = os.getenv(
        "OPENAI_API_BASE", "http://localhost:11434/v1"
    )
    os.environ["OPENAI_API_KEY"] = os.getenv(
        "OPENAI_API_KEY", "unused"
    )

    # 2) Define prompt with placeholders
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{user_input}"),
    ])

    # 3) Initialize streaming ChatOpenAI LLM
    llm = ChatOpenAI(
        model_name="qwen3:8b",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
    ).bind_tools()

    # 4) Build the RunnableSequence: prompt | llm
    base_chain = prompt | llm

    # 5) Wrap with RunnableWithMessageHistory
    chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history=get_session_history,
        input_messages_key="user_input",
        history_messages_key="history",
    )

    session_id = "default_session"
    print("Streaming Qwen3-8B via Ollama with RunnableWithMessageHistory (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        # 6) Invoke with streaming, passing session_id in config
        print("Assistant: ", end="", flush=True)
        full_reply = ""
        for fragment in chain.stream(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}},
        ):
            # fragment may be a message chunk or AIMessage
            text = getattr(fragment, "content", str(fragment))
            full_reply += text
        print()


if __name__ == "__main__":
    main()
