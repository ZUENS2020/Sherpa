
from __future__ import annotations
SYSTEM_PROMPT = """You are a helpful AI assistant that calls tools to get information when needed.
When you call a tool, you must use the exact function signature provided.
You must always call a tool when you have enough information to do so.
After calling a tool, wait for the tool's response before answering the user.
Use the tool responses to help you answer the user's questions.
When you respond, you must respond in the specified response format.
"""

from dataclasses import dataclass
import json
import os

from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

checkpointer = InMemorySaver()

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

def _default_openrouter_base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def _default_openrouter_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

@dataclass
class ResponseFormat:
    """标准化的智能体响应格式"""
    response: str                   # 智能体最终的回答
    used_tools: list[str] | None = None  # 使用过的工具名称列表
    response_key: str | None = None  # 指定主要回答的字段名称
    def to_json(self) -> str:
        """转成 JSON 字符串，供 LangChain 输出使用"""
        return json.dumps({
            "response": self.response,
            "used_tools": self.used_tools or []
        }, ensure_ascii=False)

def create_agent_outside(
    input_text: str,
    model: str | None = None,
    mdoel: str | None = None,
    temperature: float = 0.5,
    timeout: int = 10,
    max_tokens: int = 1000,
    *,
    openrouter_api_key: str | None = None,
    openrouter_base_url: str | None = None,
):

    # Prefer OpenRouter in an OpenAI-compatible mode.
    # Required env var: OPENROUTER_API_KEY (or fallback OPENAI_API_KEY).
    openrouter_key = (
        openrouter_api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not openrouter_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY (recommended) or OPENAI_API_KEY before using /chat_with_agent."
        )

    model_name = (model or mdoel or "").strip() or _default_openrouter_model()
    base_url = (openrouter_base_url or "").strip() or _default_openrouter_base_url()
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        openai_api_key=openrouter_key,
        openai_api_base=base_url,
    )
    
    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_user_location, get_weather_for_location],
        context_schema=Context,
        response_format=ResponseFormat,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": f"{input_text}"}]},
        config=config,
        context=Context(user_id="1"),
    )
    return response




if __name__ == "__main__":
    try:
        response = create_agent_outside("what is the weather outside?")
    except Exception as e:
        print(f"demo failed: {e}")
    else:
        print(response)
