import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.extensions.memory import AsyncSQLiteSession
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from dotenv import load_dotenv
from typing import Any, AsyncGenerator, Callable, List, Optional


class AgentWrapper:
    def __init__(self, name: str = "CLI Assistant", instructions: Optional[str] = None, tools: Optional[List[Callable[..., Any]]] = None):
        set_tracing_disabled(True)  # 禁用 tracing 输出
        load_dotenv()
        # 环境变量读取原样保留
        api_key = os.getenv('OPENROUTER_API_KEY')
        base_url = os.getenv('OPENROUTER_BASE_URL')
        DEFAULT_SESSION_ID = "user"
        DB_PATH = ":memory:"

        self.memory = AsyncSQLiteSession(
            session_id=DEFAULT_SESSION_ID, db_path=DB_PATH)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = OpenAIChatCompletionsModel(
            model=os.getenv('OPENROUTER_MODEL'),
            openai_client=self.client
        )
        self.agent = Agent(
            name=name,
            instructions=instructions or "你是一个有用的助手，支持多轮对话、工具调用和记忆。",
            tools=tools or [],
            model=self.model
        )

    async def stream_chat(self, user_input: str) -> AsyncGenerator[str, None]:
        result = Runner.run_streamed(
            starting_agent=self.agent,
            input=user_input,
            context=self.memory,
        )
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                delta = event.data.delta
                if delta:
                    yield delta
            elif event.type == "tool_call_item":
                yield f"\n[工具调用] {event.data.tool_name} ..."
            elif event.type == "tool_call_output_item":
                yield f"\n[工具执行完成]"

    async def chat(self, user_input: str) -> str:
        # 简单非流式接口
        output = ""
        async for delta in self.stream_chat(user_input):
            output += delta
        return output

    async def handle_input(self, user_input: str) -> None:
        """
        控制台流式输出助手回复，带格式化。
        """
        print("\n" + "="*60)
        print("🤖 Assistant: ", end="", flush=True)
        async for delta in self.stream_chat(user_input):
            print(delta, end="", flush=True)
        print("\n" + "="*60)

    def add_tool(self, tool_func: Callable[..., Any]) -> None:
        self.agent.tools.append(tool_func)

    def set_instructions(self, instructions: str) -> None:
        self.agent.instructions = instructions

    def reset_memory(self) -> None:
        self.memory = AsyncSQLiteSession(session_id="user", db_path=":memory:")
