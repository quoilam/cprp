import asyncio
import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.extensions.memory import AsyncSQLiteSession
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from dotenv import load_dotenv
from typing import Any, AsyncGenerator, Callable, List, Optional
from tools import cli_execute, web_search

class UserContinue:
    def __init__(self):
        pass
class UserExit:
    def __init__(self):
        pass



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
        print(f"[DEBUG] stream_chat called with user_input: {user_input}")
        result = Runner.run_streamed(
            starting_agent=self.agent,
            input=user_input,
            context=self.memory,
        )
        async for event in result.stream_events():
            # print(f"[DEBUG] event: type={event.type}, data={event.data}")
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                delta = event.data.delta
                if delta:
                    # print(f"[DEBUG] delta: {delta}")
                    yield delta
            elif event.type == "tool_call_item":
                print(f"[DEBUG] tool_call_item: {event.data.tool_name}")
                yield f"\n[工具调用] {event.data.tool_name} ..."
            elif event.type == "tool_call_output_item":
                print(f"[DEBUG] tool_call_output_item")
                yield f"\n[工具执行完成]"

    async def chat(self, user_input: str) -> str:
        # 简单非流式接口
        print(f"[DEBUG] chat called with user_input: {user_input}")
        output = ""
        async for delta in self.stream_chat(user_input):
            output += delta
        print(f"[DEBUG] chat output: {output}")
        return output

    async def handle_input(self, user_input: str) -> None:
        """
        控制台流式输出助手回复，带格式化。
        """
        if not user_input:
            raise UserContinue()
        if user_input.lower() in ["exit", "quit", "退出", "q"]:
            raise UserExit()
        
        print("\n思考中...", end="", flush=True)
        
        print(f"[DEBUG] handle_input called with user_input: {user_input}")
        print("\n" + "="*60)
        print("🤖 Assistant: ", end="", flush=True)
        try:
            async for delta in self.stream_chat(user_input):
                print(delta, end="", flush=True)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Exception in handle_input: {e}")
            traceback.print_exc()
        print("\n" + "="*60)

    def add_tool(self, tool_func: Callable[..., Any]) -> None:
        self.agent.tools.append(tool_func)

    def set_instructions(self, instructions: str) -> None:
        self.agent.instructions = instructions

    def reset_memory(self) -> None:
        self.memory = AsyncSQLiteSession(session_id="user", db_path=":memory:")


if __name__ == "__main__":

    agent_wrapper = AgentWrapper(
        name="CLI Assistant",
        instructions="""
        你是一个有用的助手，可以帮助用户执行命令、查询信息等。
        当你需要执行系统命令时，必须使用 cli_execute 工具。
        调用工具如果遇到错误, 你需要分析原因并告诉用户相关信息.
        """,
        tools=[cli_execute.cli_execute, web_search.web_search],
    )

    # 多轮对话主循环
    async def chat_loop():
        print("🤖 CLI Assistant 已启动（带人工审查工具）")
        print("输入 'exit' 或 'quit' 退出对话\n")
        while True:
            try:
                user_input = input("\n你: ").strip()
                if user_input.lower() in ["exit", "quit", "退出", "q"]:
                    print("👋 对话结束，再见！")
                    break
                if not user_input:
                    continue
                print("\n思考中...", end="", flush=True)
                await agent_wrapper.handle_input(user_input)
            except KeyboardInterrupt:
                print("\n\n👋 已中断对话，再见！")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")
    print("read env from .env:")
    print(f"OPENROUTER_BASE_URL={os.getenv('OPENROUTER_BASE_URL')}")
    print(f"OPENROUTER_MODEL={os.getenv('OPENROUTER_MODEL')}")
    print(f"OPENROUTER_API_KEY={os.getenv('OPENROUTER_API_KEY')}")
    asyncio.run(chat_loop())
