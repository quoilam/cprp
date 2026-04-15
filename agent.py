import os
import asyncio
import sys
from interface.agent_wrapper import AgentWrapper
from tools import cli_execute

# AgentWrapper 实例

agent_wrapper = AgentWrapper(
    name="CLI Assistant",
    instructions="""
    你是一个有用的助手，可以帮助用户执行命令、查询信息等。
    当你需要执行系统命令时，必须使用 cli_execute 工具。
    """,
    tools=[cli_execute],
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

if __name__ == "__main__":
    print("read env from .env:")
    print(f"OPENROUTER_BASE_URL={os.getenv('OPENROUTER_BASE_URL')}")
    print(f"OPENROUTER_MODEL={os.getenv('OPENROUTER_MODEL')}")
    print(f"OPENROUTER_API_KEY={os.getenv('OPENROUTER_API_KEY')}")
    asyncio.run(chat_loop())
