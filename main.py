from interface.agent_wrapper import AgentWrapper, UserContinue, UserExit
from tools import cli_execute, web_search
import asyncio
from prompts import *


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
        print(WELCOME_MESSAGE)
        while True:
            try:
                user_input = input("\n你: ").strip()
                await agent_wrapper.handle_input(user_input)
            except KeyboardInterrupt or UserExit:
                print("\n\n👋 已中断对话，再见！")
                break
            except UserContinue:
                continue
            except Exception as e:
                print(f"\n发生错误: {e}")
    asyncio.run(chat_loop())
