import asyncio

from agents import function_tool
from .log_tool import log_tool

# 工具装饰器和 cli_execute 工具定义


@function_tool
@log_tool
async def cli_execute(command: str) -> str:
    import sys
    import traceback
    print(f"\n🔧 检测到工具调用：执行命令")
    print(f"   命令: {command}")
    print("   ⚠️  是否允许执行此命令？(y/n): ", end="", flush=True)
    user_input = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
    user_input = user_input.strip().lower()
    print(f"[DEBUG] 用户输入: {user_input}")
    if user_input in ["y", "yes", "确认", "是"]:
        print("✅ 用户已确认，执行命令...\n")
        try:
            import subprocess
            print("=" * 60)
            print(f"[DEBUG] 开始执行命令: {command}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout + result.stderr
            print(f"[DEBUG] 命令返回码: {result.returncode}")
            print("=" * 60)
            
            if result.returncode == 0:
                return f"\n命令`{command}`执行成功！\n 输出结果: `{output.strip()}`"
            else:
                return f"\n执行失败 (code={result.returncode}):\n{output.strip()}"

        except subprocess.TimeoutExpired:
            print(f"[ERROR] 命令执行超时: {command}")
            return "命令执行超时（超过30秒）"
        except Exception as e:
            print(f"[ERROR] 执行异常: {e}")
            traceback.print_exc()
            return f"执行异常: {str(e)}"
    else:
        print("❌ 用户拒绝执行命令")
        return "用户已拒绝执行此命令。"
