from agents import function_tool

# 工具装饰器和 cli_execute 工具定义


@function_tool
async def cli_execute(command: str) -> str:
    print(f"\n🔧 检测到工具调用：执行命令")
    print(f"   命令: {command}")
    print("   ⚠️  是否允许执行此命令？(y/n): ", end="", flush=True)
    user_input = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
    user_input = user_input.strip().lower()
    if user_input in ["y", "yes", "确认", "是"]:
        print("✅ 用户已确认，执行命令...\n")
        try:
            import subprocess
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout + result.stderr
            if result.returncode == 0:
                return f"执行成功！\n{output.strip()}"
            else:
                return f"执行失败 (code={result.returncode}):\n{output.strip()}"
        except subprocess.TimeoutExpired:
            return "命令执行超时（超过30秒）"
        except Exception as e:
            return f"执行异常: {str(e)}"
    else:
        print("❌ 用户拒绝执行命令")
        return "用户已拒绝执行此命令。"
