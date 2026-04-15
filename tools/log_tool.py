import time
import json
import functools
from typing import Any, Callable
from agents import function_tool 


def log_tool(func: Callable) -> Callable:
    """日志中间件装饰器：记录工具调用上下文、输入、输出和耗时"""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):   # 大部分工具是 async 的
        tool_name = getattr(func, '__name__', 'unknown_tool')

        # 记录输入（避免打印过大的对象）
        input_info = {
            "args": [str(a)[:500] for a in args],   # 截断防止日志爆炸
            "kwargs": {k: str(v)[:500] for k, v in kwargs.items()}
        }

        print(f"\n🔧 [TOOL CALL] {tool_name} 开始调用")
        print(
            f"   输入参数: {json.dumps(input_info, ensure_ascii=False, default=str)}")

        start_time = time.time()

        try:
            result = await func(*args, **kwargs)

            # 记录输出（也做截断）
            output_str = str(result)[:1000]
            duration = time.time() - start_time

            print(f"   输出结果: {output_str}")
            print(f"   执行耗时: {duration:.4f} 秒")
            print(f"✅ [TOOL CALL] {tool_name} 调用成功\n")

            return result

        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ [TOOL CALL] {tool_name} 调用失败")
            print(f"   异常信息: {type(e).__name__}: {e}")
            print(f"   执行耗时: {duration:.4f} 秒\n")
            raise   # 继续抛出异常，让上层处理

    return async_wrapper
