import os

from agents import function_tool
from tavily import TavilyClient

from .log_tool import log_tool


@function_tool
@log_tool
async def web_search(query: str) -> str:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(query)

    return response
