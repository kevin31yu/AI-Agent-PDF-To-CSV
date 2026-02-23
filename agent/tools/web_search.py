import os
from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch


def search_web(query: str) -> str:
    """Run a Tavily web search and return formatted results."""
    # Initialised here (not at module level) so TAVILY_API_KEY is already loaded
    tool = TavilySearch(max_results=5, api_key=os.getenv("TAVILY_API_KEY"))
    results = tool.invoke(query)

    if not results:
        return "No results found."

    # TavilySearch returns a dict with a "results" key
    items = results.get("results", results) if isinstance(results, dict) else results

    lines = []
    for i, r in enumerate(items, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"[{i}] {title}\n    {url}\n    {content}\n")

    return "\n".join(lines) if lines else "No results found."
