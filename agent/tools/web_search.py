from langchain_community.tools.tavily_search import TavilySearchResults

# Returns up to 5 results with full content snippets
tavily_tool = TavilySearchResults(max_results=5)


def search_web(query: str) -> str:
    """Run a Tavily web search and return formatted results."""
    results = tavily_tool.invoke(query)

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    {r['url']}\n    {r['content']}\n")

    return "\n".join(lines)
