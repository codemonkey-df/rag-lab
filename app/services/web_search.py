"""
Web search service using DuckDuckGo.
"""

from typing import Dict, List


async def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search using DuckDuckGo.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of search results with title, url, content
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

        return [
            {
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "content": result.get("body", ""),
            }
            for result in results
        ]


async def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Web search using DuckDuckGo.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of search results with title, url, content
    """
    return await search_duckduckgo(query, max_results)
