"""
tools/serper_tool.py
Custom Serper Dev API tool for web search (MANDATORY as per assignment).
"""

import os
import requests
import logging
from crewai.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


class SerperSearchTool(BaseTool):
    """
    Mandatory Serper Dev API search tool.
    Used by agents to fetch real-time travel information from the web.
    """
    name: str = "SerperSearch"
    description: str = (
        "Search the internet for real-time travel information using Serper Dev API. "
        "Use this to find destination highlights, attractions, accommodation options, "
        "transport costs, local food prices, and current travel advisories. "
        "Input should be a specific search query string."
    )
    api_key: str = Field(default="")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("SERPER_API_KEY", "")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables!")

    def _run(self, query: str) -> str:
        """Execute a web search using Serper Dev API."""
        logger.info(f"[SerperTool] Searching: {query}")

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": 5, "gl": "us", "hl": "en"}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = []

            if "answerBox" in data:
                ab = data["answerBox"]
                answer = ab.get("answer") or ab.get("snippet") or ""
                if answer:
                    results.append(f"[Direct Answer]: {answer}")

            for i, result in enumerate(data.get("organic", [])[:5], 1):
                title   = result.get("title", "")
                snippet = result.get("snippet", "")
                link    = result.get("link", "")
                results.append(f"[Result {i}] {title}\n  {snippet}\n  Source: {link}")

            if "knowledgeGraph" in data:
                desc = data["knowledgeGraph"].get("description", "")
                if desc:
                    results.append(f"[Knowledge Graph]: {desc}")

            if not results:
                return "No search results found."

            output = f"Search Results for: '{query}'\n" + "="*50 + "\n"
            output += "\n\n".join(results)
            logger.info(f"[SerperTool] Got {len(results)} results for: {query}")
            return output

        except requests.exceptions.Timeout:
            msg = f"[SerperTool ERROR] Request timed out for query: {query}"
            logger.error(msg)
            return msg
        except requests.exceptions.HTTPError as e:
            msg = f"[SerperTool ERROR] HTTP {e.response.status_code}: {e.response.text}"
            logger.error(msg)
            return msg
        except Exception as e:
            msg = f"[SerperTool ERROR] Unexpected error: {str(e)}"
            logger.error(msg)
            return msg
