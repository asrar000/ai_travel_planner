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
        "Search web via Serper for current travel information. Input: query string."
    )
    api_key: str = Field(default="")
    max_results: int = Field(default=2)
    max_snippet_chars: int = Field(default=120)
    include_snippet: bool = Field(default=False)
    _cache: dict[str, str] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("SERPER_API_KEY", "")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables!")

        # Keep tool outputs compact to reduce LLM token pressure.
        self.max_results = self._parse_positive_int_env("SERPER_RESULTS_LIMIT", 2, upper=5)
        self.max_snippet_chars = self._parse_positive_int_env("SERPER_SNIPPET_MAX_CHARS", 120, upper=500)
        self.include_snippet = self._parse_bool_env("SERPER_INCLUDE_SNIPPET", False)

    @staticmethod
    def _parse_positive_int_env(name: str, default: int, upper: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value <= 0:
                return default
            return min(value, upper)
        except ValueError:
            return default

    @staticmethod
    def _parse_bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name, "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    def _compact(self, text: str) -> str:
        text = " ".join((text or "").split())
        if len(text) <= self.max_snippet_chars:
            return text
        return text[: self.max_snippet_chars - 1].rstrip() + "…"

    def _run(self, query: str) -> str:
        """Execute a web search using Serper Dev API."""
        if not query or not query.strip():
            raise ValueError("SerperSearch query must not be empty.")

        query = " ".join(query.split())
        if query in self._cache:
            logger.info(f"[SerperTool] Cache hit: {query}")
            return self._cache[query]

        logger.info(f"[SerperTool] Searching: {query}")

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": self.max_results, "gl": "us", "hl": "en"}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = []
            for i, result in enumerate(data.get("organic", [])[: self.max_results], 1):
                title = self._compact(result.get("title", ""))
                snippet = self._compact(result.get("snippet", ""))
                link = self._compact(result.get("link", ""))
                line = f"{i}. {title}"
                if self.include_snippet and snippet:
                    line += f" | {snippet}"
                if link:
                    line += f" | Source: {link}"
                results.append(line)

            if not results:
                return "No search results found."

            output = "\n".join(results)
            self._cache[query] = output
            logger.info(f"[SerperTool] Got {len(results)} results for: {query}")
            return output

        except requests.exceptions.Timeout:
            msg = f"[SerperTool ERROR] Request timed out for query: {query}"
            logger.error(msg)
            raise RuntimeError(msg)
        except requests.exceptions.HTTPError as e:
            msg = f"[SerperTool ERROR] HTTP {e.response.status_code}: {e.response.text}"
            logger.error(msg)
            raise RuntimeError(msg)
        except Exception as e:
            msg = f"[SerperTool ERROR] Unexpected error: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg)
