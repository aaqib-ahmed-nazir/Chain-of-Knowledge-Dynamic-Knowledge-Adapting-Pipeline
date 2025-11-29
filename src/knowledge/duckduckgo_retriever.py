from typing import List
import logging
from ddgs import DDGS
from .sources import KnowledgeSource

logger = logging.getLogger(__name__)


class DuckDuckGoRetriever(KnowledgeSource):
    """Web search using DuckDuckGo (used in parallel with Wikipedia)."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.request_count = 0
        self.request_limit = 100 

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Search DuckDuckGo and return top k results."""
        try:
            # Back off if approaching limit
            if self.request_count > self.request_limit:
                logger.warning(
                    "DuckDuckGo request limit approaching - returning empty"
                )
                return []

            # Execute search
            ddgs = DDGS(timeout=self.timeout)
            results = list(ddgs.text(query, max_results=top_k))

            if not results:
                logger.debug(f"DuckDuckGo: No results for '{query}'")
                return []

            # Extract and clean results
            summaries = []
            for result in results:
                # Get snippet/body
                snippet = result.get("body", "").strip()
                if not snippet:
                    snippet = result.get("title", "").strip()

                # Limit to 200 chars
                snippet = snippet[:200]
                if snippet:
                    summaries.append(snippet)

            self.request_count += 1
            logger.debug(f"DuckDuckGo: Found {len(summaries)} results for '{query}'")
            return summaries

        except Exception as e:
            logger.debug(f"DuckDuckGo search failed: {str(e)}")
            return []

    def get_name(self) -> str:
        return "duckduckgo"
