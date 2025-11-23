import wikipedia
from typing import List
import logging
from .sources import KnowledgeSource

logger = logging.getLogger(__name__)

class WikipediaRetriever(KnowledgeSource):
    """Retrieve information from Wikipedia."""
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Search Wikipedia and return top k results."""
        try:
            # Truncate query if too long (Wikipedia API limit is 300 chars)
            if len(query) > 300:
                query = query[:297] + "..."
                logger.debug(f"Truncated query to {len(query)} chars")
            
            results = wikipedia.search(query, results=top_k)
            summaries = []
            
            for result in results:
                try:
                    page = wikipedia.page(result)
                    summary = page.summary[:300]
                    summaries.append(summary)
                except Exception as e:
                    logger.debug(f"Could not retrieve page for {result}: {str(e)}")
                    continue
            
            return summaries if summaries else ["No results found"]
        except Exception as e:
            logger.error(f"Wikipedia search failed: {str(e)}")
            return ["No results found"]
    
    def get_name(self) -> str:
        return "wikipedia"

