from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class KnowledgeSource(ABC):
    """Abstract base class for knowledge sources."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Search for relevant information."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get source name."""
        pass

class CompositeKnowledgeSource:
    """Manages multiple knowledge sources."""
    
    def __init__(self):
        self.sources: Dict[str, KnowledgeSource] = {}
    
    def add_source(self, name: str, source: KnowledgeSource):
        """Add a knowledge source."""
        self.sources[name] = source
        logger.info(f"Added knowledge source: {name}")
    
    def search_all_sources(self, query: str, top_k: int = 3) -> Dict[str, List[str]]:
        """Search across all sources."""
        results = {}
        for name, source in self.sources.items():
            try:
                results[name] = source.search(query, top_k)
            except Exception as e:
                logger.warning(f"Search failed for {name}: {str(e)}")
                results[name] = []
        return results

