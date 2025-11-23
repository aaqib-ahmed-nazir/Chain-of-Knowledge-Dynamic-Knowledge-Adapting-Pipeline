"""Knowledge source ranking and selection."""
import logging
from typing import List, Dict
from .sources import KnowledgeSource

logger = logging.getLogger(__name__)

class KnowledgeSourceRanker:
    """Rank and select knowledge sources based on domain and query type."""
    
    def __init__(self):
        """Initialize source ranker."""
        # Domain to source priority mapping
        self.domain_source_priority = {
            'factual': ['wikidata_sparql', 'wikipedia'],
            'medical': ['wikipedia', 'wikidata_sparql'],
            'physics': ['wikipedia', 'wikidata_sparql'],
            'biology': ['wikipedia', 'wikidata_sparql']
        }
        
        # Query type to source priority mapping
        self.query_type_source_priority = {
            'sparql': ['wikidata_sparql'],
            'medical': ['wikipedia'],
            'natural_language': ['wikipedia', 'wikidata_sparql']
        }
    
    def rank_sources(self, domain: str, query_type: str, 
                    available_sources: Dict[str, KnowledgeSource]) -> List[str]:
        """Rank available sources by relevance to domain and query type.
        
        Args:
            domain: Knowledge domain (factual, medical, physics, biology)
            query_type: Type of query (sparql, medical, natural_language)
            available_sources: Dictionary of available knowledge sources
            
        Returns:
            List of source names ranked by priority
        """
        # Get priority lists
        domain_priority = self.domain_source_priority.get(domain, ['wikipedia'])
        query_priority = self.query_type_source_priority.get(query_type, ['wikipedia'])
        
        # Combine priorities (query type takes precedence)
        combined_priority = []
        
        # First add query type priorities
        for source in query_priority:
            if source in available_sources and source not in combined_priority:
                combined_priority.append(source)
        
        # Then add domain priorities
        for source in domain_priority:
            if source in available_sources and source not in combined_priority:
                combined_priority.append(source)
        
        # Add any remaining sources
        for source in available_sources.keys():
            if source not in combined_priority:
                combined_priority.append(source)
        
        logger.debug(f"Ranked sources for domain={domain}, type={query_type}: {combined_priority}")
        return combined_priority
    
    def select_best_source(self, domain: str, query_type: str,
                          available_sources: Dict[str, KnowledgeSource]) -> str:
        """Select the best source for given domain and query type.
        
        Args:
            domain: Knowledge domain
            query_type: Type of query
            available_sources: Dictionary of available knowledge sources
            
        Returns:
            Name of best source, or None if no sources available
        """
        ranked = self.rank_sources(domain, query_type, available_sources)
        return ranked[0] if ranked else None
    
    def get_fallback_sources(self, domain: str, query_type: str,
                            available_sources: Dict[str, KnowledgeSource],
                            exclude: List[str] = None) -> List[str]:
        """Get fallback sources if primary source fails.
        
        Args:
            domain: Knowledge domain
            query_type: Type of query
            available_sources: Dictionary of available knowledge sources
            exclude: List of source names to exclude
            
        Returns:
            List of fallback source names
        """
        exclude = exclude or []
        ranked = self.rank_sources(domain, query_type, available_sources)
        return [s for s in ranked if s not in exclude]

