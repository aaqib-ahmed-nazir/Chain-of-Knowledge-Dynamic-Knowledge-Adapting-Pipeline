from typing import Tuple, List, Dict
import logging
from src.utils.prompt_templates import (
    SPARQL_GENERATION_PROMPT_TEMPLATE,
    MEDICAL_EXTRACTION_PROMPT_TEMPLATE,
    NL_QUERY_EXTRACTION_PROMPT_TEMPLATE
)
from src.knowledge.relevance_scorer import RelevanceScorer
from src.knowledge.source_ranker import KnowledgeSourceRanker

logger = logging.getLogger(__name__)

class AdaptiveQueryGenerator:
    """Stage 2: Generate domain-specific queries."""
    
    def __init__(self, llm_client, knowledge_sources):
        self.llm_client = llm_client
        self.knowledge_sources = knowledge_sources
        self.relevance_scorer = RelevanceScorer()
        self.source_ranker = KnowledgeSourceRanker()
    
    def generate_query(self, rationale: str, domain: str) -> Tuple[str, str]:
        """Generate query based on domain."""
        if domain == 'factual':
            query = self._generate_sparql_query(rationale)
            return query, 'sparql'
        elif domain == 'medical':
            query = self._generate_medical_query(rationale)
            return query, 'medical'
        else:  # physics, biology
            query = self._generate_nl_query(rationale)
            return query, 'natural_language'
    
    def execute_query(self, query: str, query_type: str, domain: str) -> str:
        """Execute query and retrieve knowledge from multiple sources in parallel."""
        try:
            all_results = []
            
            # Query all available sources in parallel
            # DuckDuckGo (primary - fast web search)
            if 'duckduckgo' in self.knowledge_sources:
                try:
                    results = self.knowledge_sources['duckduckgo'].search(query, top_k=3)
                    if results:
                        all_results.extend(results)
                        logger.debug(f"DuckDuckGo: {len(results)} results")
                except Exception as e:
                    logger.debug(f"DuckDuckGo search failed: {str(e)}")
            
            # Wikipedia (parallel)
            if 'wikipedia' in self.knowledge_sources:
                try:
                    results = self.knowledge_sources['wikipedia'].search(query, top_k=3)
                    if results:
                        all_results.extend(results)
                        logger.debug(f"Wikipedia: {len(results)} results")
                except Exception as e:
                    logger.debug(f"Wikipedia search failed: {str(e)}")
            
            # Wikidata SPARQL (only for factual/sparql queries)
            if query_type == 'sparql' and 'wikidata_sparql' in self.knowledge_sources:
                try:
                    results = self.knowledge_sources['wikidata_sparql'].search(query, top_k=3)
                    if results:
                        all_results.extend(results)
                        logger.debug(f"Wikidata SPARQL: {len(results)} results")
                except Exception as e:
                    logger.debug(f"Wikidata SPARQL failed: {str(e)}")
            
            # Score and rank combined results
            if all_results:
                scored_results = self.relevance_scorer.score_relevance(query, all_results)
                filtered = self.relevance_scorer.filter_by_threshold(scored_results, threshold=0.1)
                top_results = self.relevance_scorer.get_top_k(filtered, top_k=3)
                if top_results:
                    knowledge = "\n".join([item['content'] for item in top_results])
                    logger.info(f"Query executed: {len(top_results)} relevant results from {len(all_results)} total")
                else:
                    knowledge = "No results"
            else:
                knowledge = "No results"
            
            if knowledge == "No results":
                logger.warning(f"Query executed but no relevant results found (type={query_type})")
            
            return knowledge if knowledge else "No results found"
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}", exc_info=True)
            return "No results found"
    
    def _generate_sparql_query(self, rationale: str) -> str:
        """Generate SPARQL query for Wikidata."""
        prompt = SPARQL_GENERATION_PROMPT_TEMPLATE.format(sentence=rationale)
        query = self.llm_client.call(prompt, temperature=0.0)
        return self._clean_sparql_query(query)
    
    def _generate_medical_query(self, rationale: str) -> str:
        """Generate medical knowledge query."""
        prompt = MEDICAL_EXTRACTION_PROMPT_TEMPLATE.format(sentence=rationale)
        query = self.llm_client.call(prompt, temperature=0.0)
        query = query.strip()
        # Truncate to 300 chars for Wikipedia API limit
        if len(query) > 300:
            query = query[:297] + "..."
        return query
    
    def _generate_nl_query(self, rationale: str) -> str:
        """Generate natural language query."""
        prompt = NL_QUERY_EXTRACTION_PROMPT_TEMPLATE.format(sentence=rationale)
        query = self.llm_client.call(prompt, temperature=0.0)
        query = query.strip()
        # Truncate to 300 chars for Wikipedia API limit
        if len(query) > 300:
            query = query[:297] + "..."
        return query
    
    def _clean_sparql_query(self, query_str: str) -> str:
        """Clean SPARQL query output."""
        if '```' in query_str:
            query_str = query_str.split('```')[1]
            if query_str.startswith('sparql'):
                query_str = query_str[7:]
        return query_str.strip()

