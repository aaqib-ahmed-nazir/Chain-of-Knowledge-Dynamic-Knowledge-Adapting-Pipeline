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
        """Execute query and retrieve knowledge with relevance scoring and source ranking."""
        try:
            if query_type == 'sparql':
                # Rank sources for SPARQL queries
                ranked_sources = self.source_ranker.rank_sources(domain, query_type, self.knowledge_sources)
                all_results = []
                
                # Try sources in priority order
                for source_name in ranked_sources:
                    try:
                        source = self.knowledge_sources[source_name]
                        if source_name == 'wikidata_sparql':
                            results = source.search(query, top_k=5)
                            if results:
                                all_results.extend(results)
                                logger.debug(f"Wikidata SPARQL query executed: {len(results)} results")
                            else:
                                # SPARQL failed or returned no results, try fallback
                                logger.debug("Wikidata SPARQL returned no results, trying fallback")
                                continue
                        else:
                            # Fallback to other sources (e.g., Wikipedia)
                            logger.debug(f"Trying fallback source: {source_name}")
                            results = source.search(query, top_k=3)
                            if results:
                                all_results.extend(results)
                        
                        # If we got results, break (don't try lower priority sources)
                        if all_results:
                            break
                    except Exception as e:
                        logger.debug(f"Source {source_name} failed: {str(e)}, trying next source")
                        continue
                
                if all_results:
                    # Score and rank results
                    scored_results = self.relevance_scorer.score_relevance(query, all_results)
                    # Filter low-relevance results and get top 3
                    filtered = self.relevance_scorer.filter_by_threshold(scored_results, threshold=0.1)
                    top_results = self.relevance_scorer.get_top_k(filtered, top_k=3)
                    knowledge = "\n".join([item['content'] for item in top_results])
                    logger.debug(f"SPARQL query executed: {len(top_results)} relevant results from {len(all_results)} total")
                else:
                    knowledge = "No results"
            elif query_type == 'medical':
                # Try multiple sources and rank results
                all_results = []
                if 'wikipedia' in self.knowledge_sources:
                    try:
                        results = self.knowledge_sources['wikipedia'].search(query, top_k=5)
                        all_results.extend(results)
                    except Exception as e:
                        logger.warning(f"Wikipedia search failed: {str(e)}")
                
                # Fallback to DuckDuckGo if Wikipedia found nothing
                if not all_results and 'duckduckgo' in self.knowledge_sources:
                    logger.debug(f"Wikipedia empty for '{query[:50]}...' - trying DuckDuckGo")
                    try:
                        results = self.knowledge_sources['duckduckgo'].search(query, top_k=3)
                        all_results.extend(results)
                    except Exception as e:
                        logger.debug(f"DuckDuckGo fallback failed: {str(e)}")
                
                # Score and rank all results
                if all_results:
                    scored_results = self.relevance_scorer.score_relevance(query, all_results)
                    filtered = self.relevance_scorer.filter_by_threshold(scored_results, threshold=0.15)
                    top_results = self.relevance_scorer.get_top_k(filtered, top_k=3)
                    knowledge = "\n".join([item['content'] for item in top_results])
                else:
                    knowledge = "No results"
            else:  # natural_language
                # Try multiple sources and rank results
                all_results = []
                if 'wikipedia' in self.knowledge_sources:
                    try:
                        results = self.knowledge_sources['wikipedia'].search(query, top_k=5)
                        all_results.extend(results)
                    except Exception as e:
                        logger.warning(f"Wikipedia search failed: {str(e)}")
                
                # Fallback to DuckDuckGo if Wikipedia found nothing
                if not all_results and 'duckduckgo' in self.knowledge_sources:
                    logger.debug(f"Wikipedia empty for '{query[:50]}...' - trying DuckDuckGo")
                    try:
                        results = self.knowledge_sources['duckduckgo'].search(query, top_k=3)
                        all_results.extend(results)
                    except Exception as e:
                        logger.debug(f"DuckDuckGo fallback failed: {str(e)}")
                
                # Score and rank all results
                if all_results:
                    scored_results = self.relevance_scorer.score_relevance(query, all_results)
                    filtered = self.relevance_scorer.filter_by_threshold(scored_results, threshold=0.15)
                    top_results = self.relevance_scorer.get_top_k(filtered, top_k=3)
                    knowledge = "\n".join([item['content'] for item in top_results])
                else:
                    knowledge = "No results"
            
            if knowledge and knowledge != "No results":
                logger.info(f"Query executed (type={query_type}, retrieved={len(knowledge)} chars)")
            else:
                logger.warning(f"Query executed but no relevant results found (type={query_type})")
            
            return knowledge if knowledge else "No results found"
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}", exc_info=True)
            # Return empty result instead of raising to allow pipeline to continue
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

