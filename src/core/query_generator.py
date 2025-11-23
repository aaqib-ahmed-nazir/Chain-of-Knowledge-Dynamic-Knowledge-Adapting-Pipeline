from typing import Tuple
import logging
from src.utils.prompt_templates import (
    SPARQL_GENERATION_PROMPT_TEMPLATE,
    MEDICAL_EXTRACTION_PROMPT_TEMPLATE,
    NL_QUERY_EXTRACTION_PROMPT_TEMPLATE
)

logger = logging.getLogger(__name__)

class AdaptiveQueryGenerator:
    """Stage 2: Generate domain-specific queries."""
    
    def __init__(self, llm_client, knowledge_sources):
        self.llm_client = llm_client
        self.knowledge_sources = knowledge_sources
    
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
        """Execute query and retrieve knowledge."""
        try:
            if query_type == 'sparql':
                # Stub SPARQL execution - return placeholder
                knowledge = f"SPARQL query executed: {query[:100]}..."
                logger.info(f"SPARQL query generated (stubbed): {query[:50]}...")
            elif query_type == 'medical':
                if 'wikipedia' in self.knowledge_sources:
                    results = self.knowledge_sources['wikipedia'].search(query, top_k=3)
                    knowledge = "\n".join(results)
                else:
                    knowledge = "No results"
            else:  # natural_language
                if 'wikipedia' in self.knowledge_sources:
                    results = self.knowledge_sources['wikipedia'].search(query, top_k=3)
                    knowledge = "\n".join(results)
                else:
                    knowledge = "No results"
            
            logger.info(f"Query executed (type={query_type}, retrieved={len(knowledge)} chars)")
            return knowledge
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
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
        return query.strip()
    
    def _generate_nl_query(self, rationale: str) -> str:
        """Generate natural language query."""
        prompt = NL_QUERY_EXTRACTION_PROMPT_TEMPLATE.format(sentence=rationale)
        query = self.llm_client.call(prompt, temperature=0.0)
        return query.strip()
    
    def _clean_sparql_query(self, query_str: str) -> str:
        """Clean SPARQL query output."""
        if '```' in query_str:
            query_str = query_str.split('```')[1]
            if query_str.startswith('sparql'):
                query_str = query_str[7:]
        return query_str.strip()

