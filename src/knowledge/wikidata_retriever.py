"""Wikidata SPARQL retriever implementation."""
import requests
import logging
import re
from typing import List
from .sources import KnowledgeSource

logger = logging.getLogger(__name__)

class WikidataSPARQLRetriever(KnowledgeSource):
    """Retrieve information from Wikidata using SPARQL queries."""
    
    WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    
    def __init__(self, timeout: int = 10):
        """Initialize Wikidata SPARQL retriever.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
    
    def search(self, sparql_query: str, top_k: int = 3) -> List[str]:
        """Execute SPARQL query and return top k results.
        
        Args:
            sparql_query: SPARQL query string
            top_k: Number of results to return
            
        Returns:
            List of result strings (formatted from query results)
        """
        try:
            # Clean and validate SPARQL query
            cleaned_query = self._clean_query(sparql_query)
            if not self._is_valid_sparql(cleaned_query):
                logger.debug(f"Invalid SPARQL query, skipping: {cleaned_query[:100]}...")
                return []
            
            # Execute SPARQL query
            results = self._execute_sparql(cleaned_query)
            
            # Format results
            formatted_results = self._format_results(results, top_k)
            
            if formatted_results:
                logger.info(f"Wikidata SPARQL query returned {len(formatted_results)} results")
            else:
                logger.debug(f"Wikidata SPARQL query returned no results")
            
            return formatted_results
            
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (400 Bad Request, etc.)
            if e.response.status_code == 400:
                logger.debug(f"Invalid SPARQL query syntax (400): {sparql_query[:100]}...")
            else:
                logger.warning(f"Wikidata SPARQL HTTP error {e.response.status_code}: {str(e)}")
            return []
        except Exception as e:
            logger.debug(f"Wikidata SPARQL query failed: {str(e)}")
            return []
    
    def _execute_sparql(self, query: str) -> dict:
        """Execute SPARQL query against Wikidata endpoint.
        
        Args:
            query: SPARQL query string
            
        Returns:
            JSON response from Wikidata
        """
        headers = {
            'User-Agent': 'Chain-of-Knowledge/1.0 (https://github.com/your-repo)',
            'Accept': 'application/sparql-results+json'
        }
        
        params = {'query': query, 'format': 'json'}
        
        response = requests.get(
            self.WIKIDATA_SPARQL_ENDPOINT,
            params=params,
            headers=headers,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def _format_results(self, results: dict, top_k: int) -> List[str]:
        """Format SPARQL results into readable strings.
        
        Args:
            results: JSON response from Wikidata
            top_k: Maximum number of results to return
            
        Returns:
            List of formatted result strings
        """
        formatted = []
        
        try:
            bindings = results.get('results', {}).get('bindings', [])
            
            for i, binding in enumerate(bindings[:top_k]):
                # Extract values from binding
                values = []
                for key, value_info in binding.items():
                    value = value_info.get('value', '')
                    # Format URIs nicely
                    if value.startswith('http'):
                        # Extract entity name from URI
                        if '/entity/' in value:
                            entity_id = value.split('/entity/')[-1]
                            values.append(f"{key}: {entity_id}")
                        else:
                            values.append(f"{key}: {value}")
                    else:
                        values.append(f"{key}: {value}")
                
                if values:
                    formatted.append(" | ".join(values))
            
            # If no structured results, try to extract any text
            if not formatted and 'results' in results:
                formatted.append("Wikidata query executed successfully")
                
        except Exception as e:
            logger.warning(f"Error formatting SPARQL results: {str(e)}")
            formatted.append("Wikidata query executed but results could not be formatted")
        
        return formatted if formatted else []
    
    def _clean_query(self, query: str) -> str:
        """Clean SPARQL query string.
        
        Args:
            query: Raw query string (may contain markdown, extra text)
            
        Returns:
            Cleaned SPARQL query
        """
        # Remove markdown code blocks
        if '```' in query:
            parts = query.split('```')
            for part in parts:
                if 'SELECT' in part.upper() or 'ASK' in part.upper():
                    query = part
                    break
        
        # Remove language markers
        if query.startswith('sparql'):
            query = query[6:].strip()
        if query.startswith('SPARQL'):
            query = query[6:].strip()
        
        # Remove leading/trailing whitespace
        query = query.strip()
        
        # Remove common prefixes that LLM might add incorrectly
        lines = query.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip comments
            if line.startswith('#'):
                continue
            # Skip empty lines
            if not line:
                continue
            # Fix common syntax errors
            # Fix: "wd:Q664 New Zealand." -> "wd:Q664 rdfs:label 'New Zealand'@en."
            # But for now, just keep the line as-is
            cleaned_lines.append(line)
        
        query = '\n'.join(cleaned_lines)
        
        # Ensure query ends properly
        if not query.endswith('}'):
            # Try to find the last complete statement
            if '}' in query:
                query = query[:query.rfind('}') + 1]
        
        return query.strip()
    
    def _is_valid_sparql(self, query: str) -> bool:
        """Check if query looks like valid SPARQL.
        
        Args:
            query: Query string to validate
            
        Returns:
            True if query appears valid
        """
        if not query or len(query.strip()) < 20:
            return False
        
        query_upper = query.upper()
        
        # Must contain SELECT or ASK
        if 'SELECT' not in query_upper and 'ASK' not in query_upper:
            return False
        
        # Must contain WHERE clause
        if 'WHERE' not in query_upper:
            return False
        
        # Check for balanced braces
        if query.count('{') != query.count('}'):
            return False
        
        # Check for common syntax errors that cause 400 errors
        # Invalid patterns that cause issues
        invalid_patterns = [
            r'wd:\w+\s+[A-Z]',  # e.g., "wd:Q664 New Zealand" (missing predicate)
            r'wdt:\w+\s+[A-Z]',  # e.g., "wdt:P39 Canada" (missing object properly formatted)
        ]
        
        import re
        for pattern in invalid_patterns:
            if re.search(pattern, query):
                return False
        
        return True
    
    def get_name(self) -> str:
        """Get source name."""
        return "wikidata_sparql"

