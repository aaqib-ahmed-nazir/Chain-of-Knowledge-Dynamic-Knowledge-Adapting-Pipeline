"""Relevance scoring for retrieved knowledge."""
import logging
from typing import List, Dict
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class RelevanceScorer:
    """Score relevance of retrieved knowledge to query."""
    
    def __init__(self):
        """Initialize relevance scorer."""
        pass
    
    def score_relevance(self, query: str, knowledge_items: List[str]) -> List[Dict[str, any]]:
        """Score relevance of knowledge items to query.
        
        Args:
            query: Original query string
            knowledge_items: List of knowledge strings to score
            
        Returns:
            List of dicts with 'content' and 'score' keys, sorted by score (highest first)
        """
        scored_items = []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for item in knowledge_items:
            score = self._calculate_score(query, query_words, item)
            scored_items.append({
                'content': item,
                'score': score,
                'source': 'unknown'  # Can be set by caller
            })
        
        # Sort by score (highest first)
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_items
    
    def _calculate_score(self, query: str, query_words: set, knowledge: str) -> float:
        """Calculate relevance score between query and knowledge.
        
        Args:
            query: Original query
            query_words: Set of words from query
            knowledge: Knowledge string to score
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not knowledge or len(knowledge.strip()) == 0:
            return 0.0
        
        query_lower = query.lower()
        knowledge_lower = knowledge.lower()
        knowledge_words = set(knowledge_lower.split())
        
        # Score components
        scores = []
        
        # 1. Word overlap score (0-1)
        if query_words:
            overlap = len(query_words & knowledge_words) / len(query_words)
            scores.append(overlap * 0.4)  # 40% weight
        
        # 2. Substring match score (0-1)
        # Check if query appears in knowledge
        if query_lower in knowledge_lower:
            scores.append(0.3)  # 30% weight for exact substring match
        elif any(word in knowledge_lower for word in query_words if len(word) > 3):
            scores.append(0.15)  # 15% weight for partial match
        
        # 3. Sequence similarity (0-1)
        similarity = SequenceMatcher(None, query_lower, knowledge_lower[:500]).ratio()
        scores.append(similarity * 0.3)  # 30% weight
        
        # Combine scores
        total_score = sum(scores)
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, total_score))
    
    def filter_by_threshold(self, scored_items: List[Dict], threshold: float = 0.1) -> List[Dict]:
        """Filter scored items by relevance threshold.
        
        Args:
            scored_items: List of scored items
            threshold: Minimum score threshold
            
        Returns:
            Filtered list of items above threshold
        """
        return [item for item in scored_items if item['score'] >= threshold]
    
    def get_top_k(self, scored_items: List[Dict], top_k: int = 3) -> List[Dict]:
        """Get top k most relevant items.
        
        Args:
            scored_items: List of scored items (should be sorted)
            top_k: Number of items to return
            
        Returns:
            Top k items
        """
        return scored_items[:top_k]

