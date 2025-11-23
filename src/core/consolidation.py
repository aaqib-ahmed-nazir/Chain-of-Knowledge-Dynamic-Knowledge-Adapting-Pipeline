from typing import List
import logging
from src.utils.prompt_templates import ANSWER_CONSOLIDATION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class AnswerConsolidation:
    """Stage 3: Consolidate final answer."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def consolidate(self, question: str, corrected_rationales: List[str]) -> str:
        """Generate final consolidated answer."""
        rationale_text = "\n".join([
            f"{i+1}. {rationale}" 
            for i, rationale in enumerate(corrected_rationales)
        ])
        
        prompt = ANSWER_CONSOLIDATION_PROMPT_TEMPLATE.format(
            question=question,
            reasoning_steps=rationale_text
        )
        
        answer = self.llm_client.call(prompt, temperature=0.0)
        logger.info("Final answer consolidated")
        return answer.strip()

