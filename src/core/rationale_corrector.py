import logging
from src.utils.prompt_templates import RATIONALE_CORRECTION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class RationaleCorrector:
    """Correct rationales using retrieved knowledge."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def correct_rationale(self, original_rationale: str, supporting_knowledge: str) -> str:
        """Correct rationale using retrieved knowledge."""
        if not supporting_knowledge or supporting_knowledge == "No results found":
            logger.debug("No supporting knowledge - returning original rationale")
            return original_rationale
        
        prompt = RATIONALE_CORRECTION_PROMPT_TEMPLATE.format(
            original_rationale=original_rationale,
            supporting_knowledge=supporting_knowledge
        )
        
        corrected = self.llm_client.call(prompt, temperature=0.0)
        logger.debug("Rationale corrected")
        return corrected.strip()

