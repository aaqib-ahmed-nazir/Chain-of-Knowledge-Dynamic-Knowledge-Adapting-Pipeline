from typing import List
import logging
from config.settings import config
from src.utils.prompt_templates import (
    REASONING_PROMPT_TEMPLATE,
    DOMAIN_IDENTIFICATION_PROMPT_TEMPLATE
)

logger = logging.getLogger(__name__)

class ReasoningPreparation:
    """Stage 1: Generate initial rationales and identify domains."""
    
    def __init__(self, llm_client, k: int = 5):
        self.llm_client = llm_client
        self.k = k
    
    def generate_rationales(self, question: str) -> List[str]:
        """Generate k rationales using chain-of-thought."""
        rationales = []
        logger.info(f"Generating {self.k} rationales")
        
        for i in range(self.k):
            prompt = REASONING_PROMPT_TEMPLATE.format(question=question)
            rationale = self.llm_client.call(
                prompt, 
                temperature=config.REASONING_TEMPERATURE
            )
            rationales.append(rationale)
            logger.debug(f"Generated rationale {i+1}/{self.k}")
        
        return rationales
    
    def generate_answers(self, question: str, rationales: List[str]) -> List[str]:
        """Extract answers from rationales."""
        answers = []
        for rationale in rationales:
            answer = self._extract_answer(rationale)
            answers.append(answer)
        logger.info(f"Extracted {len(answers)} answers")
        return answers
    
    def identify_domains(self, question: str) -> List[str]:
        """Identify relevant knowledge domains."""
        prompt = DOMAIN_IDENTIFICATION_PROMPT_TEMPLATE.format(question=question)
        response = self.llm_client.call(prompt, temperature=0.0)
        domains = self._parse_domains(response)
        logger.info(f"Identified domains: {domains}")
        return domains
    
    def has_consensus(self, answers: List[str], threshold: float = 0.5) -> bool:
        """Check if answers have consensus."""
        if not answers:
            return False
        
        from collections import Counter
        counter = Counter(answers)
        most_common_answer, count = counter.most_common(1)[0]
        agreement = count / len(answers)
        
        has_consensus = agreement > threshold
        logger.info(f"Consensus check: {agreement:.2%} agreement (threshold: {threshold:.0%})")
        
        return has_consensus
    
    def _extract_answer(self, rationale: str) -> str:
        """Extract answer from rationale."""
        indicators = ["Answer:", "Final Answer:", "Conclusion:"]
        for indicator in indicators:
            if indicator in rationale:
                return rationale.split(indicator)[-1].strip()
        sentences = rationale.split('. ')
        return sentences[-1].strip() if sentences else rationale.strip()
    
    def _parse_domains(self, domains_str: str) -> List[str]:
        """Parse domains from text."""
        valid_domains = ['factual', 'medical', 'physics', 'biology']
        found_domains = []
        
        for domain in valid_domains:
            if domain.lower() in domains_str.lower():
                found_domains.append(domain)
        
        return found_domains if found_domains else ['factual']

