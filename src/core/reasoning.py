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
        
        # Normalize answers for better comparison
        normalized_answers = [self._normalize_answer(a) for a in answers]
        
        from collections import Counter
        counter = Counter(normalized_answers)
        most_common_answer, count = counter.most_common(1)[0]
        agreement = count / len(answers)
        
        has_consensus = agreement > threshold
        logger.info(f"Consensus check: {agreement:.2%} agreement (threshold: {threshold:.0%})")
        if not has_consensus:
            logger.info(f"Answers vary - will proceed to full pipeline")
        
        return has_consensus
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison (remove extra words, lowercase, etc.)."""
        # Remove common prefixes
        prefixes = ["the answer is", "answer:", "therefore", "thus", "so"]
        answer_lower = answer.lower().strip()
        for prefix in prefixes:
            if answer_lower.startswith(prefix):
                answer_lower = answer_lower[len(prefix):].strip().lstrip(':').strip()
        
        # Take first 100 chars and remove punctuation for comparison
        answer_lower = answer_lower[:100]
        # Remove extra whitespace
        answer_lower = ' '.join(answer_lower.split())
        return answer_lower
    
    def _extract_answer(self, rationale: str) -> str:
        """Extract answer from rationale - improved version."""
        # Clean markdown formatting
        rationale = rationale.replace('**', '').replace('*', '').strip()
        
        # First priority: explicit answer markers
        indicators = ["Answer:", "Final Answer:", "Answer is", "The answer is", "Conclusion:"]
        for indicator in indicators:
            if indicator.lower() in rationale.lower():
                parts = rationale.lower().split(indicator.lower())
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    # Clean up
                    answer = answer.split('\n')[0].strip()  # First line only
                    answer = answer.split('Therefore')[0].strip()
                    answer = answer.split('Thus')[0].strip()
                    answer = answer.split('Hence')[0].strip()
                    # Remove leading punctuation
                    answer = answer.lstrip(':').strip()
                    # Take first sentence if multiple
                    if '. ' in answer:
                        answer = answer.split('. ')[0] + '.'
                    return answer.strip()
        
        # Second priority: last sentence
        sentences = [s.strip() for s in rationale.split('.') if s.strip()]
        if sentences:
            last_sentence = sentences[-1]
            # Remove common prefixes
            for prefix in ["So ", "Thus ", "Therefore ", "Hence "]:
                if last_sentence.startswith(prefix):
                    return last_sentence[len(prefix):].strip()
            return last_sentence
        
        return rationale.strip()[:100]  # Fallback
    
    def _parse_domains(self, domains_str: str) -> List[str]:
        """Parse domains from text - improved version."""
        domains_str = domains_str.lower()
        found_domains = []
        
        # Domain keywords mapping
        domain_keywords = {
            'factual': ['factual', 'wikipedia', 'historical', 'geographic', 'political', 'general knowledge'],
            'medical': ['medical', 'health', 'disease', 'treatment', 'medicine', 'patient', 'clinical', 'diagnosis'],
            'physics': ['physics', 'force', 'energy', 'motion', 'quantum', 'relativity', 'mechanics', 'electromagnetic'],
            'biology': ['biology', 'organism', 'cell', 'genetics', 'evolution', 'species', 'molecular', 'biochemical']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in domains_str for keyword in keywords):
                found_domains.append(domain)
        
        return found_domains if found_domains else ['factual']

