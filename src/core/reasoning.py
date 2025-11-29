from typing import List
import logging
from config.settings import config
from src.utils.prompt_templates import (
    REASONING_PROMPT_TEMPLATE,
    FEVER_REASONING_PROMPT_TEMPLATE,
    DOMAIN_IDENTIFICATION_PROMPT_TEMPLATE
)

logger = logging.getLogger(__name__)

class ReasoningPreparation:
    """Stage 1: Generate initial rationales and identify domains."""
    
    def __init__(self, llm_client, k: int = 5):
        self.llm_client = llm_client
        self.k = k
    
    def _estimate_question_complexity(self, question: str, dataset_name: str = None) -> float:
        """Estimate question complexity and return appropriate temperature."""
        question_lower = question.lower()
        
        # FEVER gets low temperature for consistent fact verification
        if dataset_name == 'fever':
            logger.debug("Using low temperature (0.3) for FEVER dataset")
            return 0.3
        
        # Simple factual questions - low temperature
        simple_patterns = ['what is', 'who is', 'when was', 'where is', 'what was']
        if any(pattern in question_lower for pattern in simple_patterns):
            logger.debug("Simple question detected - using temperature 0.3")
            return 0.3
        
        # Moderate complexity questions
        moderate_patterns = ['how does', 'why', 'explain', 'compare', 'describe']
        if any(pattern in question_lower for pattern in moderate_patterns):
            logger.debug("Moderate question detected - using temperature 0.6")
            return 0.6
        
        # Complex/multi-hop questions - higher temperature for diversity
        complex_patterns = ['relationship between', 'what if', 'analyze', 'evaluate', 'both', 'and also']
        if any(pattern in question_lower for pattern in complex_patterns):
            logger.debug("Complex question detected - using temperature 0.8")
            return 0.8
        
        # Default temperature
        logger.debug("Using default temperature 0.7")
        return 0.7
    
    def generate_rationales(self, question: str, dataset_name: str = None) -> List[str]:
        """Generate k rationales with adaptive temperature and dataset-specific prompts."""
        rationales = []
        
        # Quick Win 1: Adaptive temperature
        adaptive_temp = self._estimate_question_complexity(question, dataset_name)
        
        # Quick Win 2: Dataset-specific prompts
        if dataset_name == 'fever':
            prompt_template = FEVER_REASONING_PROMPT_TEMPLATE
            logger.info(f"Generating {self.k} rationales (dataset=fever, temp={adaptive_temp:.1f})")
        else:
            prompt_template = REASONING_PROMPT_TEMPLATE
            logger.info(f"Generating {self.k} rationales (temp={adaptive_temp:.1f})")
        
        for i in range(self.k):
            prompt = prompt_template.format(question=question)
            rationale = self.llm_client.call(
                prompt, 
                temperature=adaptive_temp
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
            for prefix in ["So ", "Thus ", "Therefore ", "Hence ", "Based on ", "In conclusion "]:
                if last_sentence.lower().startswith(prefix.lower()):
                    last_sentence = last_sentence[len(prefix):].strip()
            # Remove question references
            import re
            last_sentence = re.sub(r'the question[^.]*\.?\s*', '', last_sentence, flags=re.IGNORECASE)
            last_sentence = re.sub(r'"[^"]*"\s*', '', last_sentence)  # Remove quoted question text
            return last_sentence.strip()
        
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
    
    def validate_consensus_answer(self, question: str, consensus_answer: str) -> bool:
        """Validate if consensus answer is reasonable for the question."""
        validation_prompt = f"""Given this question and answer, determine if the answer is reasonable and relevant to the question.

Question: {question}
Answer: {consensus_answer}

Is this answer reasonable and relevant? Answer with YES or NO only."""
        
        validation = self.llm_client.call(validation_prompt, temperature=0.0)
        is_valid = 'yes' in validation.lower()
        
        if not is_valid:
            logger.info("Consensus answer failed validation - will use full pipeline")
        else:
            logger.info("Consensus answer validated - using early stop")
        
        return is_valid

