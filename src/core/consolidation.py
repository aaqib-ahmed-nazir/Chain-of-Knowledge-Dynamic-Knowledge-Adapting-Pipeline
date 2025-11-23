from typing import List
import logging
import re
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
        
        # Detect if this is a FEVER-style question (fact verification)
        is_fever_style = any(keyword in question.lower() for keyword in ['claim', 'statement', 'supports', 'refutes'])
        
        if is_fever_style:
            # Use FEVER-specific prompt
            prompt = self._get_fever_consolidation_prompt(question, rationale_text)
        else:
            prompt = ANSWER_CONSOLIDATION_PROMPT_TEMPLATE.format(
                question=question,
                reasoning_steps=rationale_text
            )
        
        answer = self.llm_client.call(prompt, temperature=0.0)
        logger.info("Final answer consolidated")
        
        # Extract concise answer from response
        extracted = self._extract_final_answer(answer)
        return extracted
    
    def _get_fever_consolidation_prompt(self, question: str, reasoning_steps: str) -> str:
        """Get FEVER-specific consolidation prompt."""
        return f"""Based on the following reasoning steps, determine if the claim is SUPPORTS, REFUTES, or NOT ENOUGH INFO.

Question/Claim: {question}

Reasoning steps:
{reasoning_steps}

You must respond with ONLY one of these three labels:
- SUPPORTS (if the claim is supported by evidence)
- REFUTES (if the claim is contradicted by evidence)
- NOT ENOUGH INFO (if there is insufficient information to verify)

Do NOT provide explanations. Respond with ONLY the label.

Final Answer:"""
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract concise final answer from LLM response."""
        response = response.strip()
        
        # Remove common prefixes that add unnecessary text
        prefixes_to_remove = [
            r"Therefore[,\s]+",
            r"Thus[,\s]+",
            r"Hence[,\s]+",
            r"So[,\s]+",
            r"Based on[^,]*,\s*",
            r"In conclusion[,\s]+",
            r"To answer[^,]*,\s*",
            r"The answer to[^,]*,\s*",
        ]
        
        cleaned_response = response
        for prefix in prefixes_to_remove:
            cleaned_response = re.sub(prefix, "", cleaned_response, flags=re.IGNORECASE)
        
        # Look for common answer indicators
        patterns = [
            r"Final Answer:\s*(.+?)(?:\.|$|\n)",
            r"Answer:\s*(.+?)(?:\.|$|\n)",
            r"The answer is\s*(.+?)(?:\.|$|\n)",
            r"is\s+(.+?)(?:\.|$|\n)",  # "is Paris" or "is William Shakespeare"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Remove quotes if present
                answer = answer.strip('"\'')
                # Remove "the question" or similar phrases
                answer = re.sub(r'the question[^.]*\.?\s*', '', answer, flags=re.IGNORECASE)
                answer = re.sub(r'"[^"]*"\s*', '', answer)  # Remove quoted question text
                # Take first sentence only
                if '. ' in answer:
                    answer = answer.split('. ')[0]
                # Remove trailing punctuation except for single letters (A, B, C, D)
                if len(answer) > 1 and answer[-1] in '.,;:':
                    answer = answer[:-1]
                return answer[:200].strip()  # Limit length
        
        # If no pattern found, take last sentence (often contains the answer)
        sentences = [s.strip() for s in cleaned_response.split('.') if s.strip()]
        if sentences:
            last_sentence = sentences[-1]
            # Remove common prefixes from last sentence
            for prefix in ["Therefore ", "Thus ", "Hence ", "So "]:
                if last_sentence.startswith(prefix):
                    last_sentence = last_sentence[len(prefix):].strip()
            # Remove quotes and question references
            last_sentence = last_sentence.strip('"\'')
            last_sentence = re.sub(r'the question[^.]*\.?\s*', '', last_sentence, flags=re.IGNORECASE)
            return last_sentence[:200].strip()
        
        return cleaned_response[:200].strip()

