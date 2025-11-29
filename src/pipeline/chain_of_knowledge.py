from typing import Dict, List
import logging
from config.settings import config
from src.core.reasoning import ReasoningPreparation
from src.core.query_generator import AdaptiveQueryGenerator
from src.core.rationale_corrector import RationaleCorrector
from src.core.consolidation import AnswerConsolidation

logger = logging.getLogger(__name__)

class ChainOfKnowledge:
    """Main CoK pipeline orchestrator - uses Together AI (Llama 3 70B) for all stages."""
    
    def __init__(self, llm_client, knowledge_sources):
        """Initialize CoK pipeline with a single LLM client for all stages.
        
        Args:
            llm_client: LLM client (TogetherAIClient with Llama 3 70B)
            knowledge_sources: Dictionary of knowledge sources
        """
        # Use same LLM client for all stages for consistency
        self.llm_client = llm_client
        self.reasoning = ReasoningPreparation(llm_client)
        self.query_generator = AdaptiveQueryGenerator(llm_client, knowledge_sources)
        self.corrector = RationaleCorrector(llm_client)
        self.consolidation = AnswerConsolidation(llm_client)
        logger.info("CoK pipeline initialized with Together AI (Llama 3 70B)")
    
    def run(self, question: str, dataset_name: str = None) -> Dict:
        """Execute full CoK pipeline."""
        logger.info(f"Processing question: {question[:100]}...")
        
        # Use fewer rationales for FEVER to save tokens (full pipeline uses more API calls)
        if "Claim:" in question:
            original_k = self.reasoning.k
            self.reasoning.k = config.NUM_RATIONALES_FEVER
            logger.debug(f"Using {self.reasoning.k} rationales for FEVER question")
        
        # Stage 1: Reasoning Preparation (pass dataset_name for adaptive prompts)
        rationales = self.reasoning.generate_rationales(question, dataset_name)
        
        # Restore original k if changed
        if "Claim:" in question:
            self.reasoning.k = original_k
        answers = self.reasoning.generate_answers(question, rationales)
        domains = self.reasoning.identify_domains(question)
        
        # For FEVER-style questions, ALWAYS run full pipeline (no early stopping)
        # Early stopping returns explanations instead of labels, which hurts accuracy
        is_fever_style = "Claim:" in question
        
        # Early stopping ONLY if VERY high confidence (>70% instead of >50%)
        # AND not a FEVER-style question
        if not is_fever_style and self.reasoning.has_consensus(answers, threshold=0.7):
            consensus_answer = max(set(answers), key=answers.count)
            
            # Quick Win 3: Validate consensus before early stopping
            if self.reasoning.validate_consensus_answer(question, consensus_answer):
                logger.info("Early stopping: validated consensus reached")
                return {
                    "answer": consensus_answer,
                    "rationales": rationales,
                    "stage": "consensus_validated",
                    "confidence": "high",
                    "models_used": {
                        "reasoning": "Llama 3 70B",
                        "query_generation": "None (early stop)",
                        "consolidation": "None (early stop)"
                    }
                }
            else:
                logger.info("Consensus validation failed - proceeding to full pipeline")
        
        # Stage 2: Dynamic Knowledge Adapting (Llama for queries, Gemini for correction)
        # Progressive correction: use preceding corrected rationales for subsequent ones
        logger.info("Stage 2: Dynamic Knowledge Adapting (Progressive Correction)")
        corrected_rationales = []
        
        for i, rationale in enumerate(rationales):
            # For progressive correction: build context from previous corrected rationales
            context = ""
            if corrected_rationales:
                context = "Previous corrected reasoning steps:\n"
                for j, prev_corrected in enumerate(corrected_rationales):
                    context += f"{j+1}. {prev_corrected}\n"
            
            knowledge_found = False
            for domain in domains:
                try:
                    # Generate query from current rationale (with context awareness)
                    query, query_type = self.query_generator.generate_query(rationale, domain)
                    knowledge = self.query_generator.execute_query(query, query_type, domain)
                    
                    if knowledge and knowledge != "No results found" and len(knowledge.strip()) > 10:
                        # Correct rationale using knowledge and context (progressive correction)
                        try:
                            corrected = self.corrector.correct_rationale(rationale, knowledge, context)
                            if corrected and len(corrected.strip()) > 0:
                                corrected_rationales.append(corrected)
                                knowledge_found = True
                                logger.debug(f"Corrected rationale {i+1}/{len(rationales)} using {domain} knowledge")
                                break
                        except Exception as e:
                            logger.warning(f"Rationale correction failed: {str(e)}")
                            # Continue to next domain
                            continue
                except Exception as e:
                    logger.warning(f"Query processing failed for domain {domain}: {str(e)}")
                    continue
            
            if not knowledge_found:
                # If no domain worked, use original rationale
                corrected_rationales.append(rationale)
                logger.debug(f"Using original rationale {i+1}/{len(rationales)} (no knowledge found)")
        
        # Stage 3: Answer Consolidation
        logger.info("Stage 3: Answer Consolidation")
        final_answer = self.consolidation.consolidate(question, corrected_rationales)
        
        return {
            "answer": final_answer,
            "initial_rationales": rationales,
            "corrected_rationales": corrected_rationales,
            "domains": domains,
            "stage": "full_pipeline",
            "confidence": "medium",
            "models_used": {
                "reasoning": "Llama 3 70B",
                "query_generation": "Llama 3 70B",
                "consolidation": "Llama 3 70B"
            }
        }
    
    def _format_fever_answer(self, answer: str) -> str:
        """Format answer as FEVER label if needed."""
        answer_upper = answer.upper()
        if 'SUPPORTS' in answer_upper or 'SUPPORT' in answer_upper:
            return 'SUPPORTS'
        elif 'REFUTES' in answer_upper or 'REFUTE' in answer_upper:
            return 'REFUTES'
        elif 'NOT ENOUGH INFO' in answer_upper or 'NOT ENOUGH' in answer_upper:
            return 'NOT ENOUGH INFO'
        return answer

