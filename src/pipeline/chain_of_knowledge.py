from typing import Dict, List
import logging
from config.settings import config
from src.core.reasoning import ReasoningPreparation
from src.core.query_generator import AdaptiveQueryGenerator
from src.core.rationale_corrector import RationaleCorrector
from src.core.consolidation import AnswerConsolidation

logger = logging.getLogger(__name__)

class ChainOfKnowledge:
    """Main CoK pipeline orchestrator."""
    
    def __init__(self, gemini_client, groq_client, knowledge_sources, use_llama_for_reasoning=True):
        # Use Llama for reasoning (faster, no safety filters) or Gemini (original)
        reasoning_client = groq_client if use_llama_for_reasoning else gemini_client
        self.reasoning = ReasoningPreparation(reasoning_client)
        self.query_generator = AdaptiveQueryGenerator(groq_client, knowledge_sources)
        self.corrector = RationaleCorrector(gemini_client)
        self.consolidation = AnswerConsolidation(gemini_client)
        self.reasoning_model = "Llama" if use_llama_for_reasoning else "Gemini"
        logger.info(f"CoK pipeline initialized (reasoning: {self.reasoning_model})")
    
    def run(self, question: str) -> Dict:
        """Execute full CoK pipeline."""
        logger.info(f"Processing question: {question[:100]}...")
        
        # Stage 1: Reasoning Preparation (Llama or Gemini)
        rationales = self.reasoning.generate_rationales(question)
        answers = self.reasoning.generate_answers(question, rationales)
        domains = self.reasoning.identify_domains(question)
        
        # Early stopping if consensus
        if self.reasoning.has_consensus(answers, config.CONSENSUS_THRESHOLD):
            consensus_answer = max(set(answers), key=answers.count)
            logger.info("Early stopping: consensus reached")
            return {
                "answer": consensus_answer,
                "rationales": rationales,
                "stage": "consensus",
                "confidence": "high",
                "models_used": {
                    "reasoning": self.reasoning_model,
                    "query_generation": "None (early stop)",
                    "consolidation": "None (early stop)"
                }
            }
        
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
            
            for domain in domains:
                try:
                    # Generate query from current rationale (with context awareness)
                    query, query_type = self.query_generator.generate_query(rationale, domain)
                    knowledge = self.query_generator.execute_query(query, query_type, domain)
                    
                    if knowledge != "No results found":
                        # Correct rationale using knowledge and context (progressive correction)
                        corrected = self.corrector.correct_rationale(rationale, knowledge, context)
                        corrected_rationales.append(corrected)
                        logger.debug(f"Corrected rationale {i+1}/{len(rationales)} using {domain} knowledge")
                        break
                except Exception as e:
                    logger.warning(f"Query processing failed: {str(e)}")
                    continue
            else:
                # If no domain worked, use original rationale
                corrected_rationales.append(rationale)
                logger.debug(f"Using original rationale {i+1}/{len(rationales)} (no knowledge found)")
        
        # Stage 3: Answer Consolidation (Gemini)
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
                "reasoning": self.reasoning_model,
                "query_generation": "Llama",
                "consolidation": "Gemini"
            }
        }

