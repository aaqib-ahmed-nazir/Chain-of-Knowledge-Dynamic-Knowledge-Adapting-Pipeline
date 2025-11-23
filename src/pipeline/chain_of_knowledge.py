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
    
    def __init__(self, gemini_client, groq_client, knowledge_sources):
        self.reasoning = ReasoningPreparation(gemini_client)
        self.query_generator = AdaptiveQueryGenerator(groq_client, knowledge_sources)
        self.corrector = RationaleCorrector(gemini_client)
        self.consolidation = AnswerConsolidation(gemini_client)
        logger.info("CoK pipeline initialized")
    
    def run(self, question: str) -> Dict:
        """Execute full CoK pipeline."""
        logger.info(f"Processing question: {question[:100]}...")
        
        # Stage 1: Reasoning Preparation
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
                "confidence": "high"
            }
        
        # Stage 2: Dynamic Knowledge Adapting
        logger.info("Stage 2: Dynamic Knowledge Adapting")
        corrected_rationales = []
        
        for i, rationale in enumerate(rationales):
            for domain in domains:
                try:
                    query, query_type = self.query_generator.generate_query(rationale, domain)
                    knowledge = self.query_generator.execute_query(query, query_type, domain)
                    
                    if knowledge != "No results found":
                        corrected = self.corrector.correct_rationale(rationale, knowledge)
                        corrected_rationales.append(corrected)
                        break
                except Exception as e:
                    logger.warning(f"Query processing failed: {str(e)}")
                    continue
            else:
                corrected_rationales.append(rationale)
        
        # Stage 3: Answer Consolidation
        logger.info("Stage 3: Answer Consolidation")
        final_answer = self.consolidation.consolidate(question, corrected_rationales)
        
        return {
            "answer": final_answer,
            "initial_rationales": rationales,
            "corrected_rationales": corrected_rationales,
            "domains": domains,
            "stage": "full_pipeline",
            "confidence": "medium"
        }

