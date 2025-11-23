"""Resume evaluation from where it left off."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from config.settings import config
from src.utils.logger import setup_logger
from src.models.llm_client import LLMFactory
from src.knowledge.wikipedia_retriever import WikipediaRetriever
from src.pipeline.chain_of_knowledge import ChainOfKnowledge
from evaluation.benchmark_datasets import DatasetManager
from evaluation.evaluator import CoKEvaluator

# Setup logging
logger = setup_logger(__name__)

def main():
    logger.info("Resuming CoK evaluation")
    
    # Initialize LLM client - using only Llama 3.3 70B
    llm_client = LLMFactory.create_groq_client(
        config.GROQ_API_KEY,
        config.LLM_MODEL
    )
    
    # Initialize knowledge sources
    knowledge_sources = {'wikipedia': WikipediaRetriever()}
    
    # Initialize CoK pipeline - uses Llama for all stages
    cok = ChainOfKnowledge(llm_client, knowledge_sources)
    
    # Initialize evaluator
    dataset_manager = DatasetManager()
    evaluator = CoKEvaluator(cok, dataset_manager)
    
    # Resume evaluation (will skip completed datasets)
    results = evaluator.evaluate_all(num_samples_per_dataset=50, resume=True)
    
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()

