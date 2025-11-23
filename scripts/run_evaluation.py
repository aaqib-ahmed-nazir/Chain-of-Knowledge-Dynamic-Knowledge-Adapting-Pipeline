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
    logger.info("Initializing CoK evaluation")
    
    # Initialize LLM clients
    gemini_client = LLMFactory.create_gemini_client(
        config.GEMINI_API_KEY,
        config.GEMINI_MODEL
    )
    groq_client = LLMFactory.create_groq_client(
        config.GROQ_API_KEY,
        config.GROQ_MODEL
    )
    
    # Initialize knowledge sources
    knowledge_sources = {'wikipedia': WikipediaRetriever()}
    
    # Initialize CoK pipeline
    cok = ChainOfKnowledge(gemini_client, groq_client, knowledge_sources)
    
    # Initialize evaluator
    dataset_manager = DatasetManager()
    evaluator = CoKEvaluator(cok, dataset_manager)
    
    # Run evaluation on all datasets
    results = evaluator.evaluate_all(num_samples_per_dataset=50)
    
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()

