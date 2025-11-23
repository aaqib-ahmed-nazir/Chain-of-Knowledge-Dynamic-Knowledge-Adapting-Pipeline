import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from config.settings import config
from src.utils.logger import setup_logger
from src.models.llm_client import LLMFactory
from src.knowledge.wikipedia_retriever import WikipediaRetriever
from src.knowledge.wikidata_retriever import WikidataSPARQLRetriever
from src.pipeline.chain_of_knowledge import ChainOfKnowledge
from evaluation.benchmark_datasets import DatasetManager
from evaluation.evaluator import CoKEvaluator

# Setup logging
logger = setup_logger(__name__)

def main():
    logger.info("Initializing CoK evaluation with Together AI (Llama 3 70B)")
    
    # Verify API key
    if not config.TOGETHER_API_KEY:
        logger.error("TOGETHER_API_KEY not set. Export it: export TOGETHER_API_KEY='your_key'")
        sys.exit(1)
    
    # Initialize LLM client - using Together AI
    llm_client = LLMFactory.create_together_client(
        config.TOGETHER_API_KEY,
        config.TOGETHER_MODEL
    )
    
    # Initialize knowledge sources
    knowledge_sources = {
        'wikipedia': WikipediaRetriever(),
        'wikidata_sparql': WikidataSPARQLRetriever()
    }
    
    # Initialize CoK pipeline
    cok = ChainOfKnowledge(llm_client, knowledge_sources)
    
    # Initialize evaluator
    dataset_manager = DatasetManager()
    evaluator = CoKEvaluator(cok, dataset_manager)
    
    # Run evaluation on all datasets
    # Set resume=False to force re-run, or True to skip completed datasets
    results = evaluator.evaluate_all(num_samples_per_dataset=50, resume=False)
    
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()
