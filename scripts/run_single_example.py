import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from config.settings import config
from src.utils.logger import setup_logger
from src.models.llm_client import LLMFactory
from src.knowledge.wikipedia_retriever import WikipediaRetriever
from src.knowledge.wikidata_retriever import WikidataSPARQLRetriever
from src.knowledge.duckduckgo_retriever import DuckDuckGoRetriever
from src.pipeline.chain_of_knowledge import ChainOfKnowledge

# Setup logging
logger = setup_logger(__name__)

def main():
    logger.info("Running smoke test on single examples")
    
    # Verify API key
    if not config.TOGETHER_API_KEY:
        logger.error("TOGETHER_API_KEY not set. Export it: export TOGETHER_API_KEY='your_key'")
        sys.exit(1)
    
    # Initialize LLM client - using Together AI
    llm_client = LLMFactory.create_together_client(
        config.TOGETHER_API_KEY,
        config.TOGETHER_MODEL
    )
    
    # Initialize knowledge sources (DuckDuckGo as fallback)
    knowledge_sources = {
        'wikipedia': WikipediaRetriever(),
        'wikidata_sparql': WikidataSPARQLRetriever(),
        'duckduckgo': DuckDuckGoRetriever(timeout=5)
    }
    
    # Initialize CoK pipeline
    cok = ChainOfKnowledge(llm_client, knowledge_sources)
    
    # Test questions - mix of simple (consensus) and complex (full pipeline)
    questions = [
        "What is the capital of France?",  # Simple - should reach consensus
        "Who wrote Romeo and Juliet?",  # Simple - should reach consensus
        "What year was the Argentine actor who directed El Tio Disparate born?",  # Complex - should trigger full pipeline
        "Which two countries share the longest border in the world?",  # Complex - might trigger full pipeline
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        try:
            result = cok.run(question)
            
            # Clean answer (remove markdown formatting)
            answer = result['answer'].replace('**', '').strip()
            print(f"\nAnswer: {answer}")
            
            # Show models used
            models = result.get('models_used', {})
            print(f"\nModels Used:")
            for stage, model in models.items():
                print(f"  {stage.replace('_', ' ').title()}: {model}")
            
            print(f"\nStage: {result['stage']}")
            print(f"Confidence: {result['confidence']}")
            if result.get('domains'):
                print(f"Domains: {', '.join(result['domains'])}")
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print(f"Error: {str(e)}")
        
        print('-'*60)

if __name__ == "__main__":
    main()
