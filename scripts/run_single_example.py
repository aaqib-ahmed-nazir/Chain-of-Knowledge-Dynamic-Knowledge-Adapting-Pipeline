import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from config.settings import config
from src.utils.logger import setup_logger
from src.models.llm_client import LLMFactory
from src.knowledge.wikipedia_retriever import WikipediaRetriever
from src.pipeline.chain_of_knowledge import ChainOfKnowledge

# Setup logging
logger = setup_logger(__name__)

def main():
    logger.info("Running smoke test on single examples")
    
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
    
    # Test questions
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the cure for pneumonia?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        try:
            result = cok.run(question)
            print(f"Answer: {result['answer']}")
            print(f"Stage: {result['stage']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Domains: {result.get('domains', 'N/A')}")
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print(f"Error: {str(e)}")
        
        print('-'*60)

if __name__ == "__main__":
    main()

