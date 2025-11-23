import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys - Using Together AI
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    
    # Models - Llama 3 70B Chat
    TOGETHER_MODEL = "meta-llama/Llama-3-70b-chat-hf"
    # Alternative models available:
    # - "meta-llama/Llama-3-8b-chat-hf" (faster)
    # - "meta-llama/Llama-2-70b-chat-hf" (older but stable)
    
    # LLM Parameters
    REASONING_TEMPERATURE = 0.7
    QUERY_TEMPERATURE = 0.0
    CONSOLIDATION_TEMPERATURE = 0.0
    MAX_TOKENS = 1024
    
    # Pipeline Parameters
    NUM_RATIONALES = 5
    CONSENSUS_THRESHOLD = 0.5
    EARLY_STOPPING_ENABLED = True
    # Reduce rationales for FEVER to save tokens (full pipeline uses more)
    NUM_RATIONALES_FEVER = 3  # Use fewer for FEVER since it always runs full pipeline
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/cok.log"
    
    # Data
    DATASET_CACHE_DIR = "./data/datasets"
    RESULTS_DIR = "./data/results"

config = Config()
