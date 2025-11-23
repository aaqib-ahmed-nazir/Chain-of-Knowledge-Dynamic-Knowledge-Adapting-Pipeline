import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Models
    GROQ_MODEL = "llama-3.1-70b-versatile"
    GEMINI_MODEL = "gemini-2.5-flash"
    
    # LLM Parameters
    REASONING_TEMPERATURE = 0.7
    QUERY_TEMPERATURE = 0.0
    CONSOLIDATION_TEMPERATURE = 0.0
    MAX_TOKENS = 1024
    
    # Pipeline Parameters
    NUM_RATIONALES = 5
    CONSENSUS_THRESHOLD = 0.5
    EARLY_STOPPING_ENABLED = True
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/cok.log"
    
    # Data
    DATASET_CACHE_DIR = "./data/datasets"
    RESULTS_DIR = "./data/results"

config = Config()

