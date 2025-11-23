import logging
import os
from config.settings import config

def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup logger with file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Create directories
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    
    # File handler
    fh = logging.FileHandler(config.LOG_FILE)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

