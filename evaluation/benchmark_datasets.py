from datasets import load_dataset
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manage loading of benchmark datasets."""
    
    def __init__(self, cache_dir: str = './data/datasets'):
        self.cache_dir = cache_dir
    
    def load_fever(self, split: str = 'validation', num_samples: int = 50):
        """Load FEVER dataset."""
        try:
            dataset = load_dataset('fever', 'paper_dev', cache_dir=self.cache_dir)
            samples = dataset[split].shuffle(seed=42).select(range(num_samples))
            logger.info(f"Loaded {len(samples)} samples from FEVER {split}")
            return samples
        except Exception as e:
            logger.error(f"Failed to load FEVER dataset: {str(e)}")
            raise
    
    def load_hotpotqa(self, split: str = 'validation', num_samples: int = 50):
        """Load HotpotQA dataset."""
        try:
            dataset = load_dataset('hotpot_qa', 'distractor', cache_dir=self.cache_dir)
            samples = dataset[split].shuffle(seed=42).select(range(num_samples))
            logger.info(f"Loaded {len(samples)} samples from HotpotQA {split}")
            return samples
        except Exception as e:
            logger.error(f"Failed to load HotpotQA dataset: {str(e)}")
            raise
    
    def load_medmcqa(self, split: str = 'test', num_samples: int = 50):
        """Load MedMCQA dataset."""
        try:
            dataset = load_dataset('medmcqa', cache_dir=self.cache_dir)
            samples = dataset[split].shuffle(seed=42).select(range(num_samples))
            logger.info(f"Loaded {len(samples)} samples from MedMCQA {split}")
            return samples
        except Exception as e:
            logger.error(f"Failed to load MedMCQA dataset: {str(e)}")
            raise
    
    def load_mmlu_physics(self, num_samples: int = 30):
        """Load MMLU Physics subset."""
        try:
            dataset = load_dataset('cais/mmlu', 'physics', cache_dir=self.cache_dir)
            samples = dataset['test'].shuffle(seed=42).select(range(num_samples))
            logger.info(f"Loaded {len(samples)} samples from MMLU Physics")
            return samples
        except Exception as e:
            logger.error(f"Failed to load MMLU Physics dataset: {str(e)}")
            raise
    
    def load_mmlu_biology(self, num_samples: int = 30):
        """Load MMLU Biology subset."""
        try:
            dataset = load_dataset('cais/mmlu', 'biology', cache_dir=self.cache_dir)
            samples = dataset['test'].shuffle(seed=42).select(range(num_samples))
            logger.info(f"Loaded {len(samples)} samples from MMLU Biology")
            return samples
        except Exception as e:
            logger.error(f"Failed to load MMLU Biology dataset: {str(e)}")
            raise

