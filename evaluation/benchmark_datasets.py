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
            # Try loading FEVER from different sources
            try:
                dataset = load_dataset('fever', cache_dir=self.cache_dir)
                logger.info("Loaded FEVER dataset")
            except Exception as e1:
                logger.warning(f"FEVER dataset not available ({str(e1)}), trying alternative")
                try:
                    # Use a fact-checking dataset as substitute
                    dataset = load_dataset('tweet_eval', 'stance_climate', cache_dir=self.cache_dir)
                    logger.info("Using tweet_eval/stance_climate as FEVER substitute")
                except Exception as e2:
                    logger.error(f"Alternative dataset also failed: {str(e2)}")
                    raise
            
            available_splits = list(dataset.keys())
            if split not in available_splits:
                split = available_splits[0] if available_splits else 'validation'
            max_samples = min(num_samples, len(dataset[split]))
            samples = dataset[split].shuffle(seed=42).select(range(max_samples))
            logger.info(f"Loaded {len(samples)} samples from dataset {split} split")
            return samples
        except Exception as e:
            logger.error(f"Failed to load FEVER dataset: {str(e)}")
            raise
    
    def load_hotpotqa(self, split: str = 'validation', num_samples: int = 50):
        """Load HotpotQA dataset."""
        try:
            dataset = load_dataset('hotpot_qa', 'distractor', cache_dir=self.cache_dir)
            available_splits = list(dataset.keys())
            if split not in available_splits:
                split = available_splits[0] if available_splits else 'validation'
            samples = dataset[split].shuffle(seed=42).select(range(min(num_samples, len(dataset[split]))))
            logger.info(f"Loaded {len(samples)} samples from HotpotQA {split}")
            return samples
        except Exception as e:
            logger.error(f"Failed to load HotpotQA dataset: {str(e)}")
            raise
    
    def load_medmcqa(self, split: str = 'validation', num_samples: int = 50):
        """Load MedMCQA dataset. Use validation split as test split doesn't have labels."""
        try:
            dataset = load_dataset('medmcqa', cache_dir=self.cache_dir)
            available_splits = list(dataset.keys())
            # Prefer validation over test (test split has cop=-1, no labels)
            if 'validation' in available_splits:
                split = 'validation'
            elif split not in available_splits:
                split = available_splits[0] if available_splits else 'validation'
            samples = dataset[split].shuffle(seed=42).select(range(min(num_samples, len(dataset[split]))))
            logger.info(f"Loaded {len(samples)} samples from MedMCQA {split}")
            return samples
        except Exception as e:
            logger.error(f"Failed to load MedMCQA dataset: {str(e)}")
            raise
    
    def load_mmlu_physics(self, num_samples: int = 30):
        """Load MMLU Physics subset."""
        try:
            # Try college_physics first, fallback to high_school_physics
            try:
                dataset = load_dataset('cais/mmlu', 'college_physics', cache_dir=self.cache_dir)
                logger.info("Using MMLU college_physics")
            except:
                dataset = load_dataset('cais/mmlu', 'high_school_physics', cache_dir=self.cache_dir)
                logger.info("Using MMLU high_school_physics")
            
            available_splits = list(dataset.keys())
            split = 'test' if 'test' in available_splits else available_splits[0]
            samples = dataset[split].shuffle(seed=42).select(range(min(num_samples, len(dataset[split]))))
            logger.info(f"Loaded {len(samples)} samples from MMLU Physics")
            return samples
        except Exception as e:
            logger.error(f"Failed to load MMLU Physics dataset: {str(e)}")
            raise
    
    def load_mmlu_biology(self, num_samples: int = 30):
        """Load MMLU Biology subset."""
        try:
            # Try college_biology first, fallback to high_school_biology
            try:
                dataset = load_dataset('cais/mmlu', 'college_biology', cache_dir=self.cache_dir)
                logger.info("Using MMLU college_biology")
            except:
                dataset = load_dataset('cais/mmlu', 'high_school_biology', cache_dir=self.cache_dir)
                logger.info("Using MMLU high_school_biology")
            
            available_splits = list(dataset.keys())
            split = 'test' if 'test' in available_splits else available_splits[0]
            samples = dataset[split].shuffle(seed=42).select(range(min(num_samples, len(dataset[split]))))
            logger.info(f"Loaded {len(samples)} samples from MMLU Biology")
            return samples
        except Exception as e:
            logger.error(f"Failed to load MMLU Biology dataset: {str(e)}")
            raise

