from typing import Dict, List
import logging
import os
import json
from datetime import datetime
from tqdm import tqdm
from evaluation.benchmark_datasets import DatasetManager
from evaluation.metrics import accuracy, exact_match

logger = logging.getLogger(__name__)

class CoKEvaluator:
    """Evaluate Chain-of-Knowledge pipeline on benchmark datasets."""
    
    def __init__(self, cok_model, dataset_manager: DatasetManager):
        self.cok_model = cok_model
        self.dataset_manager = dataset_manager
        self.results = []
    
    def evaluate_dataset(self, dataset_name: str, num_samples: int = 50) -> Dict:
        """Evaluate CoK on specific dataset.
        
        Args:
            dataset_name: Name of dataset (fever, hotpotqa, medmcqa, mmlu_physics, mmlu_biology)
            num_samples: Number of samples to evaluate
        
        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_name} ({num_samples} samples)")
        
        # Load dataset
        if dataset_name == 'fever':
            samples = self.dataset_manager.load_fever(num_samples=num_samples)
        elif dataset_name == 'hotpotqa':
            samples = self.dataset_manager.load_hotpotqa(num_samples=num_samples)
        elif dataset_name == 'medmcqa':
            samples = self.dataset_manager.load_medmcqa(num_samples=num_samples)
        elif dataset_name == 'mmlu_physics':
            samples = self.dataset_manager.load_mmlu_physics(num_samples=num_samples)
        elif dataset_name == 'mmlu_biology':
            samples = self.dataset_manager.load_mmlu_biology(num_samples=num_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        predictions = []
        gold_labels = []
        
        # Run inference
        for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {dataset_name}")):
            question = self._extract_question(sample, dataset_name)
            gold_label = self._extract_gold_label(sample, dataset_name)
            
            try:
                # Run CoK
                result = self.cok_model.run(question)
                prediction = result['answer']
                
                predictions.append(prediction)
                gold_labels.append(gold_label)
                
                logger.debug(f"Sample {i+1}: Q={question[:50]}... Pred={prediction[:30]}... Gold={gold_label[:30]}...")
            except Exception as e:
                logger.error(f"Error processing sample {i+1}: {str(e)}")
                predictions.append("")
                gold_labels.append(gold_label)
        
        # Calculate metrics
        metric = self._calculate_metric(dataset_name, predictions, gold_labels)
        
        result_dict = {
            'dataset': dataset_name,
            'num_samples': num_samples,
            'metric_value': metric,
            'predictions': predictions,
            'gold_labels': gold_labels
        }
        
        logger.info(f"{dataset_name}: {metric:.2f}%")
        self.results.append(result_dict)
        
        return result_dict
    
    def evaluate_all(self, num_samples_per_dataset: int = 50) -> Dict:
        """Evaluate on all datasets."""
        datasets = ['fever', 'hotpotqa', 'medmcqa', 'mmlu_physics', 'mmlu_biology']
        
        all_results = {}
        for dataset_name in datasets:
            all_results[dataset_name] = self.evaluate_dataset(dataset_name, num_samples_per_dataset)
        
        self._print_summary(all_results)
        self._save_results(all_results)
        
        return all_results
    
    def _extract_question(self, sample: Dict, dataset_name: str) -> str:
        """Extract question from sample."""
        if dataset_name == 'fever':
            # Try 'claim' first (FEVER), then 'text' (tweet_eval fallback)
            return sample.get('claim', sample.get('text', str(sample)))
        elif dataset_name == 'hotpotqa':
            return sample['question']
        elif dataset_name == 'medmcqa':
            return sample['question']
        elif dataset_name in ['mmlu_physics', 'mmlu_biology']:
            return sample['question']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _extract_gold_label(self, sample: Dict, dataset_name: str) -> str:
        """Extract gold label from sample."""
        if dataset_name == 'fever':
            # Try 'label' field
            label = sample.get('label', None)
            if label is None:
                return 'UNKNOWN'
            # Convert to string and normalize
            label_str = str(label).upper()
            # Map tweet_eval labels (0=AGAINST, 1=FAVOR, 2=NONE) to FEVER format
            if label_str in ['0', 'AGAINST', 'REFUTES']:
                return 'REFUTES'
            elif label_str in ['1', 'FAVOR', 'SUPPORTS']:
                return 'SUPPORTS'
            elif label_str in ['2', 'NONE', 'NOT ENOUGH INFO']:
                return 'NOT ENOUGH INFO'
            # If already in FEVER format, return as is
            return label_str
        elif dataset_name == 'hotpotqa':
            return sample['answer']
        elif dataset_name == 'medmcqa':
            return str(sample['exp'])
        elif dataset_name in ['mmlu_physics', 'mmlu_biology']:
            return sample['answer']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _calculate_metric(self, dataset_name: str, predictions: List[str], gold_labels: List[str]) -> float:
        """Calculate appropriate metric for dataset."""
        if dataset_name == 'fever':
            return accuracy(predictions, gold_labels)
        elif dataset_name == 'hotpotqa':
            return exact_match(predictions, [[g] for g in gold_labels])
        else:
            return accuracy(predictions, gold_labels)
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        for dataset_name, result in results.items():
            print(f"{dataset_name:20s}: {result['metric_value']:6.2f}%")
        print("="*60)
    
    def _save_results(self, results: Dict):
        """Save results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'./data/results/evaluation_{timestamp}.json'
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

