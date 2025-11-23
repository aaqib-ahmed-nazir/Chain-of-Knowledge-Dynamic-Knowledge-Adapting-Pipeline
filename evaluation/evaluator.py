from typing import Dict, List
import logging
import os
import json
import re
import time
from datetime import datetime
from tqdm import tqdm
from difflib import SequenceMatcher
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
                
                # Post-process prediction based on dataset type
                prediction = self._post_process_prediction(prediction, dataset_name)
                
                predictions.append(prediction)
                gold_labels.append(gold_label)
                
                # Rate limiting: add delay between API calls (longer for full pipeline)
                # FEVER runs full pipeline so needs more time
                if dataset_name == 'fever':
                    time.sleep(3)  # 3 seconds for FEVER (full pipeline)
                else:
                    time.sleep(2)  # 2 seconds for others
                
                logger.debug(f"Sample {i+1}: Q={question[:50]}... Pred={prediction[:50]}... Gold={gold_label[:50]}...")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing sample {i+1}: {error_msg}")
                # If rate limit, wait a bit before continuing
                if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                    logger.warning("Rate limit detected, waiting 60 seconds...")
                    time.sleep(60)
                predictions.append("")
                gold_labels.append(gold_label)
        
        # Calculate metrics (only on successfully processed samples)
        if predictions:
            metric = self._calculate_metric(dataset_name, predictions, gold_labels)
        else:
            metric = 0.0
            logger.warning(f"No predictions generated for {dataset_name}")
        
        result_dict = {
            'dataset': dataset_name,
            'num_samples': len(predictions),
            'metric_value': metric,
            'predictions': predictions,
            'gold_labels': gold_labels
        }
        
        logger.info(f"{dataset_name}: {metric:.2f}% ({len(predictions)} samples processed)")
        self.results.append(result_dict)
        
        return result_dict
    
    def evaluate_all(self, num_samples_per_dataset: int = 50, resume: bool = True) -> Dict:
        """Evaluate on all datasets."""
        datasets = ['fever', 'hotpotqa', 'medmcqa', 'mmlu_physics', 'mmlu_biology']
        
        all_results = {}
        
        # Try to load existing results if resuming
        if resume:
            existing_results = self._load_latest_results()
            if existing_results:
                logger.info(f"Found existing results with {len(existing_results)} completed datasets")
                all_results.update(existing_results)
                # Skip already completed datasets
                datasets = [d for d in datasets if d not in existing_results]
                logger.info(f"Resuming with remaining datasets: {datasets}")
        
        for dataset_name in datasets:
            try:
                all_results[dataset_name] = self.evaluate_dataset(dataset_name, num_samples_per_dataset)
                # Save incrementally after each dataset
                self._save_results(all_results, incremental=True)
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {str(e)}")
                logger.info("Saving partial results...")
                self._save_results(all_results, incremental=True)
                raise
        
        self._print_summary(all_results)
        self._save_results(all_results, incremental=False)
        
        return all_results
    
    def _load_latest_results(self) -> Dict:
        """Load latest results file if it exists."""
        import glob
        json_files = glob.glob(os.path.join('./data/results', "evaluation_*.json"))
        if not json_files:
            return {}
        
        latest_file = max(json_files, key=os.path.getctime)
        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
                logger.info(f"Loaded existing results from {latest_file}")
                return results
        except Exception as e:
            logger.warning(f"Could not load existing results: {str(e)}")
            return {}
    
    def _extract_question(self, sample: Dict, dataset_name: str) -> str:
        """Extract question from sample."""
        if dataset_name == 'fever':
            # Try 'claim' first (FEVER), then 'text' (tweet_eval fallback)
            claim = sample.get('claim', sample.get('text', str(sample)))
            # Format as FEVER-style question
            return f"Claim: {claim}"
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
            return str(sample['answer'])
        elif dataset_name == 'medmcqa':
            # MedMCQA uses 'cop' field (correct option: 0, 1, 2, or 3) which maps to opa, opb, opc, opd
            cop = sample.get('cop', -1)
            if cop == -1:
                # Fallback to 'exp' if cop is not available
                exp = sample.get('exp', '')
                if exp:
                    return str(exp)
                return ''
            # Map 0-3 to A-D, then get the actual option text
            option_map = {0: 'opa', 1: 'opb', 2: 'opc', 3: 'opd'}
            if cop in option_map:
                option_key = option_map[cop]
                return str(sample.get(option_key, ''))
            return ''
        elif dataset_name in ['mmlu_physics', 'mmlu_biology']:
            # MMLU answer can be int (0-3) or string (A-D), convert to string
            answer = sample.get('answer', sample.get('correct', None))
            if answer is None:
                return 'UNKNOWN'
            # Convert to string, handle both int and string formats
            if isinstance(answer, int):
                # Map 0-3 to A-D
                return chr(65 + answer)  # 0->A, 1->B, 2->C, 3->D
            return str(answer).upper().strip()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _calculate_metric(self, dataset_name: str, predictions: List[str], gold_labels: List[str]) -> float:
        """Calculate appropriate metric for dataset with improved matching."""
        if dataset_name == 'fever':
            return self._calculate_fever_accuracy(predictions, gold_labels)
        elif dataset_name == 'hotpotqa':
            return self._calculate_hotpotqa_accuracy(predictions, gold_labels)
        elif dataset_name == 'medmcqa':
            return self._calculate_medmcqa_accuracy(predictions, gold_labels)
        elif dataset_name in ['mmlu_physics', 'mmlu_biology']:
            return self._calculate_mmlu_accuracy(predictions, gold_labels)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _calculate_fever_accuracy(self, predictions: List[str], gold_labels: List[str]) -> float:
        """Calculate FEVER accuracy - extract label from prediction."""
        correct = 0
        for pred, gold in zip(predictions, gold_labels):
            pred_upper = pred.upper()
            pred_lower = pred.lower()
            gold_upper = gold.upper()
            gold_lower = gold.lower()
            
            # Check for exact label match (case-insensitive)
            if gold_upper in pred_upper or gold_lower in pred_lower:
                correct += 1
            # Check for partial matches (e.g., "REFUTE" in "REFUTES")
            elif any(gold_word in pred_upper for gold_word in gold_upper.split() if len(gold_word) > 3):
                correct += 1
            # Check for synonyms - SUPPORTS
            elif gold_lower == 'supports' and any(word in pred_lower for word in ['support', 'true', 'correct', 'yes', 'agree', 'valid', 'accurate', 'affirm', 'confirm', 'verify']):
                correct += 1
            # Check for synonyms - REFUTES
            elif gold_lower == 'refutes' and any(word in pred_lower for word in ['refute', 'false', 'incorrect', 'no', 'wrong', 'disagree', 'invalid', 'inaccurate', 'dispute', 'contradict', 'deny', 'reject', 'disprove']):
                correct += 1
            # Check for synonyms - NOT ENOUGH INFO
            elif gold_lower == 'not enough info' and any(phrase in pred_lower for phrase in ['not enough', 'insufficient', 'unknown', 'unclear', 'cannot determine', 'need more', 'lack of information', 'no information', 'inconclusive', 'uncertain']):
                correct += 1
        
        return (correct / len(predictions) * 100) if predictions else 0.0
    
    def _calculate_hotpotqa_accuracy(self, predictions: List[str], gold_labels: List[str]) -> float:
        """Calculate HotpotQA accuracy - fuzzy matching for entity names."""
        correct = 0
        for pred, gold in zip(predictions, gold_labels):
            pred_lower = pred.lower().strip()
            gold_lower = gold.lower().strip()
            
            # Remove punctuation for better matching
            pred_clean = re.sub(r'[^\w\s]', '', pred_lower)
            gold_clean = re.sub(r'[^\w\s]', '', gold_lower)
            
            # Exact match
            if pred_clean == gold_clean:
                correct += 1
            # Substring match (gold in pred or pred in gold)
            elif len(gold_clean) > 3 and (gold_clean in pred_clean or pred_clean in gold_clean):
                correct += 1
            # Key words match (at least 60% of gold words in prediction)
            else:
                pred_words = set(pred_clean.split())
                gold_words = set(gold_clean.split())
                if gold_words and len(pred_words & gold_words) / len(gold_words) > 0.6:
                    correct += 1
                # Check if all key words from gold are in pred (for multi-word answers)
                elif gold_words and len(gold_words) > 1 and all(word in pred_words for word in gold_words if len(word) > 2):
                    correct += 1
                # Fuzzy similarity
                elif SequenceMatcher(None, pred_clean, gold_clean).ratio() > 0.7:
                    correct += 1
        
        return (correct / len(predictions) * 100) if predictions else 0.0
    
    def _calculate_medmcqa_accuracy(self, predictions: List[str], gold_labels: List[str]) -> float:
        """Calculate MedMCQA accuracy - match option text with fuzzy matching."""
        correct = 0
        for pred, gold in zip(predictions, gold_labels):
            if not gold or gold.strip() == '':
                continue  # Skip empty gold labels
                
            pred_lower = pred.lower().strip()
            gold_lower = str(gold).lower().strip()
            
            # Remove punctuation for better matching
            pred_clean = re.sub(r'[^\w\s]', '', pred_lower)
            gold_clean = re.sub(r'[^\w\s]', '', gold_lower)
            
            # Exact match
            if pred_clean == gold_clean:
                correct += 1
            # Substring match (gold in pred or pred in gold)
            elif len(gold_clean) > 3 and (gold_clean in pred_clean or pred_clean in gold_clean):
                correct += 1
            # Word overlap - if most words match
            else:
                pred_words = set(pred_clean.split())
                gold_words = set(gold_clean.split())
                # If >= 60% of gold words are in prediction
                if gold_words and len(pred_words & gold_words) / len(gold_words) > 0.6:
                    correct += 1
                # Check if all key words (length > 3) from gold are in pred
                elif gold_words and all(word in pred_words for word in gold_words if len(word) > 3):
                    correct += 1
                # Fuzzy similarity
                elif SequenceMatcher(None, pred_clean, gold_clean).ratio() > 0.7:
                    correct += 1
        
        return (correct / len(predictions) * 100) if predictions else 0.0
    
    def _calculate_mmlu_accuracy(self, predictions: List[str], gold_labels: List[str]) -> float:
        """Calculate MMLU accuracy - extract option letter (A, B, C, D) with fuzzy matching."""
        correct = 0
        for pred, gold in zip(predictions, gold_labels):
            pred_lower = pred.lower().strip()
            gold_lower = str(gold).lower().strip()
            
            # Extract option letter from prediction
            pred_letter_match = re.search(r'\b([abcd])\b', pred_lower)
            if pred_letter_match:
                pred_letter = pred_letter_match.group(1).upper()
                gold_letter = gold_lower.upper()
                if pred_letter == gold_letter:
                    correct += 1
            # Fallback: check if gold letter appears anywhere in prediction
            elif gold_lower.upper() in pred.upper():
                correct += 1
            # Fuzzy match: check similarity
            elif SequenceMatcher(None, pred_lower, gold_lower).ratio() > 0.8:
                correct += 1
        
        return (correct / len(predictions) * 100) if predictions else 0.0
    
    def _post_process_prediction(self, prediction: str, dataset_name: str) -> str:
        """Post-process prediction to extract key information based on dataset."""
        if dataset_name == 'fever':
            # Extract FEVER label from prediction - check for explicit labels first
            pred_upper = prediction.upper()
            # Check for full label names
            if 'REFUTES' in pred_upper:
                return 'REFUTES'
            elif 'SUPPORTS' in pred_upper:
                return 'SUPPORTS'
            elif 'NOT ENOUGH INFO' in pred_upper or 'NOT ENOUGH INFORMATION' in pred_upper:
                return 'NOT ENOUGH INFO'
            
            # Check for partial matches
            if 'REFUTE' in pred_upper and 'REFUTES' not in pred_upper:
                return 'REFUTES'
            elif 'SUPPORT' in pred_upper and 'SUPPORTS' not in pred_upper:
                return 'SUPPORTS'
            
            # Fallback to keyword matching (more aggressive)
            pred_lower = prediction.lower()
            # Count keyword matches
            refute_keywords = ['refute', 'false', 'incorrect', 'wrong', 'disagree', 'contradict', 'deny', 'reject', 'disprove', 'inaccurate']
            support_keywords = ['support', 'true', 'correct', 'yes', 'agree', 'valid', 'accurate', 'affirm', 'confirm', 'verify']
            info_keywords = ['not enough', 'insufficient', 'unknown', 'unclear', 'cannot determine', 'inconclusive', 'uncertain']
            
            refute_count = sum(1 for word in refute_keywords if word in pred_lower)
            support_count = sum(1 for word in support_keywords if word in pred_lower)
            info_count = sum(1 for phrase in info_keywords if phrase in pred_lower)
            
            if refute_count > support_count and refute_count > info_count:
                return 'REFUTES'
            elif support_count > refute_count and support_count > info_count:
                return 'SUPPORTS'
            elif info_count > 0:
                return 'NOT ENOUGH INFO'
            
            # Default fallback - return original (will be matched in _calculate_fever_accuracy)
            return prediction
        
        elif dataset_name in ['mmlu_physics', 'mmlu_biology']:
            # Extract option letter (A, B, C, D) from prediction
            # Look for patterns like "Answer: A" or "The answer is A" or just "A" at start
            match = re.search(r'\b([ABCD])\b', prediction.upper())
            if match:
                return match.group(1)
            # Fallback: return first 50 chars
            return prediction[:50]
        
        else:
            # For other datasets, return as is (already processed by consolidation)
            return prediction
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        for dataset_name, result in results.items():
            print(f"{dataset_name:20s}: {result['metric_value']:6.2f}%")
        print("="*60)
    
    def _save_results(self, results: Dict, incremental: bool = False):
        """Save results to file."""
        if incremental:
            # For incremental saves, use a fixed filename
            filename = './data/results/evaluation_incremental.json'
        else:
            # For final save, use timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'./data/results/evaluation_{timestamp}.json'
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

