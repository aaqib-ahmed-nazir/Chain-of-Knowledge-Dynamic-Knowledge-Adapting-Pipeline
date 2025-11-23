from typing import List, Union

def accuracy(predictions: List[str], gold_labels: List[str]) -> float:
    """Calculate accuracy metric (case-insensitive string comparison)."""
    if len(predictions) != len(gold_labels):
        raise ValueError("Predictions and gold labels must have same length")
    
    correct = sum(p.lower().strip() == g.lower().strip() for p, g in zip(predictions, gold_labels))
    return (correct / len(predictions)) * 100


def exact_match(predictions: List[str], gold_answers_list: List[List[str]]) -> float:
    """Calculate exact match metric for QA tasks.
    
    Args:
        predictions: List of predicted answers
        gold_answers_list: List of lists of acceptable gold answers
    
    Returns:
        Exact match percentage
    """
    if len(predictions) != len(gold_answers_list):
        raise ValueError("Predictions and gold answers must have same length")
    
    correct = 0
    for pred, gold_list in zip(predictions, gold_answers_list):
        pred_lower = pred.lower().strip()
        # Check if prediction matches any gold answer (substring match)
        if any(pred_lower in g.lower() or g.lower() in pred_lower for g in gold_list):
            correct += 1
    
    return (correct / len(predictions)) * 100

