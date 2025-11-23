"""Detailed analysis of evaluation results."""
import json
import os
from collections import Counter
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_results(results_file: str = "./data/results/evaluation_results.json") -> Dict:
    """Load evaluation results."""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_fever_confusion_matrix(results: Dict):
    """Create confusion matrix for FEVER dataset."""
    if 'fever' not in results:
        return
    
    fever_data = results['fever']
    predictions = fever_data['predictions']
    gold_labels = fever_data['gold_labels']
    
    labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    
    # Create confusion matrix
    confusion_matrix = np.zeros((3, 3), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    for pred, gold in zip(predictions, gold_labels):
        pred_idx = label_to_idx.get(pred.upper(), 0)
        gold_idx = label_to_idx.get(gold.upper(), 0)
        confusion_matrix[gold_idx][pred_idx] += 1
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, confusion_matrix[i, j],
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('FEVER Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('./data/results/fever_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("FEVER confusion matrix saved to: ./data/results/fever_confusion_matrix.png")
    plt.close()
    
    # Print per-class accuracy
    print("\nFEVER Per-Class Analysis:")
    print("="*60)
    for i, label in enumerate(labels):
        total = sum(confusion_matrix[i])
        correct = confusion_matrix[i][i]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{label:20s}: {correct:2d}/{total:2d} = {accuracy:5.2f}%")

def analyze_error_patterns(results: Dict):
    """Analyze error patterns across datasets."""
    print("\n" + "="*80)
    print("ERROR PATTERN ANALYSIS")
    print("="*80)
    
    for dataset_name, data in results.items():
        predictions = data['predictions']
        gold_labels = data['gold_labels']
        
        total = len(predictions)
        correct = sum(p.lower().strip() == g.lower().strip() 
                     for p, g in zip(predictions, gold_labels))
        errors = total - correct
        error_rate = (errors / total * 100) if total > 0 else 0
        
        print(f"\n{dataset_name.upper()}:")
        print(f"  Total Samples: {total}")
        print(f"  Correct: {correct} ({100-error_rate:.2f}%)")
        print(f"  Errors: {errors} ({error_rate:.2f}%)")
        
        # For multiple choice datasets, show distribution
        if dataset_name in ['medmcqa', 'mmlu_physics', 'mmlu_biology']:
            pred_dist = Counter([p.upper().strip() for p in predictions])
            gold_dist = Counter([g.upper().strip() for g in gold_labels])
            print(f"  Prediction Distribution: {dict(pred_dist)}")
            print(f"  Gold Distribution: {dict(gold_dist)}")

def create_comparison_chart(results: Dict):
    """Create a detailed comparison chart."""
    paper_baseline = {
        'fever': 63.4,
        'hotpotqa': 34.1,
        'medmcqa': 70.5,
        'mmlu_physics': 45.5,
        'mmlu_biology': 83.0
    }
    
    datasets = []
    our_scores = []
    paper_scores = []
    differences = []
    
    for dataset_name, result in results.items():
        datasets.append(dataset_name.replace('_', ' ').title())
        our_score = result['metric_value']
        paper_score = paper_baseline.get(dataset_name, 0)
        our_scores.append(our_score)
        paper_scores.append(paper_score)
        differences.append(our_score - paper_score)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Side-by-side comparison
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, our_scores, width, label='Our Implementation', 
                   color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, paper_scores, width, label='Paper Baseline',
                   color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Difference chart
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in differences]
    bars = ax2.barh(datasets, differences, color=colors, alpha=0.8)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Difference (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Difference (Our - Paper)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        ax2.text(diff, i, f'{diff:+.1f}%', 
                va='center', ha='left' if diff > 0 else 'right', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./data/results/detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("\nDetailed comparison chart saved to: ./data/results/detailed_comparison.png")
    plt.close()

def generate_analysis_report(results: Dict):
    """Generate a comprehensive analysis report."""
    paper_baseline = {
        'fever': 63.4,
        'hotpotqa': 34.1,
        'medmcqa': 70.5,
        'mmlu_physics': 45.5,
        'mmlu_biology': 83.0
    }
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE EVALUATION ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Overall summary
    our_avg = sum(r['metric_value'] for r in results.values()) / len(results)
    paper_avg = sum(paper_baseline.get(ds, 0) for ds in results.keys()) / len(results)
    
    report.append("OVERALL SUMMARY")
    report.append("-"*80)
    report.append(f"Average Score (Our Implementation): {our_avg:.2f}%")
    report.append(f"Average Score (Paper Baseline): {paper_avg:.2f}%")
    report.append(f"Overall Difference: {our_avg - paper_avg:+.2f}%")
    report.append("")
    
    # Dataset-by-dataset analysis
    report.append("DATASET-BY-DATASET ANALYSIS")
    report.append("-"*80)
    
    for dataset_name, result in results.items():
        our_score = result['metric_value']
        paper_score = paper_baseline.get(dataset_name, 0)
        diff = our_score - paper_score
        pct_change = (diff / paper_score * 100) if paper_score > 0 else 0
        
        report.append(f"\n{dataset_name.upper()}:")
        report.append(f"  Our Result: {our_score:.2f}%")
        report.append(f"  Paper Baseline: {paper_score:.2f}%")
        report.append(f"  Difference: {diff:+.2f}% ({pct_change:+.2f}%)")
        
        if diff > 0:
            report.append(f"  Status: EXCEEDS paper baseline by {diff:.2f}%")
        elif abs(diff) < 5:
            report.append(f"  Status: CLOSE to paper baseline (within 5%)")
        else:
            report.append(f"  Status: BELOW paper baseline by {abs(diff):.2f}%")
    
    report.append("")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open('./data/results/analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nAnalysis report saved to: ./data/results/analysis_report.txt")

def main():
    """Main analysis function."""
    results = load_results()
    
    print("Generating comprehensive analysis...")
    print("="*80)
    
    # Generate all analyses
    generate_analysis_report(results)
    analyze_fever_confusion_matrix(results)
    analyze_error_patterns(results)
    create_comparison_chart(results)
    
    print("\n" + "="*80)
    print("Analysis complete! All visualizations and reports saved to ./data/results/")
    print("="*80)

if __name__ == "__main__":
    main()

