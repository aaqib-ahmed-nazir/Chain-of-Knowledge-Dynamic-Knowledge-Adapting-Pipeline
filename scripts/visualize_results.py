"""Script to visualize evaluation results from JSON files."""
import json
import os
import glob
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd

def load_latest_results(results_dir: str = "./data/results") -> Dict:
    """Load the most recent evaluation results."""
    # Try the specific timestamped results file first (our final results)
    final_file = os.path.join(results_dir, "evaluation_20251129_165815.json")
    if os.path.exists(final_file):
        print(f"Loading results from: {final_file}")
        with open(final_file, 'r') as f:
            return json.load(f)
    
    # Fallback to timestamped files
    json_files = glob.glob(os.path.join(results_dir, "evaluation_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No evaluation results found in {results_dir}")
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_results_table(results: Dict) -> pd.DataFrame:
    """Create a results table from evaluation results with paper baseline."""
    paper_baseline = {
        'fever': 63.4,
        'hotpotqa': 34.1,
        'medmcqa': 70.5,
        'mmlu_physics': 45.5,
        'mmlu_biology': 83.0
    }
    
    data = []
    for dataset_name, result in results.items():
        our_score = result['metric_value']
        paper_score = paper_baseline.get(dataset_name, 0)
        difference = our_score - paper_score
        
        data.append({
            'Dataset': dataset_name.upper(),
            'Samples': result['num_samples'],
            'Our Result (%)': f"{our_score:.2f}",
            'Paper Baseline (%)': f"{paper_score:.2f}",
            'Difference (%)': f"{difference:+.2f}",
            'Metric': 'Accuracy' if dataset_name != 'hotpotqa' else 'Exact Match'
        })
    
    df = pd.DataFrame(data)
    return df

def create_bar_chart(results: Dict, output_path: str = "./data/results/results_chart.png"):
    """Create a bar chart of results with paper baseline comparison."""
    # Paper baseline results
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
    
    for dataset_name, result in results.items():
        datasets.append(dataset_name.upper())
        our_scores.append(result['metric_value'])
        paper_scores.append(paper_baseline.get(dataset_name, 0))
    
    x = range(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar([i - width/2 for i in x], our_scores, width, 
                   label='Our Implementation', color='#3498db', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], paper_scores, width,
                   label='Paper Baseline', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Chain-of-Knowledge Evaluation Results vs Paper Baseline', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    plt.close()

def print_detailed_results(results: Dict):
    """Print detailed results table with paper baseline comparison."""
    # Paper baseline results
    paper_baseline = {
        'fever': 63.4,
        'hotpotqa': 34.1,
        'medmcqa': 70.5,
        'mmlu_physics': 45.5,
        'mmlu_biology': 83.0
    }
    
    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS")
    print("="*80)
    
    data = []
    for dataset_name, result in results.items():
        our_score = result['metric_value']
        paper_score = paper_baseline.get(dataset_name, 0)
        difference = our_score - paper_score
        if difference > 0:
            status = "Exceeds"
        elif difference < -5:
            status = "Below"
        else:
            status = "Close"
        
        data.append({
            'Dataset': dataset_name.upper(),
            'Samples': result['num_samples'],
            'Our Result (%)': f"{our_score:.2f}",
            'Paper Baseline (%)': f"{paper_score:.2f}",
            'Difference': f"{difference:+.2f}",
            'Status': status
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    scores = [result['metric_value'] for result in results.values()]
    paper_scores = [paper_baseline.get(ds, 0) for ds in results.keys()]
    
    print(f"Average Score (Our): {sum(scores)/len(scores):.2f}%")
    print(f"Average Score (Paper): {sum(paper_scores)/len(paper_scores):.2f}%")
    print(f"Average Difference: {(sum(scores) - sum(paper_scores))/len(scores):+.2f}%")
    print(f"\nBest Score: {max(scores):.2f}% ({max(results.items(), key=lambda x: x[1]['metric_value'])[0].upper()})")
    print(f"Worst Score: {min(scores):.2f}% ({min(results.items(), key=lambda x: x[1]['metric_value'])[0].upper()})")
    
    # Count datasets that exceed paper baseline
    exceeding = sum(1 for ds, result in results.items() 
                   if result['metric_value'] > paper_baseline.get(ds, 0))
    print(f"\nDatasets Exceeding Paper Baseline: {exceeding}/{len(results)}")
    print("="*80)

def main():
    """Main function to visualize results."""
    try:
        results = load_latest_results()
        
        print_detailed_results(results)
        
        # Create visualization
        create_bar_chart(results)
        
        # Save table as CSV
        df = create_results_table(results)
        csv_path = "./data/results/results_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nTable saved to: {csv_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the evaluation first: python scripts/run_evaluation.py")

if __name__ == "__main__":
    main()

