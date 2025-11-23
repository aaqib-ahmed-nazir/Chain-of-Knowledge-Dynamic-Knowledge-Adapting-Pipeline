"""Script to visualize evaluation results from JSON files."""
import json
import os
import glob
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd

def load_latest_results(results_dir: str = "./data/results") -> Dict:
    """Load the most recent evaluation results."""
    json_files = glob.glob(os.path.join(results_dir, "evaluation_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No evaluation results found in {results_dir}")
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_results_table(results: Dict) -> pd.DataFrame:
    """Create a results table from evaluation results."""
    data = []
    for dataset_name, result in results.items():
        data.append({
            'Dataset': dataset_name.upper(),
            'Samples': result['num_samples'],
            'Accuracy (%)': f"{result['metric_value']:.2f}",
            'Metric': 'Accuracy' if dataset_name != 'hotpotqa' else 'Exact Match'
        })
    
    df = pd.DataFrame(data)
    return df

def create_bar_chart(results: Dict, output_path: str = "./data/results/results_chart.png"):
    """Create a bar chart of results."""
    datasets = []
    scores = []
    
    for dataset_name, result in results.items():
        datasets.append(dataset_name.upper())
        scores.append(result['metric_value'])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(datasets, scores, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Chain-of-Knowledge Evaluation Results', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")

def print_detailed_results(results: Dict):
    """Print detailed results table."""
    print("\n" + "="*70)
    print("DETAILED EVALUATION RESULTS")
    print("="*70)
    
    df = create_results_table(results)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    scores = [result['metric_value'] for result in results.values()]
    print(f"Average Score: {sum(scores)/len(scores):.2f}%")
    print(f"Best Score: {max(scores):.2f}% ({max(results.items(), key=lambda x: x[1]['metric_value'])[0].upper()})")
    print(f"Worst Score: {min(scores):.2f}% ({min(results.items(), key=lambda x: x[1]['metric_value'])[0].upper()})")
    print("="*70)

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

