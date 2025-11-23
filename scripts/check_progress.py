"""Check evaluation progress."""
import os
import json
import glob
from datetime import datetime

def check_evaluation_progress():
    """Check if evaluation is running and show latest results."""
    results_dir = "./data/results"
    
    # Check for log file
    log_file = "evaluation_output.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f"Log file has {len(lines)} lines")
            if lines:
                print("\nLast 10 lines:")
                for line in lines[-10:]:
                    print(line.rstrip())
    
    # Check for results files (both incremental and final)
    json_files = glob.glob(os.path.join(results_dir, "evaluation_*.json"))
    if json_files:
        # Prefer incremental file if it exists (most recent progress)
        incremental_file = os.path.join(results_dir, "evaluation_incremental.json")
        if os.path.exists(incremental_file):
            latest_file = incremental_file
            print(f"\nIncremental results file: {latest_file}")
        else:
            latest_file = max(json_files, key=os.path.getctime)
            print(f"\nLatest results file: {latest_file}")
        
        print(f"Modified: {datetime.fromtimestamp(os.path.getctime(latest_file))}")
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
            print(f"\nCompleted datasets: {len(results)}/5")
            print("\nResults Summary:")
            print("="*60)
            for dataset, result in results.items():
                metric = result.get('metric_value', 0)
                samples = result.get('num_samples', 0)
                print(f"  {dataset:20s}: {samples:3d} samples, {metric:6.2f}%")
            print("="*60)
            
            # Show which datasets are remaining
            all_datasets = ['fever', 'hotpotqa', 'medmcqa', 'mmlu_physics', 'mmlu_biology']
            remaining = [d for d in all_datasets if d not in results]
            if remaining:
                print(f"\nRemaining datasets: {', '.join(remaining)}")
                print(f"\nTo resume: python scripts/resume_evaluation.py")
    else:
        print("No results files found yet.")
        print("Run: python scripts/run_evaluation.py")

if __name__ == "__main__":
    check_evaluation_progress()

