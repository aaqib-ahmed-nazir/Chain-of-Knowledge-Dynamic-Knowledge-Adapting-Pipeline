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
    
    # Check for results files
    json_files = glob.glob(os.path.join(results_dir, "evaluation_*.json"))
    if json_files:
        latest_file = max(json_files, key=os.path.getctime)
        print(f"\nLatest results file: {latest_file}")
        print(f"Modified: {datetime.fromtimestamp(os.path.getctime(latest_file))}")
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
            print(f"\nCompleted datasets: {len(results)}")
            for dataset, result in results.items():
                print(f"  {dataset}: {result.get('num_samples', 0)} samples, {result.get('metric_value', 0):.2f}%")
    else:
        print("No results files found yet.")

if __name__ == "__main__":
    check_evaluation_progress()

