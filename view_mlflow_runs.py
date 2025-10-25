"""Script to view MLflow runs programmatically."""

import mlflow
import pandas as pd

# Set tracking URI to the new directory
mlflow.set_tracking_uri("file:./mlruns_new")

# Get all experiments
experiments = mlflow.search_experiments()

print("\n" + "="*80)
print("MLFLOW EXPERIMENTS")
print("="*80 + "\n")

for exp in experiments:
    print(f"Experiment: {exp.name}")
    print(f"  ID: {exp.experiment_id}")
    print(f"  Artifact Location: {exp.artifact_location}")
    
    # Get runs for this experiment
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    
    if len(runs) > 0:
        print(f"  Runs: {len(runs)}\n")
        
        for idx, run in runs.iterrows():
            print(f"  Run {idx + 1}:")
            print(f"    Run ID: {run['run_id']}")
            print(f"    Status: {run['status']}")
            print(f"    Start Time: {run['start_time']}")
            
            # Print parameters
            param_cols = [col for col in runs.columns if col.startswith('params.')]
            if param_cols:
                print(f"    Parameters:")
                for col in param_cols[:10]:  # Show first 10 params
                    param_name = col.replace('params.', '')
                    print(f"      {param_name}: {run[col]}")
            
            # Print metrics
            metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
            if metric_cols:
                print(f"    Metrics:")
                for col in metric_cols:
                    metric_name = col.replace('metrics.', '')
                    print(f"      {metric_name}: {run[col]}")
            
            # Print artifacts
            artifacts_uri = run['artifact_uri']
            print(f"    Artifacts URI: {artifacts_uri}")
            print()
    else:
        print(f"  No runs found\n")

print("="*80)
print("\nTo view in browser, run:")
print("  cd /Users/eugenekim/Emo-CLIM")
print("  mlflow ui --backend-store-uri file:./mlruns_new --port 5001")
print("\nThen open: http://localhost:5001")
print("="*80 + "\n")

