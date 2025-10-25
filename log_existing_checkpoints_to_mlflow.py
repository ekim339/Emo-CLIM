"""Script to retroactively log existing trained checkpoints to MLflow."""

import os
import argparse
import yaml
import mlflow
from pathlib import Path
try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not installed. Metrics from event files won't be extracted.")


def parse_hparams_from_yaml(hparams_file):
    """Parse hyperparameters from hparams.yaml file."""
    with open(hparams_file, 'r') as f:
        hparams = yaml.safe_load(f)
    return hparams


def extract_metrics_from_tensorboard(event_file):
    """Extract metrics from TensorBoard event files."""
    if not TENSORBOARD_AVAILABLE:
        return {}
    
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    metrics = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        # Get the last value for each metric
        if events:
            metrics[tag] = events[-1].value
            # Also log all values as a history
            metrics[f"{tag}_history"] = [(e.step, e.value) for e in events]
    
    return metrics


def log_checkpoint_to_mlflow(checkpoint_dir, experiment_name, run_name=None):
    """
    Log an existing checkpoint directory to MLflow.
    
    Args:
        checkpoint_dir: Path to the version directory (e.g., train_logs/.../version_0/)
        experiment_name: MLflow experiment name
        run_name: Optional run name
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Check if directory exists
    if not checkpoint_dir.exists():
        print(f"Directory not found: {checkpoint_dir}")
        return False
    
    # Find hparams.yaml
    hparams_file = checkpoint_dir / "hparams.yaml"
    if not hparams_file.exists():
        print(f"No hparams.yaml found in {checkpoint_dir}")
        return False
    
    # Find checkpoint files
    checkpoints_dir = checkpoint_dir / "checkpoints"
    checkpoint_files = []
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
    
    # Find TensorBoard event files
    event_files = list(checkpoint_dir.glob("events.out.tfevents.*"))
    
    print(f"\n{'='*80}")
    print(f"Logging to MLflow: {checkpoint_dir}")
    print(f"  - Found {len(checkpoint_files)} checkpoint(s)")
    print(f"  - Found {len(event_files)} event file(s)")
    print(f"{'='*80}\n")
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # 1. Log hyperparameters
        print("Logging hyperparameters...")
        hparams = parse_hparams_from_yaml(hparams_file)
        for key, value in hparams.items():
            # MLflow params must be strings or numbers
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))
        
        # 2. Extract and log metrics from TensorBoard events
        if event_files:
            print("Extracting metrics from TensorBoard events...")
            for event_file in event_files:
                metrics = extract_metrics_from_tensorboard(str(event_file))
                
                # Log final metrics (last value)
                for key, value in metrics.items():
                    if not key.endswith('_history'):
                        mlflow.log_metric(key, value)
                
                # Log metric histories
                for key, value in metrics.items():
                    if key.endswith('_history'):
                        metric_name = key.replace('_history', '')
                        for step, metric_value in value:
                            mlflow.log_metric(metric_name, metric_value, step=step)
        
        # 3. Log checkpoint files as artifacts
        if checkpoint_files:
            print(f"Logging {len(checkpoint_files)} checkpoint file(s) as artifacts...")
            for ckpt_file in checkpoint_files:
                mlflow.log_artifact(str(ckpt_file), artifact_path="checkpoints")
        
        # 4. Log the hparams.yaml file
        print("Logging hparams.yaml...")
        mlflow.log_artifact(str(hparams_file))
        
        # 5. Add tags for organization
        mlflow.set_tag("source", "retroactive_logging")
        mlflow.set_tag("checkpoint_dir", str(checkpoint_dir))
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n✓ Successfully logged to MLflow!")
        print(f"  Run ID: {run_id}")
        print(f"  Experiment: {experiment_name}\n")
        
        return True


def scan_and_log_all_checkpoints(train_logs_dir, mlflow_tracking_uri="file:./mlruns", dry_run=False):
    """
    Scan train_logs directory and log all checkpoints to MLflow.
    
    Args:
        train_logs_dir: Root train_logs directory
        mlflow_tracking_uri: MLflow tracking URI
        dry_run: If True, only print what would be logged without actually logging
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    train_logs_path = Path(train_logs_dir)
    
    if not train_logs_path.exists():
        print(f"Error: {train_logs_dir} does not exist")
        return
    
    # Find all version directories
    version_dirs = []
    for root, dirs, files in os.walk(train_logs_path):
        if 'hparams.yaml' in files:
            version_dirs.append(Path(root))
    
    if not version_dirs:
        print(f"No checkpoint directories found in {train_logs_dir}")
        return
    
    print(f"\nFound {len(version_dirs)} checkpoint directory(ies) to log:\n")
    
    for i, version_dir in enumerate(version_dirs, 1):
        # Construct experiment name from path structure
        # e.g., train_logs/single_task/CLAP/frozen/all_losses/version_0
        # -> experiment: single_task/CLAP/frozen/all_losses
        relative_path = version_dir.relative_to(train_logs_path)
        parts = relative_path.parts
        
        if len(parts) > 1:
            experiment_name = "/".join(parts[:-1])  # Everything except version_X
            run_name = parts[-1]  # version_0, version_1, etc.
        else:
            experiment_name = "default"
            run_name = parts[-1]
        
        print(f"[{i}/{len(version_dirs)}] {relative_path}")
        print(f"    Experiment: {experiment_name}")
        print(f"    Run: {run_name}")
        
        if dry_run:
            print("    (DRY RUN - not logging)")
            continue
        
        success = log_checkpoint_to_mlflow(
            version_dir,
            experiment_name=experiment_name,
            run_name=run_name
        )
        
        if not success:
            print(f"    ✗ Failed to log")
    
    print(f"\n{'='*80}")
    if dry_run:
        print("DRY RUN COMPLETE - No data was logged to MLflow")
    else:
        print("ALL CHECKPOINTS LOGGED TO MLFLOW!")
        print(f"\nView your experiments:")
        print(f"  mlflow ui --port 5000")
        print(f"  Then open: http://localhost:5000")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Retroactively log existing trained checkpoints to MLflow"
    )
    parser.add_argument(
        "--train_logs_dir",
        type=str,
        default="train_logs",
        help="Path to train_logs directory (default: train_logs)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to specific checkpoint directory to log (e.g., train_logs/.../version_0)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="MLflow experiment name (auto-detected if not provided)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (auto-detected if not provided)"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI (default: file:./mlruns)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be logged without actually logging"
    )
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Log a specific checkpoint directory
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir)
        
        # Auto-detect experiment name from path if not provided
        if not args.experiment_name:
            if "train_logs" in checkpoint_path.parts:
                idx = checkpoint_path.parts.index("train_logs")
                parts = checkpoint_path.parts[idx+1:]
                if len(parts) > 1:
                    args.experiment_name = "/".join(parts[:-1])
                    if not args.run_name:
                        args.run_name = parts[-1]
            
            if not args.experiment_name:
                args.experiment_name = "default"
        
        if args.dry_run:
            print(f"DRY RUN: Would log {checkpoint_path}")
            print(f"  Experiment: {args.experiment_name}")
            print(f"  Run: {args.run_name}")
        else:
            log_checkpoint_to_mlflow(
                checkpoint_path,
                experiment_name=args.experiment_name,
                run_name=args.run_name
            )
    
    # Scan and log all checkpoints
    else:
        scan_and_log_all_checkpoints(
            args.train_logs_dir,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()

