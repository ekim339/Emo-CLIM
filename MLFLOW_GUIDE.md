# MLflow Integration Guide

## Overview
Your training script now supports MLflow logging alongside TensorBoard. All metrics logged with `self.log()` in the `Image2Music` class will automatically be tracked.

## What Gets Logged Automatically

Based on your `Image2Music` trainer, these metrics are logged:

### Training Metrics (per batch)
- `train/image2image_loss`
- `train/audio2audio_loss`
- `train/image2audio_loss`
- `train/audio2image_loss`
- `train/total_loss`

### Validation Metrics (per epoch)
- `validation/image2image_loss`
- `validation/audio2audio_loss`
- `validation/image2audio_loss`
- `validation/audio2image_loss`
- `validation/total_loss`

### Hyperparameters
All training hyperparameters from `configs/config_train.yaml` are automatically logged.

## Usage

### 1. Run Training
```bash
python climur/scripts/training.py --config_file configs/config_train.yaml
```

The script is configured to use **both** TensorBoard and MLflow by default.

### 2. View MLflow UI
Open the MLflow UI to view your experiments:

```bash
mlflow ui
```

Then open your browser to: **http://localhost:5000**

### 3. MLflow UI Features

#### Main Dashboard
- **Experiments**: Organized by your experiment names (e.g., `single_task/CLAP/frozen/all_losses`)
- **Runs**: Each training run is tracked separately with a unique ID
- **Compare Runs**: Select multiple runs to compare metrics side-by-side

#### For Each Run You Can See:
- **Metrics**: Interactive plots of all logged metrics over time
- **Parameters**: All hyperparameters used
- **Artifacts**: Model checkpoints (if you add them - see below)
- **Tags**: Metadata about the run
- **System Metrics**: GPU/CPU usage, memory, etc.

## Configuration Options

### Current Setup (Both Loggers)
```python
logger = [tensorboard_logger, mlflow_logger]
```
- Logs to both TensorBoard and MLflow
- Good for transition period

### MLflow Only
```python
logger = mlflow_logger
```
- Only logs to MLflow
- Cleaner if you're fully switching

### TensorBoard Only
```python
logger = tensorboard_logger
```
- Default behavior (before MLflow integration)

## Advanced: Log Additional Items

### 1. Log Model Checkpoints to MLflow

Add to your training script after the trainer is created:

```python
from pytorch_lightning.callbacks import ModelCheckpoint

# Update the checkpoint callback
model_ckpt_callback = ModelCheckpoint(
    dirpath=mlflow_logger.experiment.get_artifact_uri(),  # Save to MLflow
    monitor="validation/total_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
)
```

### 2. Log Custom Metrics in Your Trainer

In `climur/trainers/image2music.py`, you can add custom metrics:

```python
def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    # ... existing code ...
    
    # Log custom metric
    self.log("validation/custom_metric", some_value)
    
    return total_loss
```

### 3. Log Additional Artifacts

Add at the end of training script:

```python
# Log the config file
mlflow_logger.experiment.log_artifact(args.config_file)

# Log plots or other files
mlflow_logger.experiment.log_artifact("plots/training_curve.png")
```

## MLflow Directory Structure

```
Emo-CLIM/
├── mlruns/                          # MLflow tracking directory
│   ├── 0/                          # Default experiment
│   ├── .../                        # Numbered experiments
│   │   ├── <run_id>/              # Each training run
│   │   │   ├── artifacts/         # Model checkpoints, plots, etc.
│   │   │   ├── metrics/           # Metric files
│   │   │   ├── params/            # Hyperparameters
│   │   │   └── tags/              # Metadata tags
│   └── models/                     # Registered models (optional)
```

## Remote MLflow Tracking Server (Optional)

For team collaboration, you can use a remote MLflow server:

```python
mlflow_logger = MLFlowLogger(
    experiment_name=experiment_name,
    tracking_uri="http://your-mlflow-server:5000",  # Remote server
    run_name=logging_configs.get("experiment_version", None),
)
```

## Querying Experiments Programmatically

```python
import mlflow

# List all experiments
experiments = mlflow.search_experiments()

# Get runs from an experiment
runs = mlflow.search_runs(
    experiment_names=["single_task/CLAP/frozen/all_losses"],
    filter_string="metrics.validation/total_loss < 0.5"
)

# Load a specific run
run = mlflow.get_run(run_id="your-run-id")
print(run.data.metrics)
print(run.data.params)
```

## Tips

1. **Add to .gitignore**: The `mlruns/` directory is large and shouldn't be committed
   ```
   mlruns/
   ```

2. **Experiment Organization**: MLflow automatically organizes runs by experiment name (which is auto-generated from your config)

3. **Compare Models**: Use MLflow UI to compare different audio backbones, loss weights, etc.

4. **Track Everything**: MLflow persists forever, so you can always go back and compare old experiments

## Troubleshooting

### "No module named 'mlflow'"
```bash
pip install mlflow
```

### MLflow UI shows no experiments
Make sure you're running `mlflow ui` from the project root directory where `mlruns/` is located.

### Can't see metrics
Check that your metrics are being logged with `self.log()` in the LightningModule.

