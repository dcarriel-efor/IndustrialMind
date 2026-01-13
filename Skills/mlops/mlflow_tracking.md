# MLflow Experiment Tracking Pattern

## Purpose
Standardized pattern for tracking ML experiments with MLflow, including parameters, metrics, artifacts, and models.

## When to Use
- Training any ML model (Month 2+)
- Comparing different model architectures
- Tracking hyperparameter tuning
- Maintaining model registry

## Prerequisites
- MLflow installed
- MLflow tracking server running (local or remote)

---

## Template

```python
import mlflow
import mlflow.pytorch
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json

class MLflowExperiment:
    """
    Wrapper for MLflow experiment tracking.

    Example:
        with MLflowExperiment("anomaly-detection") as experiment:
            experiment.log_params({
                "learning_rate": 1e-3,
                "batch_size": 64,
                "model": "autoencoder"
            })

            for epoch in range(epochs):
                train_loss = train_epoch()
                val_loss = validate()

                experiment.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, step=epoch)

            experiment.log_model(model, "model")
            experiment.log_artifact("config.yaml")
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://localhost:5000",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            run_name: Optional name for this run
            tags: Optional tags for the run
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.tags = tags or {}

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            self.experiment = mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Error setting experiment: {e}")
            raise

        self.run = None
        self.run_id = None

    def __enter__(self):
        """Start MLflow run."""
        self.run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self.run.info.run_id

        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

        # Auto-tag with git info if available
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            mlflow.set_tag("git_commit", repo.head.object.hexsha)
            mlflow.set_tag("git_branch", repo.active_branch.name)
        except:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", str(exc_val))
        else:
            mlflow.set_tag("status", "completed")

        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        mlflow.log_params(params)

    def log_param(self, key: str, value: Any):
        """Log single parameter."""
        mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact (file)."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log artifacts (directory)."""
        mlflow.log_artifacts(local_dir, artifact_path)

    def log_dict(self, dictionary: Dict, filename: str):
        """Log dictionary as JSON artifact."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dictionary, f, indent=2)
            temp_path = f.name

        self.log_artifact(temp_path, filename)
        Path(temp_path).unlink()

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log PyTorch model.

        Args:
            model: PyTorch model
            artifact_path: Path within run's artifact directory
            registered_model_name: If provided, register model in Model Registry
        """
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )

    def log_figure(self, figure, filename: str):
        """Log matplotlib figure."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            figure.savefig(f.name, dpi=150, bbox_inches='tight')
            self.log_artifact(f.name, filename)
            Path(f.name).unlink()


def log_training_run(
    experiment_name: str,
    model: torch.nn.Module,
    config: Dict[str, Any],
    history: Dict[str, list],
    model_name: Optional[str] = None
):
    """
    Convenience function for logging complete training run.

    Example:
        log_training_run(
            experiment_name="anomaly-detection",
            model=trained_model,
            config={
                "learning_rate": 1e-3,
                "batch_size": 64,
                "epochs": 100,
                "model_type": "autoencoder"
            },
            history={
                "train_loss": [0.5, 0.4, 0.3, ...],
                "val_loss": [0.6, 0.5, 0.4, ...]
            },
            model_name="sensor-autoencoder"
        )
    """
    with MLflowExperiment(experiment_name) as exp:
        # Log configuration as parameters
        exp.log_params(config)

        # Log final metrics
        final_metrics = {
            f"final_{key}": values[-1]
            for key, values in history.items()
        }
        exp.log_metrics(final_metrics)

        # Log best metrics
        if 'val_loss' in history:
            best_val_loss = min(history['val_loss'])
            best_epoch = history['val_loss'].index(best_val_loss)
            exp.log_metrics({
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch
            })

        # Log history as artifact
        exp.log_dict(history, "training_history.json")

        # Log model
        exp.log_model(model, "model", registered_model_name=model_name)

        print(f"✓ Logged run to MLflow")
        print(f"  Run ID: {exp.run_id}")
        print(f"  Experiment: {experiment_name}")
```

---

## Example Usage for IndustrialMind

### Basic Training with MLflow

```python
from ml_models.anomaly_detector.model import SensorAutoencoder
from Skills.mlops.mlflow_tracking import MLflowExperiment

# Initialize model and training setup
model = SensorAutoencoder(input_dim=4, latent_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Start MLflow tracking
with MLflowExperiment(
    experiment_name="anomaly-detection",
    run_name="autoencoder-baseline",
    tags={"model_type": "autoencoder", "version": "v1"}
) as experiment:

    # Log hyperparameters
    experiment.log_params({
        "input_dim": 4,
        "latent_dim": 2,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 100,
        "optimizer": "Adam",
        "criterion": "MSELoss"
    })

    # Training loop
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)

        # Log metrics every epoch
        experiment.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

        # Log learning rate if using scheduler
        current_lr = optimizer.param_groups[0]['lr']
        experiment.log_metric("learning_rate", current_lr, step=epoch)

    # Log final model
    experiment.log_model(
        model,
        artifact_path="model",
        registered_model_name="sensor-anomaly-detector"
    )

    # Log model architecture diagram
    # experiment.log_artifact("model_architecture.png")

    # Log confusion matrix or other plots
    # experiment.log_figure(confusion_matrix_fig, "confusion_matrix.png")
```

### Hyperparameter Tuning with MLflow

```python
from itertools import product

# Define hyperparameter grid
param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'latent_dim': [2, 4, 8],
    'batch_size': [32, 64, 128]
}

# Grid search
best_val_loss = float('inf')
best_params = None

for lr, latent_dim, batch_size in product(*param_grid.values()):
    config = {
        'learning_rate': lr,
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'input_dim': 4,
        'epochs': 50
    }

    with MLflowExperiment(
        experiment_name="anomaly-detection-hyperparam-tuning",
        run_name=f"lr{lr}_latent{latent_dim}_bs{batch_size}"
    ) as exp:

        exp.log_params(config)

        # Train model with these hyperparameters
        model = SensorAutoencoder(
            input_dim=config['input_dim'],
            latent_dim=config['latent_dim']
        )
        val_loss = train_and_evaluate(model, config)

        exp.log_metric("final_val_loss", val_loss)

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = config
            exp.log_model(model, "model", registered_model_name="best-autoencoder")

print(f"Best params: {best_params}")
print(f"Best val loss: {best_val_loss}")
```

---

## Variations

### With Automatic System Metrics

```python
import psutil
import GPUtil

class MLflowExperimentWithSystemMetrics(MLflowExperiment):
    def log_system_metrics(self):
        """Log CPU, memory, and GPU usage."""
        # CPU and memory
        mlflow.log_metric("cpu_percent", psutil.cpu_percent())
        mlflow.log_metric("memory_percent", psutil.virtual_memory().percent)

        # GPU if available
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                mlflow.log_metric(f"gpu_{i}_utilization", gpu.load * 100)
                mlflow.log_metric(f"gpu_{i}_memory", gpu.memoryUtil * 100)
        except:
            pass
```

### With Automatic Model Comparison

```python
def compare_models_in_mlflow(experiment_name: str, metric: str = "val_loss"):
    """
    Compare all runs in an experiment and return best.

    Returns:
        Best run ID and its metrics
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1
    )

    if runs:
        best_run = runs[0]
        print(f"Best run: {best_run.info.run_id}")
        print(f"Best {metric}: {best_run.data.metrics[metric]}")
        return best_run.info.run_id, best_run.data.metrics
    else:
        return None, None
```

---

## Common Pitfalls

### ❌ Not Ending Runs Properly
```python
# Wrong - run stays active if exception
mlflow.start_run()
train_model()  # If this fails, run never ends!

# Correct - use context manager
with MLflowExperiment("exp") as exp:
    train_model()  # Automatically ends even if exception
```

### ❌ Logging Too Much
```python
# Wrong - logging every batch (too noisy!)
for batch in train_loader:
    loss = train_step(batch)
    mlflow.log_metric("batch_loss", loss)  # Thousands of datapoints!

# Correct - log epoch-level metrics
for epoch in range(epochs):
    avg_loss = train_epoch()
    mlflow.log_metric("train_loss", avg_loss, step=epoch)
```

### ❌ Not Using Model Registry
```python
# Wrong - just log model as artifact
mlflow.pytorch.log_model(model, "model")

# Correct - register for easy deployment
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="sensor-anomaly-detector"  # Enables versioning!
)
```

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow PyTorch Integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

*Created: 2026-01-12*
*For: IndustrialMind Month 3 - MLOps*
*Version: 1.0*
