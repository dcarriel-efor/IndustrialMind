"""
PyTorch Autoencoder Training Pipeline with MLflow Tracking

Train anomaly detection model on industrial sensor data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.ml_platform.models.autoencoder import SensorAutoencoder, VariationalAutoencoder, vae_loss
from services.ml_platform.datasets.sensor_dataset import SensorDataset, create_dataloaders

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if should stop.

        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class AnomalyDetectionTrainer:
    """Trainer for autoencoder-based anomaly detection."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        model_type: str = "autoencoder"
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model (SensorAutoencoder or VariationalAutoencoder)
            device: Device to train on (cuda/cpu)
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            model_type: 'autoencoder' or 'vae'
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_type = model_type

        # Loss function
        if model_type == "autoencoder":
            self.criterion = nn.MSELoss()
        elif model_type == "vae":
            self.criterion = None  # VAE uses custom loss

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.model_type == "autoencoder":
                reconstructed, latent = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_x)

            elif self.model_type == "vae":
                reconstructed, mu, logvar = self.model(batch_x)
                loss = vae_loss(batch_x, reconstructed, mu, logvar, beta=1.0)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        return {"train_loss": avg_loss}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)

                if self.model_type == "autoencoder":
                    reconstructed, latent = self.model(batch_x)
                    loss = self.criterion(reconstructed, batch_x)

                elif self.model_type == "vae":
                    reconstructed, mu, logvar = self.model(batch_x)
                    loss = vae_loss(batch_x, reconstructed, mu, logvar, beta=1.0)

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches

        return {"val_loss": avg_loss}

    def compute_reconstruction_errors(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reconstruction errors for all samples.

        Args:
            data_loader: Data loader

        Returns:
            Tuple of (errors, labels):
                - errors: Reconstruction error per sample
                - labels: Ground truth labels (0=normal, 1=anomaly)
        """
        self.model.eval()
        all_errors = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)

                # Compute reconstruction error
                error = self.model.reconstruction_error(batch_x)

                all_errors.append(error.cpu().numpy())
                all_labels.append(batch_y.numpy())

        errors = np.concatenate(all_errors)
        labels = np.concatenate(all_labels)

        return errors, labels


def select_threshold(
    errors: np.ndarray,
    labels: np.ndarray,
    method: str = "percentile",
    percentile: float = 95.0
) -> float:
    """
    Select anomaly detection threshold.

    Methods:
    - 'percentile': Use N-th percentile of normal sample errors
    - 'best_f1': Choose threshold that maximizes F1 score on validation set

    Args:
        errors: Reconstruction errors
        labels: Ground truth labels
        method: Threshold selection method
        percentile: Percentile for 'percentile' method

    Returns:
        Selected threshold value
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    if method == "percentile":
        # Use N-th percentile of NORMAL samples only
        normal_errors = errors[labels == 0]
        threshold = np.percentile(normal_errors, percentile)

    elif method == "best_f1":
        # Try different thresholds and pick best F1
        thresholds = np.percentile(errors, np.linspace(90, 99.9, 100))
        best_f1 = 0
        best_threshold = 0

        for thresh in thresholds:
            preds = (errors > thresh).astype(int)
            f1 = f1_score(labels, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        threshold = best_threshold

    else:
        raise ValueError(f"Unknown method: {method}")

    return threshold


def evaluate_model(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        errors: Reconstruction errors
        labels: Ground truth labels
        threshold: Anomaly threshold

    Returns:
        Dictionary with metrics (F1, precision, recall, accuracy)
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, accuracy_score,
        confusion_matrix, roc_auc_score
    )

    # Predictions
    predictions = (errors > threshold).astype(int)

    # Compute metrics
    metrics = {
        "f1_score": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
        "threshold": threshold
    }

    # ROC AUC (using errors as scores)
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(labels, errors)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    })

    return metrics


def train(args):
    """Main training function."""

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")

    # Create DataLoaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        train_df, val_df, test_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler_type=args.scaler_type
    )

    # Save scaler
    scaler_path = Path(args.output_dir) / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)

    if args.model_type == "autoencoder":
        model = SensorAutoencoder(
            input_dim=4,
            latent_dim=args.latent_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout
        )
    elif args.model_type == "vae":
        model = VariationalAutoencoder(
            input_dim=4,
            latent_dim=args.latent_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print(f"Model: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Trainer
    trainer = AnomalyDetectionTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        model_type=args.model_type
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)

    # MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_params({
            "model_type": args.model_type,
            "latent_dim": args.latent_dim,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "scaler_type": args.scaler_type,
            "seed": args.seed,
            "device": str(device)
        })

        # Training loop
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, args.max_epochs + 1):
            # Train
            train_metrics = trainer.train_epoch(train_loader)

            # Validate
            val_metrics = trainer.validate(val_loader)

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"]
            }, step=epoch)

            # Print progress
            print(f"Epoch {epoch}/{args.max_epochs} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Learning rate scheduler
            scheduler.step(val_metrics["val_loss"])

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_epoch = epoch

                # Save checkpoint
                checkpoint_path = Path(args.output_dir) / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, checkpoint_path)

            # Early stopping
            if early_stopping(val_metrics["val_loss"]):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

        # Load best model
        checkpoint = torch.load(Path(args.output_dir) / "best_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluation
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)

        # Compute reconstruction errors
        val_errors, val_labels = trainer.compute_reconstruction_errors(val_loader)
        test_errors, test_labels = trainer.compute_reconstruction_errors(test_loader)

        # Select threshold
        threshold = select_threshold(
            val_errors,
            val_labels,
            method=args.threshold_method,
            percentile=args.threshold_percentile
        )
        print(f"Selected threshold: {threshold:.4f}")

        # Evaluate on test set
        test_metrics = evaluate_model(test_errors, test_labels, threshold)

        print("\nTest Set Metrics:")
        print(f"  F1 Score:   {test_metrics['f1_score']:.4f}")
        print(f"  Precision:  {test_metrics['precision']:.4f}")
        print(f"  Recall:     {test_metrics['recall']:.4f}")
        print(f"  Accuracy:   {test_metrics['accuracy']:.4f}")
        if 'roc_auc' in test_metrics:
            print(f"  ROC AUC:    {test_metrics['roc_auc']:.4f}")

        # Log test metrics
        mlflow.log_metrics(test_metrics)

        # Save threshold
        threshold_path = Path(args.output_dir) / "threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({"threshold": float(threshold)}, f)

        # Log artifacts
        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_artifact(str(scaler_path))
        mlflow.log_artifact(str(threshold_path))

        # Log model
        mlflow.pytorch.log_model(model, "model")

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Artifacts saved to: {args.output_dir}")
        print(f"MLflow tracking: {args.mlflow_uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autoencoder for anomaly detection")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="models/autoencoder", help="Output directory")

    # Model
    parser.add_argument("--model-type", type=str, default="autoencoder", choices=["autoencoder", "vae"], help="Model type")
    parser.add_argument("--latent-dim", type=int, default=4, help="Latent dimension")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32], help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # Training
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--scaler-type", type=str, default="minmax", choices=["minmax", "standard"], help="Scaler type")

    # Threshold selection
    parser.add_argument("--threshold-method", type=str, default="percentile", choices=["percentile", "best_f1"], help="Threshold selection method")
    parser.add_argument("--threshold-percentile", type=float, default=95.0, help="Percentile for threshold (if using percentile method)")

    # MLflow
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5011", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default="month_02_anomaly_detection", help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()

    # Set run name if not provided
    if args.run_name is None:
        args.run_name = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    train(args)
