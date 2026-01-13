# PyTorch Training Loop Pattern

## Purpose
Standard training loop pattern with validation, early stopping, and checkpointing for PyTorch models.

## When to Use
- Training any PyTorch model
- Need validation during training
- Want early stopping to prevent overfitting
- Need model checkpointing

## Prerequisites
- PyTorch installed
- Model class defined (inheriting from nn.Module)
- Dataset and DataLoader configured

---

## Template

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Callable

class Trainer:
    """
    Standard PyTorch training loop with validation and early stopping.

    Example:
        model = YourModel()
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        history = trainer.train(epochs=100, early_stopping_patience=10)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (val_loss: {val_loss:.6f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['val_loss']

    def train(
        self,
        epochs: int,
        early_stopping_patience: Optional[int] = None,
        min_delta: float = 1e-6,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.

        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
            min_delta: Minimum change to qualify as improvement
            verbose: Print training progress

        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch'].append(epoch)

            # Check for improvement
            is_best = val_loss < (best_val_loss - min_delta)
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)

            # Print progress
            if verbose:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Best: {best_val_loss:.6f}")

            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                print(f"   No improvement for {early_stopping_patience} epochs")
                break

        return self.history
```

---

## Example Usage for IndustrialMind

### For Autoencoder (Month 2)

```python
from ml_models.anomaly_detector.model import SensorAutoencoder
from ml_models.anomaly_detector.dataset import SensorDataset

# Create model
model = SensorAutoencoder(input_dim=4, latent_dim=2)

# Create datasets
train_dataset = SensorDataset(influxdb_client, time_range="2024-01-01/2024-02-01")
val_dataset = SensorDataset(influxdb_client, time_range="2024-02-01/2024-02-15")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Setup trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='checkpoints/autoencoder'
)

# Train with early stopping
history = trainer.train(
    epochs=100,
    early_stopping_patience=10,
    min_delta=1e-6
)

# Load best model for inference
trainer.load_checkpoint('checkpoints/autoencoder/best_checkpoint.pt')
```

---

## Variations

### With Learning Rate Scheduling

```python
class TrainerWithScheduler(Trainer):
    def __init__(self, *args, scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler

    def train_epoch(self):
        loss = super().train_epoch()
        if self.scheduler:
            self.scheduler.step()
        return loss

# Usage
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
trainer = TrainerWithScheduler(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler
)
```

### With Gradient Clipping

```python
def train_epoch_with_clip(self, max_grad_norm: float = 1.0) -> float:
    """Train epoch with gradient clipping."""
    self.model.train()
    total_loss = 0.0

    for data, target in self.train_loader:
        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()
        total_loss += loss.item()

    return total_loss / len(self.train_loader)
```

---

## Common Pitfalls

### ❌ Forgetting to Set Train/Eval Mode
```python
# Wrong - no mode setting
for batch in train_loader:
    output = model(batch)  # Dropout behaves incorrectly!

# Correct
model.train()  # Set training mode
for batch in train_loader:
    output = model(batch)
```

### ❌ Not Moving Data to Device
```python
# Wrong - data on CPU, model on GPU
output = model(data)  # RuntimeError!

# Correct
data = data.to(device)
output = model(data)
```

### ❌ Forgetting to Zero Gradients
```python
# Wrong - gradients accumulate!
loss.backward()
optimizer.step()

# Correct
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## References

- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Early Stopping Pattern](https://github.com/Bjarten/early-stopping-pytorch)

---

*Created: 2026-01-12*
*For: IndustrialMind Month 2 - PyTorch Model Training*
*Version: 1.0*
