"""
Quick test script to verify ML platform setup.

Tests:
1. Model initialization
2. Dataset creation
3. Training loop (1 epoch)
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from services.ml_platform.models.autoencoder import SensorAutoencoder, VariationalAutoencoder
from services.ml_platform.datasets.sensor_dataset import SensorDataset
from torch.utils.data import DataLoader


def test_model_initialization():
    """Test model can be created."""
    print("=" * 60)
    print("TEST 1: Model Initialization")
    print("=" * 60)

    try:
        # Standard Autoencoder
        model_ae = SensorAutoencoder(input_dim=4, latent_dim=4)
        print("[OK] SensorAutoencoder created")
        print(f"  Parameters: {sum(p.numel() for p in model_ae.parameters()):,}")

        # VAE
        model_vae = VariationalAutoencoder(input_dim=4, latent_dim=4)
        print("[OK] VariationalAutoencoder created")
        print(f"  Parameters: {sum(p.numel() for p in model_vae.parameters()):,}")

        # Test forward pass
        x = torch.randn(32, 4)
        reconstructed, latent = model_ae(x)
        assert reconstructed.shape == (32, 4), "Wrong output shape"
        assert latent.shape == (32, 4), "Wrong latent shape"
        print(f"[OK] Forward pass works (input: {x.shape}, output: {reconstructed.shape})")

        return True
    except Exception as e:
        print(f"[FAIL] Model initialization failed: {e}")
        return False


def test_dataset_creation():
    """Test dataset creation."""
    print("\n" + "=" * 60)
    print("TEST 2: Dataset Creation")
    print("=" * 60)

    try:
        # Create dummy data
        np.random.seed(42)
        n_samples = 1000

        dummy_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1s"),
            "machine_id": "MACHINE_001",
            "temperature": np.random.normal(55, 5, n_samples),
            "vibration": np.random.normal(1.0, 0.2, n_samples),
            "pressure": np.random.normal(50, 5, n_samples),
            "power_consumption": np.random.normal(250, 25, n_samples),
            "is_anomaly": np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        })
        print(f"[OK] Created dummy data: {len(dummy_data)} samples")

        # Create dataset
        dataset = SensorDataset(dummy_data, fit_scaler=True)
        print(f"[OK] SensorDataset created: {len(dataset)} samples")

        # Get sample
        x, y = dataset[0]
        print(f"[OK] Sample retrieved: features={x.shape}, label={y}")

        # Create DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        print(f"[OK] DataLoader works: batch_shape={batch_x.shape}")

        return True
    except Exception as e:
        print(f"[FAIL] Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test one epoch of training."""
    print("\n" + "=" * 60)
    print("TEST 3: Training Loop (1 Epoch)")
    print("=" * 60)

    try:
        # Create dummy data
        np.random.seed(42)
        dummy_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=500, freq="1s"),
            "machine_id": "MACHINE_001",
            "temperature": np.random.normal(55, 5, 500),
            "vibration": np.random.normal(1.0, 0.2, 500),
            "pressure": np.random.normal(50, 5, 500),
            "power_consumption": np.random.normal(250, 25, 500),
            "is_anomaly": 0
        })

        # Dataset and loader
        dataset = SensorDataset(dummy_data, fit_scaler=True)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        print("[OK] Created dataset and loader")

        # Model
        model = SensorAutoencoder(input_dim=4, latent_dim=4)
        device = torch.device("cpu")
        model = model.to(device)
        print(f"[OK] Model initialized on {device}")

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        print("[OK] Optimizer and loss function ready")

        # Training epoch
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            reconstructed, latent = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"[OK] Training epoch complete: avg_loss={avg_loss:.4f}")

        return True
    except Exception as e:
        print(f"[FAIL] Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ML PLATFORM SETUP TEST")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Model Initialization", test_model_initialization()))
    results.append(("Dataset Creation", test_dataset_creation()))
    results.append(("Training Loop", test_training_loop()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status:4}] | {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n*** All tests passed! ML platform is ready. ***")
        print("\nNext steps:")
        print("  1. Prepare data: python training/prepare_data.py")
        print("  2. Train model: python training/train_autoencoder.py")
    else:
        print("\n*** Some tests failed. Check errors above. ***")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
