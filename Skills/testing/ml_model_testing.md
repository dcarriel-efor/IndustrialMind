# ML Model Testing Pattern

## Purpose
Comprehensive testing patterns for ML models, including unit tests, integration tests, and model-specific tests.

## When to Use
- Testing ML models and pipelines (all months)
- Validating model behavior
- Regression testing after changes
- CI/CD integration

## Prerequisites
- pytest installed
- Models and datasets available

---

## Template

```python
import pytest
import torch
import numpy as np
from pathlib import Path

# ============================================================================
# Model Unit Tests
# ============================================================================

class TestSensorAutoencoder:
    """Unit tests for autoencoder model."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        from ml_models.anomaly_detector.model import SensorAutoencoder
        return SensorAutoencoder(input_dim=4, latent_dim=2)

    @pytest.fixture
    def sample_input(self):
        """Create sample input."""
        return torch.randn(10, 4)  # Batch of 10, 4 features

    def test_model_forward_pass(self, model, sample_input):
        """Test forward pass produces correct shape."""
        output = model(sample_input)
        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_reconstruction_error(self, model, sample_input):
        """Test reconstruction error computation."""
        errors = model.get_reconstruction_error(sample_input)
        assert errors.shape == (10,)
        assert (errors >= 0).all()

    def test_model_parameters(self, model):
        """Test model has trainable parameters."""
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_model_device_compatibility(self, model, sample_input):
        """Test model works on CPU and GPU."""
        # CPU
        output_cpu = model(sample_input)
        assert output_cpu.device.type == 'cpu'

        # GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            input_gpu = sample_input.cuda()
            output_gpu = model_gpu(input_gpu)
            assert output_gpu.device.type == 'cuda'

    def test_model_eval_mode(self, model, sample_input):
        """Test eval mode disables dropout/batchnorm training."""
        model.eval()
        output1 = model(sample_input)
        output2 = model(sample_input)
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)

    def test_model_save_load(self, model, tmp_path):
        """Test model can be saved and loaded."""
        save_path = tmp_path / "model.pt"

        # Save
        torch.save(model.state_dict(), save_path)

        # Load
        from ml_models.anomaly_detector.model import SensorAutoencoder
        loaded_model = SensorAutoencoder(input_dim=4, latent_dim=2)
        loaded_model.load_state_dict(torch.load(save_path))

        # Compare parameters
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)


# ============================================================================
# Dataset Tests
# ============================================================================

class TestSensorDataset:
    """Tests for sensor dataset."""

    @pytest.fixture
    def mock_data(self):
        """Create mock sensor data."""
        np.random.seed(42)
        n_samples = 1000
        return {
            'data': np.random.randn(n_samples, 4),
            'labels': np.random.randint(0, 3, n_samples)
        }

    def test_dataset_length(self, mock_data):
        """Test dataset returns correct length."""
        from ml_models.anomaly_detector.dataset import AnomalyDetectionDataset

        dataset = AnomalyDetectionDataset(
            data=mock_data['data'],
            anomaly_labels=mock_data['labels']
        )

        assert len(dataset) == len(mock_data['data'])

    def test_dataset_getitem(self, mock_data):
        """Test dataset __getitem__ works."""
        from ml_models.anomaly_detector.dataset import AnomalyDetectionDataset

        dataset = AnomalyDetectionDataset(
            data=mock_data['data'],
            anomaly_labels=mock_data['labels']
        )

        sample, target, label = dataset[0]

        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (4,)
        assert isinstance(label, torch.Tensor)

    def test_dataset_normalization(self, mock_data):
        """Test normalization is applied correctly."""
        from ml_models.anomaly_detector.dataset import AnomalyDetectionDataset

        dataset = AnomalyDetectionDataset(
            data=mock_data['data'],
            normalize=True
        )

        # Check mean and std are stored
        assert dataset.mean is not None
        assert dataset.std is not None

        # Check data is normalized (approximately zero mean, unit std)
        all_data = np.array([dataset[i][0].numpy() for i in range(len(dataset))])
        assert np.abs(all_data.mean()) < 0.1
        assert np.abs(all_data.std() - 1.0) < 0.1


# ============================================================================
# Training Pipeline Tests
# ============================================================================

class TestTrainer:
    """Tests for training pipeline."""

    @pytest.fixture
    def setup_training(self):
        """Setup minimal training environment."""
        from ml_models.anomaly_detector.model import SensorAutoencoder
        from torch.utils.data import DataLoader, TensorDataset

        model = SensorAutoencoder(input_dim=4, latent_dim=2)

        # Create dummy data
        train_data = torch.randn(100, 4)
        val_data = torch.randn(20, 4)

        train_dataset = TensorDataset(train_data, train_data)
        val_dataset = TensorDataset(val_data, val_data)

        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=16)

        return {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader
        }

    def test_training_decreases_loss(self, setup_training):
        """Test that training decreases loss."""
        from Skills.pytorch.training_loop import Trainer

        trainer = Trainer(
            model=setup_training['model'],
            train_loader=setup_training['train_loader'],
            val_loader=setup_training['val_loader'],
            criterion=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(setup_training['model'].parameters()),
            device='cpu'
        )

        # Train for a few epochs
        history = trainer.train(epochs=5, verbose=False)

        # Check loss decreased
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        assert final_loss < initial_loss

    def test_checkpoint_saving(self, setup_training, tmp_path):
        """Test checkpoints are saved correctly."""
        from Skills.pytorch.training_loop import Trainer

        trainer = Trainer(
            model=setup_training['model'],
            train_loader=setup_training['train_loader'],
            val_loader=setup_training['val_loader'],
            criterion=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(setup_training['model'].parameters()),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )

        trainer.train(epochs=2, verbose=False)

        # Check checkpoint files exist
        assert (tmp_path / 'latest_checkpoint.pt').exists()
        assert (tmp_path / 'best_checkpoint.pt').exists()


# ============================================================================
# Model Behavior Tests
# ============================================================================

class TestModelBehavior:
    """Test expected model behaviors."""

    def test_anomaly_detector_on_normal_data(self):
        """Test model gives low scores for normal data."""
        from ml_models.anomaly_detector.model import SensorAutoencoder

        model = SensorAutoencoder(input_dim=4, latent_dim=2)
        model.eval()

        # Normal sensor readings (in typical range)
        normal_data = torch.tensor([[
            50.0,  # temperature
            1.0,   # vibration
            40.0,  # pressure
            200.0  # power
        ]])

        with torch.no_grad():
            error = model.get_reconstruction_error(normal_data).item()

        # For untrained model, can't assert exact values,
        # but error should be finite
        assert not np.isnan(error)
        assert not np.isinf(error)
        assert error >= 0

    def test_model_consistency(self):
        """Test model gives consistent results."""
        from ml_models.anomaly_detector.model import SensorAutoencoder

        model = SensorAutoencoder(input_dim=4, latent_dim=2)
        model.eval()

        input_data = torch.randn(5, 4)

        with torch.no_grad():
            output1 = model(input_data)
            output2 = model(input_data)

        # Same input should give same output in eval mode
        assert torch.allclose(output1, output2)


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndPipeline:
    """Test complete pipeline."""

    def test_full_training_inference_pipeline(self, tmp_path):
        """Test training and inference end-to-end."""
        from ml_models.anomaly_detector.model import SensorAutoencoder
        from torch.utils.data import DataLoader, TensorDataset
        from Skills.pytorch.training_loop import Trainer

        # 1. Create and train model
        model = SensorAutoencoder(input_dim=4, latent_dim=2)

        train_data = torch.randn(100, 4)
        train_dataset = TensorDataset(train_data, train_data)
        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(train_dataset, batch_size=16)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )

        trainer.train(epochs=2, verbose=False)

        # 2. Load checkpoint
        checkpoint_path = tmp_path / 'best_checkpoint.pt'
        assert checkpoint_path.exists()

        # 3. Load for inference
        inference_model = SensorAutoencoder(input_dim=4, latent_dim=2)
        checkpoint = torch.load(checkpoint_path)
        inference_model.load_state_dict(checkpoint['model_state_dict'])
        inference_model.eval()

        # 4. Run inference
        test_input = torch.randn(1, 4)
        with torch.no_grad():
            output = inference_model(test_input)

        assert output.shape == (1, 4)


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.slow
class TestModelPerformance:
    """Test model performance characteristics."""

    def test_inference_latency(self):
        """Test inference is fast enough for real-time."""
        from ml_models.anomaly_detector.model import SensorAutoencoder
        import time

        model = SensorAutoencoder(input_dim=4, latent_dim=2)
        model.eval()

        input_data = torch.randn(1, 4)

        # Warm up
        with torch.no_grad():
            _ = model(input_data)

        # Time inference
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(input_data)
        end = time.time()

        avg_latency_ms = (end - start) / 100 * 1000

        # Should be < 10ms for real-time requirements
        assert avg_latency_ms < 10, f"Latency too high: {avg_latency_ms:.2f}ms"

    def test_batch_processing_speedup(self):
        """Test batch processing is faster than sequential."""
        from ml_models.anomaly_detector.model import SensorAutoencoder
        import time

        model = SensorAutoencoder(input_dim=4, latent_dim=2)
        model.eval()

        batch_size = 32
        single_inputs = [torch.randn(1, 4) for _ in range(batch_size)]
        batch_input = torch.cat(single_inputs, dim=0)

        # Sequential processing
        start = time.time()
        with torch.no_grad():
            for inp in single_inputs:
                _ = model(inp)
        sequential_time = time.time() - start

        # Batch processing
        start = time.time()
        with torch.no_grad():
            _ = model(batch_input)
        batch_time = time.time() - start

        # Batch should be faster
        assert batch_time < sequential_time


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
```

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest tests/ --cov=ml_models --cov-report=html

# Run only fast tests (skip slow)
pytest tests/ -m "not slow"

# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_models.py::TestSensorAutoencoder::test_model_forward_pass
```

---

## pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
addopts =
    -v
    --strict-markers
    --tb=short
    --disable-warnings
```

---

## Common Pitfalls

### ❌ Not Using Fixtures
```python
# Wrong - duplicate setup code
def test_1():
    model = create_model()
    # test...

def test_2():
    model = create_model()  # Duplicate!
    # test...

# Correct - use fixture
@pytest.fixture
def model():
    return create_model()

def test_1(model):
    # test...

def test_2(model):
    # test...
```

### ❌ Tests Depend on Each Other
```python
# Wrong - tests should be independent
class TestModel:
    model = None

    def test_train(self):
        self.model = train_model()

    def test_inference(self):
        output = self.model(input)  # Depends on test_train!

# Correct - each test is independent
class TestModel:
    @pytest.fixture
    def trained_model(self):
        return train_model()

    def test_inference(self, trained_model):
        output = trained_model(input)
```

---

*Created: 2026-01-12*
*For: IndustrialMind All Months - Testing*
*Version: 1.0*
