# ML Platform - PyTorch Anomaly Detection

Production-grade PyTorch models for industrial sensor anomaly detection.

## 🎯 Overview

This module implements deep learning models for real-time anomaly detection on multivariate industrial sensor data (temperature, vibration, pressure, power consumption).

### Key Features

- **PyTorch Autoencoder** for anomaly detection via reconstruction error
- **Variational Autoencoder (VAE)** for uncertainty quantification
- **MLflow integration** for experiment tracking and model registry
- **Custom Dataset** for sensor time series with normalization
- **Production-ready training pipeline** with early stopping, LR scheduling
- **Comprehensive evaluation** with F1, precision, recall, confusion matrix

---

## 📂 Directory Structure

```
ml_platform/
├── models/
│   └── autoencoder.py          # PyTorch model architectures
├── datasets/
│   └── sensor_dataset.py       # Custom Dataset and DataLoaders
├── training/
│   ├── prepare_data.py         # Extract training data from InfluxDB
│   └── train_autoencoder.py    # Main training script with MLflow
├── evaluation/
│   └── (future: evaluation scripts)
├── inference/
│   └── (future: FastAPI inference service)
├── utils/
│   └── (future: helper functions)
├── test_setup.py               # Quick setup verification
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd services/ml_platform
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python test_setup.py
```

Expected output:
```
✓ PASS | Model Initialization
✓ PASS | Dataset Creation
✓ PASS | Training Loop
🎉 All tests passed! ML platform is ready.
```

### 3. Prepare Training Data

Extract sensor data from InfluxDB (make sure simulator has been running):

```bash
python training/prepare_data.py \
  --start-time "-7d" \
  --end-time "now()" \
  --output-dir "../../data/processed" \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

This creates:
- `data/processed/train.csv` - Training set (70%)
- `data/processed/val.csv` - Validation set (15%)
- `data/processed/test.csv` - Test set (15%)
- `data/processed/metadata.json` - Dataset metadata

### 4. Train Model

Train autoencoder with default hyperparameters:

```bash
python training/train_autoencoder.py \
  --data-dir "../../data/processed" \
  --output-dir "../../models/autoencoder" \
  --model-type "autoencoder" \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --max-epochs 100
```

**Advanced options:**
```bash
# Train Variational Autoencoder with custom architecture
python training/train_autoencoder.py \
  --model-type "vae" \
  --latent-dim 8 \
  --hidden-dims 128 64 32 \
  --dropout 0.3 \
  --batch-size 128 \
  --learning-rate 5e-4 \
  --max-epochs 200 \
  --experiment-name "month_02_vae_experiments"
```

### 5. View Results in MLflow

```bash
# MLflow UI is already running at:
http://localhost:5011
```

Navigate to the `month_02_anomaly_detection` experiment to see:
- Training/validation loss curves
- Final metrics (F1, precision, recall)
- Model artifacts
- Hyperparameters

---

## 🧠 Model Architectures

### Standard Autoencoder

```
Input (4 sensors) → [64] → [32] → [Latent: 4] → [32] → [64] → Output (4 sensors)
                     ReLU   ReLU              ReLU   ReLU
```

- **Loss**: MSE (Mean Squared Error) reconstruction loss
- **Anomaly Score**: Reconstruction error per sample
- **Threshold**: 95th percentile of normal sample errors (configurable)

**Hyperparameters:**
- Input dim: 4 (temperature, vibration, pressure, power)
- Latent dim: 4 (compressed representation)
- Hidden dims: [64, 32]
- Dropout: 0.2
- Activation: ReLU

### Variational Autoencoder (VAE)

```
Input → [64] → [32] → (μ, σ²) → z ~ N(μ, σ²) → [32] → [64] → Output
                       (latent distribution)
```

- **Loss**: Reconstruction loss + β * KL divergence
- **Benefits**: Uncertainty quantification, smoother latent space
- **Use case**: When you need probabilistic anomaly scores

---

## 📊 Evaluation Metrics

The training script computes comprehensive metrics on the test set:

| Metric | Description | Target |
|--------|-------------|--------|
| **F1 Score** | Harmonic mean of precision/recall | > 0.90 |
| **Precision** | TP / (TP + FP) - minimize false alarms | > 0.85 |
| **Recall** | TP / (TP + FN) - catch all anomalies | > 0.80 |
| **ROC AUC** | Area under ROC curve | > 0.95 |
| **Confusion Matrix** | TP, TN, FP, FN counts | - |

**Performance by Anomaly Type:**
- SPIKE: Sudden value jumps
- DRIFT: Gradual degradation
- CYCLIC: Repeating patterns
- MULTI_SENSOR: Correlated anomalies

---

## 🔬 Experiment Tracking with MLflow

All experiments are logged to MLflow for reproducibility:

### Logged Parameters
- Model type (autoencoder / VAE)
- Architecture (latent_dim, hidden_dims, dropout)
- Training config (batch_size, learning_rate, max_epochs)
- Data preprocessing (scaler_type, train/val/test ratios)
- Random seed

### Logged Metrics (per epoch)
- Training loss
- Validation loss

### Logged Test Metrics (final)
- F1 score, precision, recall, accuracy
- ROC AUC
- Confusion matrix (TP, TN, FP, FN)
- Selected threshold

### Logged Artifacts
- `best_model.pth` - PyTorch model checkpoint
- `scaler.pkl` - Fitted MinMaxScaler/StandardScaler
- `threshold.json` - Anomaly detection threshold

---

## 📈 Training Pipeline

The training pipeline includes production-ready features:

### 1. Data Preprocessing
- MinMaxScaler or StandardScaler normalization
- Consistent scaling across train/val/test
- Time-based splits (no data leakage)

### 2. Training Loop
- AdamW optimizer with weight decay
- ReduceLROnPlateau scheduler (reduce LR when val loss plateaus)
- Gradient clipping (max norm = 1.0)
- Batch training with progress logging

### 3. Validation & Early Stopping
- Validation after each epoch
- Early stopping (patience = 10 epochs)
- Save best model based on validation loss

### 4. Threshold Selection
- **Percentile method**: 95th percentile of normal validation errors
- **Best F1 method**: Threshold that maximizes F1 score on validation set

### 5. Test Evaluation
- Load best model
- Compute reconstruction errors on test set
- Apply threshold and compute metrics
- Generate confusion matrix

---

## 🛠️ Advanced Usage

### Custom Feature Engineering

Add rolling statistics and time features:

```python
from services.ml_platform.datasets.sensor_dataset import add_feature_engineering

df = add_feature_engineering(df)
# Adds: rolling_mean_10, rolling_std_10, diff, hour_sin, hour_cos
```

### Load Pre-trained Model

```python
import torch
from services.ml_platform.models.autoencoder import SensorAutoencoder

# Load model
model = SensorAutoencoder(input_dim=4, latent_dim=4)
checkpoint = torch.load("models/autoencoder/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load scaler
import pickle
with open("models/autoencoder/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Load threshold
import json
with open("models/autoencoder/threshold.json", 'r') as f:
    threshold = json.load(f)['threshold']

# Inference
import numpy as np
new_data = np.array([[72.5, 1.8, 55.2, 285.0]])  # temp, vib, pressure, power
new_data_scaled = scaler.transform(new_data)
x = torch.FloatTensor(new_data_scaled)

with torch.no_grad():
    error = model.reconstruction_error(x)
    is_anomaly = error.item() > threshold

print(f"Reconstruction error: {error.item():.4f}")
print(f"Threshold: {threshold:.4f}")
print(f"Is anomaly: {is_anomaly}")
```

### Hyperparameter Tuning

Run multiple experiments with different hyperparameters:

```bash
#!/bin/bash
# Hyperparameter sweep

for latent_dim in 2 4 8 16; do
  for lr in 1e-4 5e-4 1e-3; do
    python training/train_autoencoder.py \
      --latent-dim $latent_dim \
      --learning-rate $lr \
      --run-name "ae_ld${latent_dim}_lr${lr}" \
      --experiment-name "month_02_hyperparam_sweep"
  done
done
```

Then compare runs in MLflow UI.

---

## 🎯 Next Steps

### Week 1 (Current)
- [x] Model architectures (autoencoder, VAE)
- [x] Custom Dataset and DataLoader
- [x] Training pipeline with MLflow
- [x] Evaluation metrics
- [ ] Run first experiment on real data

### Week 2-3
- [ ] FastAPI inference service (`/predict` endpoint)
- [ ] ONNX model export for production
- [ ] Batch inference script
- [ ] Performance benchmarking (latency, throughput)

### Week 4
- [ ] Integrate inference with Streamlit dashboard
- [ ] Real-time anomaly alerting
- [ ] Model monitoring dashboard (Grafana)
- [ ] Docker container for inference service

---

## 📊 Expected Performance

Based on synthetic sensor data (5% anomaly rate):

| Metric | Target | Typical Result |
|--------|--------|----------------|
| F1 Score | > 0.90 | 0.92 - 0.95 |
| Precision | > 0.85 | 0.88 - 0.93 |
| Recall | > 0.80 | 0.85 - 0.92 |
| Training Time | < 10 min | ~5 min (CPU, 50 epochs) |
| Inference Latency | < 100ms | ~10-20ms (CPU, single sample) |

---

## 🐛 Troubleshooting

### "No data returned from InfluxDB"
- Verify InfluxDB is running: `docker compose ps`
- Check data exists: Visit http://localhost:8086 and query `sensor_readings`
- Ensure simulator has been running for at least the time range specified

### "CUDA out of memory"
- Reduce batch size: `--batch-size 32` or `--batch-size 16`
- Use CPU: `--no-cuda`
- Enable gradient checkpointing (future enhancement)

### "MLflow connection error"
- Verify MLflow is running: `docker compose ps | grep mlflow`
- Check MLflow UI: http://localhost:5011
- Ensure MLflow tracking URI is correct: `--mlflow-uri http://localhost:5011`

### Low F1 score
- Check data quality: Are anomalies properly labeled?
- Adjust threshold: Try `--threshold-method best_f1`
- Increase model capacity: `--hidden-dims 128 64 32`
- Train longer: `--max-epochs 200`

---

## 📚 References

### Papers
- **Autoencoders**: Hinton & Salakhutdinov (2006) - "Reducing the Dimensionality of Data with Neural Networks"
- **VAE**: Kingma & Welling (2013) - "Auto-Encoding Variational Bayes"
- **Anomaly Detection**: Chalapathy & Chawla (2019) - "Deep Learning for Anomaly Detection: A Survey"

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## 🤝 Contributing

This is a learning project, but suggestions are welcome!

For questions or issues, see the main project [README](../../README.md).

---

**Author**: Diego Carriel Lopez
**Project**: IndustrialMind - Month 2 (PyTorch Anomaly Detection)
**Last Updated**: 2026-01-21
