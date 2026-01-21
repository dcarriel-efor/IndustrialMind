"""
PyTorch Dataset for Industrial Sensor Time Series Data

Loads data from InfluxDB and prepares it for model training/inference.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from pathlib import Path


class SensorDataset(Dataset):
    """
    PyTorch Dataset for multivariate sensor time series.

    Features:
    - Temperature, Vibration, Pressure, Power (4 sensors)
    - Optional: Rolling statistics, time features
    - Automatic normalization (MinMaxScaler or StandardScaler)
    - Support for train/val/test splits
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str] = None,
        label_col: str = "is_anomaly",
        scaler: Optional[object] = None,
        fit_scaler: bool = True,
        scaler_type: str = "minmax"
    ):
        """
        Initialize dataset.

        Args:
            data: DataFrame with sensor readings
            feature_cols: List of feature column names (defaults to 4 sensors)
            label_col: Name of label column (0=normal, 1=anomaly)
            scaler: Pre-fitted scaler (for val/test sets)
            fit_scaler: Whether to fit scaler on this data (True for train)
            scaler_type: 'minmax' or 'standard'
        """
        self.data = data.copy()
        self.label_col = label_col

        # Default feature columns (4 sensors)
        if feature_cols is None:
            self.feature_cols = [
                "temperature",
                "vibration",
                "pressure",
                "power_consumption"
            ]
        else:
            self.feature_cols = feature_cols

        # Verify columns exist
        missing_cols = set(self.feature_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # Extract features and labels
        self.X = data[self.feature_cols].values

        if label_col in data.columns:
            self.y = data[label_col].values
        else:
            # If no labels, create dummy labels (for inference)
            self.y = np.zeros(len(data))

        # Initialize or use provided scaler
        if scaler is None:
            if scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            elif scaler_type == "standard":
                self.scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaler_type: {scaler_type}")
        else:
            self.scaler = scaler

        # Fit scaler on training data
        if fit_scaler:
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.X = self.scaler.transform(self.X)

        # Convert to tensors
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (features, label):
                - features: (4,) tensor of sensor values
                - label: 0 (normal) or 1 (anomaly)
        """
        return self.X[idx], self.y[idx]

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_cols

    def save_scaler(self, path: str):
        """Save fitted scaler for later use."""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {path}")

    @staticmethod
    def load_scaler(path: str):
        """Load pre-fitted scaler."""
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler


class TimeWindowDataset(Dataset):
    """
    Dataset with sliding time windows for temporal patterns.

    Useful for LSTM/Transformer models that need sequence history.
    For autoencoder, we use single time step (SensorDataset).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        window_size: int = 60,
        stride: int = 10,
        scaler: Optional[object] = None,
        fit_scaler: bool = True
    ):
        """
        Initialize windowed dataset.

        Args:
            data: DataFrame with sensor readings
            feature_cols: Feature column names
            window_size: Number of time steps per window (e.g., 60 = 1 minute)
            stride: Stride between windows (e.g., 10 = 10 seconds overlap)
            scaler: Pre-fitted scaler
            fit_scaler: Whether to fit scaler
        """
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.stride = stride

        # Extract features
        features = data[feature_cols].values

        # Fit/apply scaler
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler

        if fit_scaler:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        # Create sliding windows
        self.windows = []
        self.labels = []

        for i in range(0, len(features) - window_size + 1, stride):
            window = features[i:i + window_size]
            self.windows.append(window)

            # Label: 1 if any point in window is anomaly
            if "is_anomaly" in data.columns:
                label = data["is_anomaly"].iloc[i:i + window_size].max()
            else:
                label = 0
            self.labels.append(label)

        self.windows = torch.FloatTensor(np.array(self.windows))
        self.labels = torch.LongTensor(self.labels)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single window.

        Returns:
            Tuple of (window, label):
                - window: (window_size, num_features) tensor
                - label: 0 or 1
        """
        return self.windows[idx], self.labels[idx]


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 4,
    scaler_type: str = "minmax"
) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """
    Create train/val/test DataLoaders with consistent scaling.

    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        scaler_type: 'minmax' or 'standard'

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Create train dataset and fit scaler
    train_dataset = SensorDataset(
        train_df,
        scaler=None,
        fit_scaler=True,
        scaler_type=scaler_type
    )

    # Use same scaler for val and test
    val_dataset = SensorDataset(
        val_df,
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    test_dataset = SensorDataset(
        test_df,
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.scaler


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to sensor data.

    Features:
    - Rolling statistics (mean, std)
    - First-order differences
    - Cyclical time features (hour)

    Args:
        df: DataFrame with sensor readings

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # Rolling statistics (10-second windows)
    for col in ["temperature", "vibration", "pressure", "power_consumption"]:
        if col in df.columns:
            df[f"{col}_rolling_mean_10"] = df[col].rolling(window=10, min_periods=1).mean()
            df[f"{col}_rolling_std_10"] = df[col].rolling(window=10, min_periods=1).std().fillna(0)
            df[f"{col}_diff"] = df[col].diff().fillna(0)

    # Cyclical time features
    if "timestamp" in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


if __name__ == "__main__":
    # Test dataset creation
    print("Testing SensorDataset...")

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

    # Create dataset
    dataset = SensorDataset(dummy_data, fit_scaler=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature shape: {dataset[0][0].shape}")
    print(f"Label: {dataset[0][1]}")

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"\nBatch shape: {batch_x.shape}")
    print(f"Labels shape: {batch_y.shape}")
    print(f"Anomaly ratio in batch: {batch_y.float().mean():.2%}")

    # Test feature engineering
    print("\nTesting feature engineering...")
    df_enhanced = add_feature_engineering(dummy_data)
    print(f"Original columns: {len(dummy_data.columns)}")
    print(f"Enhanced columns: {len(df_enhanced.columns)}")
    print(f"New features: {set(df_enhanced.columns) - set(dummy_data.columns)}")
