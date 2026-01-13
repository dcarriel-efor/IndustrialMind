# PyTorch Custom Dataset Pattern

## Purpose
Create custom PyTorch Dataset classes for loading and preprocessing data, specifically for time series sensor data.

## When to Use
- Loading data from databases (InfluxDB, PostgreSQL)
- Custom data preprocessing needed
- Time series windowing required
- Need efficient batching

## Prerequisites
- PyTorch installed
- Data source available (InfluxDB, files, etc.)
- Understanding of your data structure

---

## Template

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List
import pandas as pd

class TimeSeriesDataset(Dataset):
    """
    Base class for time series datasets.

    Example:
        dataset = SensorDataset(
            data_source=influxdb_client,
            time_range="2024-01-01/2024-02-01",
            window_size=50,
            features=['temperature', 'vibration', 'pressure', 'power']
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        window_size: int = 1,
        stride: int = 1,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data: Input data (N, features)
            labels: Optional labels (N,) or (N, label_dim)
            window_size: Number of timesteps per sample
            stride: Step size between windows
            transform: Optional transform to apply
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.transform = transform

        # Calculate valid indices for windowing
        self.valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self) -> List[int]:
        """Compute valid starting indices for windows."""
        max_start = len(self.data) - self.window_size + 1
        return list(range(0, max_start, self.stride))

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get one sample."""
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size

        # Extract window
        window = self.data[start_idx:end_idx]

        # Apply transform if provided
        if self.transform:
            window = self.transform(window)

        # Convert to tensor
        window_tensor = torch.tensor(window, dtype=torch.float32)

        # Return with or without labels
        if self.labels is not None:
            # For sequence labeling, get label at end of window
            label = self.labels[end_idx - 1]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return window_tensor, label_tensor
        else:
            return window_tensor

class SensorDataset(TimeSeriesDataset):
    """
    Dataset for sensor readings from InfluxDB.

    Loads data from InfluxDB and creates windowed samples
    for time series analysis.
    """

    def __init__(
        self,
        influxdb_client,
        bucket: str,
        measurement: str,
        time_range: str,
        features: List[str],
        machine_id: Optional[str] = None,
        window_size: int = 50,
        stride: int = 1,
        normalize: bool = True
    ):
        """
        Args:
            influxdb_client: InfluxDB client instance
            bucket: InfluxDB bucket name
            measurement: Measurement name
            time_range: Time range (e.g., "2024-01-01/2024-02-01")
            features: List of field names to extract
            machine_id: Optional machine ID filter
            window_size: Timesteps per sample
            stride: Step between windows
            normalize: Whether to normalize features
        """
        self.client = influxdb_client
        self.bucket = bucket
        self.measurement = measurement
        self.features = features

        # Load data from InfluxDB
        data, labels = self._load_from_influx(
            time_range, machine_id
        )

        # Normalize if requested
        if normalize:
            data, self.mean, self.std = self._normalize(data)
        else:
            self.mean = None
            self.std = None

        # Initialize parent
        super().__init__(
            data=data,
            labels=labels,
            window_size=window_size,
            stride=stride
        )

    def _load_from_influx(
        self,
        time_range: str,
        machine_id: Optional[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load data from InfluxDB."""
        # Build Flux query
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: {time_range.split('/')[0]},
                     stop: {time_range.split('/')[1]})
            |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
        '''

        if machine_id:
            query += f'''
            |> filter(fn: (r) => r["machine_id"] == "{machine_id}")
            '''

        query += '''
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        # Execute query
        tables = self.client.query_api().query(query)

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                row = {feat: record.values.get(feat) for feat in self.features}
                row['state'] = record.values.get('state', None)
                records.append(row)

        df = pd.DataFrame(records)

        # Extract features and labels
        data = df[self.features].values
        labels = df['state'].values if 'state' in df.columns else None

        # Convert categorical labels to numeric
        if labels is not None:
            label_map = {'normal': 0, 'degrading': 1, 'failing': 2}
            labels = np.array([label_map.get(l, 0) for l in labels])

        return data, labels

    def _normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize data to zero mean and unit variance."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8  # Avoid division by zero
        normalized = (data - mean) / std
        return normalized, mean, std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        if self.mean is None or self.std is None:
            return data
        return data * self.std + self.mean


class AnomalyDetectionDataset(Dataset):
    """
    Dataset for autoencoder-based anomaly detection.

    For autoencoders, input = target (reconstruction task).
    """

    def __init__(
        self,
        data: np.ndarray,
        anomaly_labels: Optional[np.ndarray] = None,
        normalize: bool = True
    ):
        """
        Args:
            data: Input features (N, features)
            anomaly_labels: Binary labels (0=normal, 1=anomaly)
            normalize: Whether to normalize
        """
        self.data = data
        self.anomaly_labels = anomaly_labels

        if normalize:
            self.data, self.mean, self.std = self._normalize(data)
        else:
            self.mean = None
            self.std = None

    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std, mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)

        # For autoencoder: input = target
        if self.anomaly_labels is not None:
            label = torch.tensor(self.anomaly_labels[idx], dtype=torch.float32)
            return sample, sample, label  # (input, target, anomaly_label)
        else:
            return sample, sample  # (input, target)
```

---

## Example Usage for IndustrialMind

### Loading Sensor Data from InfluxDB

```python
from influxdb_client import InfluxDBClient

# Initialize InfluxDB client
client = InfluxDBClient(
    url="http://localhost:8086",
    token="your-token",
    org="your-org"
)

# Create dataset
dataset = SensorDataset(
    influxdb_client=client,
    bucket="industrial_sensors",
    measurement="sensor_readings",
    time_range="2024-01-01T00:00:00Z/2024-02-01T00:00:00Z",
    features=['temperature', 'vibration', 'pressure', 'power'],
    machine_id="MACHINE_001",
    window_size=50,  # 50 timesteps per sample
    stride=10,       # Slide window by 10 steps
    normalize=True
)

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch_idx, (windows, labels) in enumerate(train_loader):
    # windows shape: (batch_size, window_size, num_features)
    # labels shape: (batch_size,)
    print(f"Batch {batch_idx}: {windows.shape}, {labels.shape}")
```

### For Autoencoder Training

```python
# Load normal data for training
normal_data = load_normal_sensor_data()  # Your data loading function

# Create dataset
train_dataset = AnomalyDetectionDataset(
    data=normal_data,
    normalize=True
)

# Split into train/val
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# Train autoencoder
for epoch in range(epochs):
    for input_batch, target_batch in train_loader:
        # input_batch == target_batch for autoencoder
        output = model(input_batch)
        loss = criterion(output, target_batch)
        # ... training step
```

---

## Variations

### With Data Augmentation

```python
class AugmentedSensorDataset(SensorDataset):
    def __init__(self, *args, noise_std: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_std = noise_std

    def __getitem__(self, idx):
        window, label = super().__getitem__(idx)

        # Add Gaussian noise for augmentation
        if self.training:
            noise = torch.randn_like(window) * self.noise_std
            window = window + noise

        return window, label
```

### With Caching for Speed

```python
class CachedSensorDataset(SensorDataset):
    def __init__(self, *args, cache: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = cache
        self._cache_dict = {} if cache else None

    def __getitem__(self, idx):
        if self.cache and idx in self._cache_dict:
            return self._cache_dict[idx]

        sample = super().__getitem__(idx)

        if self.cache:
            self._cache_dict[idx] = sample

        return sample
```

---

## Common Pitfalls

### ❌ Off-by-One Errors in Windowing
```python
# Wrong - can go out of bounds
end_idx = start_idx + self.window_size
window = self.data[start_idx:end_idx]  # May be incomplete!

# Correct - validate indices
max_start = len(self.data) - self.window_size + 1
valid_indices = range(0, max_start, self.stride)
```

### ❌ Not Handling Missing Data
```python
# Wrong - NaN will propagate through model
data = df.values  # May contain NaN

# Correct - handle missing values
data = df.fillna(method='ffill').fillna(method='bfill').values
```

### ❌ Memory Issues with Large Datasets
```python
# Wrong - load all data at once
all_data = load_entire_database()  # OOM!

# Correct - lazy loading or chunking
def __getitem__(self, idx):
    # Load only what's needed for this sample
    return self._load_sample(idx)
```

---

## References

- [PyTorch Dataset Documentation](https://pytorch.org/docs/stable/data.html)
- [Time Series Datasets](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [InfluxDB Python Client](https://docs.influxdata.com/influxdb/v2/api-guide/client-libraries/python/)

---

*Created: 2026-01-12*
*For: IndustrialMind Month 2 - Data Loading*
*Version: 1.0*
