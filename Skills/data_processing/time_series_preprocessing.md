# Time Series Preprocessing Pattern

## Purpose
Standard preprocessing patterns for time series sensor data, including cleaning, normalization, feature engineering, and windowing.

## When to Use
- Preparing sensor data for ML models
- Cleaning noisy industrial data
- Creating features from raw time series
- Handling missing values and outliers

## Prerequisites
- pandas, numpy installed
- Understanding of your data characteristics

---

## Template

```python
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datetime import datetime, timedelta

class TimeSeriesPreprocessor:
    """
    Preprocessing pipeline for time series sensor data.

    Example:
        preprocessor = TimeSeriesPreprocessor(
            features=['temperature', 'vibration', 'pressure', 'power'],
            freq='1s',
            interpolation_method='linear'
        )

        clean_df = preprocessor.fit_transform(raw_df)
    """

    def __init__(
        self,
        features: List[str],
        freq: str = '1s',
        interpolation_method: str = 'linear',
        outlier_std: float = 3.0
    ):
        """
        Args:
            features: List of feature column names
            freq: Resampling frequency (e.g., '1s', '5s', '1min')
            interpolation_method: Method for filling gaps ('linear', 'ffill', 'bfill')
            outlier_std: Standard deviations for outlier detection
        """
        self.features = features
        self.freq = freq
        self.interpolation_method = interpolation_method
        self.outlier_std = outlier_std

        # Fitted statistics
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'interpolate'
    ) -> pd.DataFrame:
        """
        Handle missing values in time series.

        Args:
            df: Input DataFrame with datetime index
            method: 'interpolate', 'ffill', 'bfill', or 'drop'
        """
        df = df.copy()

        if method == 'interpolate':
            # Interpolate using specified method
            df[self.features] = df[self.features].interpolate(
                method=self.interpolation_method,
                limit_direction='both'
            )
        elif method == 'ffill':
            df[self.features] = df[self.features].fillna(method='ffill')
        elif method == 'bfill':
            df[self.features] = df[self.features].fillna(method='bfill')
        elif method == 'drop':
            df = df.dropna(subset=self.features)
        else:
            raise ValueError(f"Unknown method: {method}")

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Remove or clip outliers.

        Args:
            df: Input DataFrame
            method: 'zscore' or 'iqr'
        """
        df = df.copy()

        for feature in self.features:
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
                df.loc[z_scores > self.outlier_std, feature] = np.nan

            elif method == 'iqr':
                # IQR method
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df.loc[(df[feature] < lower) | (df[feature] > upper), feature] = np.nan

        # Fill removed outliers
        df = self.handle_missing_values(df, method='interpolate')

        return df

    def resample_regular(
        self,
        df: pd.DataFrame,
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Resample to regular frequency.

        Args:
            df: Input DataFrame with datetime index
            aggregation: 'mean', 'median', 'first', 'last'
        """
        if aggregation == 'mean':
            resampled = df[self.features].resample(self.freq).mean()
        elif aggregation == 'median':
            resampled = df[self.features].resample(self.freq).median()
        elif aggregation == 'first':
            resampled = df[self.features].resample(self.freq).first()
        elif aggregation == 'last':
            resampled = df[self.features].resample(self.freq).last()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return resampled

    def normalize(
        self,
        df: pd.DataFrame,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Normalize features.

        Args:
            df: Input DataFrame
            method: 'standard' (z-score), 'minmax', or 'robust'
        """
        df = df.copy()

        if method == 'standard':
            # Z-score normalization
            if self.mean_ is None:
                self.mean_ = df[self.features].mean()
                self.std_ = df[self.features].std()

            df[self.features] = (df[self.features] - self.mean_) / (self.std_ + 1e-8)

        elif method == 'minmax':
            # Min-max scaling to [0, 1]
            if self.min_ is None:
                self.min_ = df[self.features].min()
                self.max_ = df[self.features].max()

            df[self.features] = (df[self.features] - self.min_) / (self.max_ - self.min_ + 1e-8)

        elif method == 'robust':
            # Robust scaling using median and IQR
            median = df[self.features].median()
            Q1 = df[self.features].quantile(0.25)
            Q3 = df[self.features].quantile(0.75)
            IQR = Q3 - Q1

            df[self.features] = (df[self.features] - median) / (IQR + 1e-8)

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()

        # Extract datetime components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month

        # Cyclical encoding for hour and day_of_week
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        lags: List[int] = [1, 5, 10]
    ) -> pd.DataFrame:
        """Add lagged features."""
        df = df.copy()

        for feature in self.features:
            for lag in lags:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 30]
    ) -> pd.DataFrame:
        """Add rolling statistics."""
        df = df.copy()

        for feature in self.features:
            for window in windows:
                df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window).mean()
                df[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window).std()
                df[f'{feature}_rolling_min_{window}'] = df[feature].rolling(window).min()
                df[f'{feature}_rolling_max_{window}'] = df[feature].rolling(window).max()

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        # 1. Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have datetime index or 'timestamp' column")

        # 2. Sort by time
        df = df.sort_index()

        # 3. Resample to regular frequency
        df = self.resample_regular(df, aggregation='mean')

        # 4. Handle missing values
        df = self.handle_missing_values(df, method='interpolate')

        # 5. Remove outliers
        df = self.remove_outliers(df, method='zscore')

        # 6. Normalize
        df = self.normalize(df, method='standard')

        # 7. Add features (optional)
        # df = self.add_temporal_features(df)
        # df = self.add_rolling_features(df)

        # 8. Drop remaining NaN (from rolling operations)
        df = df.dropna()

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted parameters."""
        if self.mean_ is None:
            raise ValueError("Must call fit_transform first!")

        return self.fit_transform(df)  # Reuse fit_transform logic
```

---

## Example Usage for IndustrialMind

### Basic Preprocessing

```python
import pandas as pd
from Skills.data_processing.time_series_preprocessing import TimeSeriesPreprocessor

# Load raw sensor data
raw_df = pd.read_csv('sensor_data.csv', parse_dates=['timestamp'])

# Initialize preprocessor
preprocessor = TimeSeriesPreprocessor(
    features=['temperature', 'vibration', 'pressure', 'power'],
    freq='1s',  # Resample to 1 second intervals
    interpolation_method='linear',
    outlier_std=3.0
)

# Preprocess
clean_df = preprocessor.fit_transform(raw_df)

print(f"Original shape: {raw_df.shape}")
print(f"Clean shape: {clean_df.shape}")
print(f"Missing values: {clean_df.isnull().sum().sum()}")
```

### With Feature Engineering

```python
# Preprocess and add features
preprocessor = TimeSeriesPreprocessor(
    features=['temperature', 'vibration', 'pressure', 'power'],
    freq='1s'
)

df = preprocessor.fit_transform(raw_df)

# Add temporal features
df = preprocessor.add_temporal_features(df)

# Add rolling statistics (5, 10, 30 second windows)
df = preprocessor.add_rolling_features(df, windows=[5, 10, 30])

# Drop NaN from rolling operations
df = df.dropna()

print(f"Features after engineering: {df.shape[1]}")
print(f"New features: {[c for c in df.columns if c not in preprocessor.features]}")
```

---

## Common Pitfalls

### ❌ Not Sorting by Time
```python
# Wrong - unsorted data causes issues
df = df.resample('1s').mean()  # Incorrect results!

# Correct
df = df.sort_index()
df = df.resample('1s').mean()
```

### ❌ Leaking Future Information
```python
# Wrong - using future data in rolling window
df['rolling_mean'] = df['temperature'].rolling(window=10, center=True).mean()

# Correct - only use past data
df['rolling_mean'] = df['temperature'].rolling(window=10, center=False).mean()
```

### ❌ Not Handling Timezone Issues
```python
# Wrong - mixed timezones cause errors
df.index = pd.to_datetime(df.timestamp)  # May have mixed timezones!

# Correct - standardize timezone
df.index = pd.to_datetime(df.timestamp).tz_localize('UTC')
```

---

## References

- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Time Series Feature Engineering](https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf)

---

*Created: 2026-01-12*
*For: IndustrialMind Month 1-2 - Data Preprocessing*
*Version: 1.0*
