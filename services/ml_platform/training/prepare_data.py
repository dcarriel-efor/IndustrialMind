"""
Data Preparation Pipeline for ML Training

Extracts sensor data from InfluxDB and prepares train/val/test splits.
"""

import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta
from typing import Tuple, Optional
import os
from pathlib import Path
import argparse


class DataPreparation:
    """Prepare training data from InfluxDB."""

    def __init__(
        self,
        influx_url: str = "http://localhost:8086",
        token: str = None,
        org: str = "industrialmind",
        bucket: str = "sensors"
    ):
        """
        Initialize InfluxDB connection.

        Args:
            influx_url: InfluxDB URL
            token: Auth token
            org: Organization name
            bucket: Bucket name
        """
        self.url = influx_url
        self.token = token or os.getenv("INFLUXDB_TOKEN", "industrialmind-token-123456")
        self.org = org
        self.bucket = bucket

        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.query_api = self.client.query_api()

    def query_sensor_data(
        self,
        start_time: str,
        end_time: str,
        machine_ids: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Query sensor data from InfluxDB.

        Args:
            start_time: Start time (ISO format or relative, e.g., '-7d')
            end_time: End time (ISO format or 'now()')
            machine_ids: List of machine IDs to filter (None = all)

        Returns:
            DataFrame with columns: timestamp, machine_id, temperature, vibration, pressure, power_consumption
        """
        # Build machine filter
        if machine_ids:
            machine_filter = " or ".join([f'r["machine_id"] == "{mid}"' for mid in machine_ids])
            machine_filter = f'|> filter(fn: (r) => {machine_filter})'
        else:
            machine_filter = ""

        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_time}, stop: {end_time})
          |> filter(fn: (r) => r["_measurement"] == "sensor_readings")
          {machine_filter}
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        print(f"Querying InfluxDB: {start_time} to {end_time}")
        result = self.query_api.query_data_frame(query)

        if result.empty:
            print("WARNING: No data returned from InfluxDB")
            return pd.DataFrame()

        # Rename columns
        df = result.rename(columns={
            "_time": "timestamp",
            "temperature": "temperature",
            "vibration": "vibration",
            "pressure": "pressure",
            "power": "power_consumption"
        })

        # Select relevant columns
        cols = ["timestamp", "machine_id", "temperature", "vibration", "pressure", "power_consumption"]
        df = df[cols]

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"Retrieved {len(df)} sensor readings")
        return df

    def label_anomalies(self, df: pd.DataFrame, method: str = "threshold") -> pd.DataFrame:
        """
        Label anomalies in the dataset.

        For Week 1 data, we need to infer anomalies from sensor values
        since we don't have ground truth labels yet.

        Methods:
        - 'threshold': Simple threshold-based (temp > 85, vib > 2.5, etc.)
        - 'statistical': Statistical outliers (3-sigma rule)

        Args:
            df: DataFrame with sensor readings
            method: Labeling method

        Returns:
            DataFrame with 'is_anomaly' column
        """
        df = df.copy()

        if method == "threshold":
            # Define thresholds based on simulator ranges
            anomaly_conditions = (
                (df["temperature"] > 85) |  # Failing temperature
                (df["vibration"] > 2.5) |   # Failing vibration
                (df["pressure"] < 30) | (df["pressure"] > 70) |  # Failing pressure
                (df["power_consumption"] > 400)  # Failing power
            )
            df["is_anomaly"] = anomaly_conditions.astype(int)

        elif method == "statistical":
            # 3-sigma rule for each sensor
            anomaly_mask = pd.Series(False, index=df.index)

            for col in ["temperature", "vibration", "pressure", "power_consumption"]:
                mean = df[col].mean()
                std = df[col].std()
                outliers = (df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)
                anomaly_mask |= outliers

            df["is_anomaly"] = anomaly_mask.astype(int)

        else:
            raise ValueError(f"Unknown labeling method: {method}")

        anomaly_rate = df["is_anomaly"].mean()
        print(f"Labeled {df['is_anomaly'].sum()} anomalies ({anomaly_rate:.2%} of data)")

        return df

    def create_train_val_test_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets (time-based).

        Important: Use time-based splits to avoid data leakage!

        Args:
            df: Full dataset
            train_ratio: Proportion for training (0.7 = 70%)
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        print(f"\nData splits:")
        print(f"  Train: {len(train_df)} samples ({len(train_df)/n:.1%})")
        print(f"  Val:   {len(val_df)} samples ({len(val_df)/n:.1%})")
        print(f"  Test:  {len(test_df)} samples ({len(test_df)/n:.1%})")

        # Print anomaly distribution
        print(f"\nAnomaly distribution:")
        print(f"  Train: {train_df['is_anomaly'].sum()} ({train_df['is_anomaly'].mean():.2%})")
        print(f"  Val:   {val_df['is_anomaly'].sum()} ({val_df['is_anomaly'].mean():.2%})")
        print(f"  Test:  {test_df['is_anomaly'].sum()} ({test_df['is_anomaly'].mean():.2%})")

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = "data/processed"
    ):
        """
        Save train/val/test splits to CSV files.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_path = output_path / "train.csv"
        val_path = output_path / "val.csv"
        test_path = output_path / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\nSaved data splits:")
        print(f"  Train: {train_path}")
        print(f"  Val:   {val_path}")
        print(f"  Test:  {test_path}")

        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "total_samples": len(train_df) + len(val_df) + len(test_df),
            "train_anomalies": int(train_df["is_anomaly"].sum()),
            "val_anomalies": int(val_df["is_anomaly"].sum()),
            "test_anomalies": int(test_df["is_anomaly"].sum()),
            "features": ["temperature", "vibration", "pressure", "power_consumption"],
            "time_range": {
                "start": str(train_df["timestamp"].min()),
                "end": str(test_df["timestamp"].max())
            }
        }

        import json
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Metadata: {metadata_path}")

    def close(self):
        """Close InfluxDB connection."""
        self.client.close()


def main():
    """Main data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare training data from InfluxDB")
    parser.add_argument("--start-time", type=str, default="-7d", help="Start time (e.g., '-7d', '2024-01-01T00:00:00Z')")
    parser.add_argument("--end-time", type=str, default="now()", help="End time")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test data ratio")
    parser.add_argument("--labeling-method", type=str, default="threshold", choices=["threshold", "statistical"], help="Anomaly labeling method")

    args = parser.parse_args()

    # Initialize data preparation
    prep = DataPreparation()

    try:
        # Query data from InfluxDB
        print("=" * 60)
        print("DATA PREPARATION PIPELINE")
        print("=" * 60)

        df = prep.query_sensor_data(
            start_time=args.start_time,
            end_time=args.end_time
        )

        if df.empty:
            print("ERROR: No data retrieved. Make sure:")
            print("  1. InfluxDB is running (docker compose ps)")
            print("  2. Data simulator has been running for at least 7 days")
            print("  3. Ingestion service is writing data")
            return

        # Label anomalies
        print("\n" + "=" * 60)
        print("LABELING ANOMALIES")
        print("=" * 60)
        df = prep.label_anomalies(df, method=args.labeling_method)

        # Create splits
        print("\n" + "=" * 60)
        print("CREATING TRAIN/VAL/TEST SPLITS")
        print("=" * 60)
        train_df, val_df, test_df = prep.create_train_val_test_splits(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )

        # Save to disk
        print("\n" + "=" * 60)
        print("SAVING DATA")
        print("=" * 60)
        prep.save_splits(train_df, val_df, test_df, output_dir=args.output_dir)

        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETE!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Verify data: ls -lh {args.output_dir}")
        print(f"  2. Train model: python train_autoencoder.py")

    finally:
        prep.close()


if __name__ == "__main__":
    main()
