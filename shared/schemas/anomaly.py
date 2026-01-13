"""
Anomaly detection data models.

Defines Pydantic models for anomaly detection results.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class AnomalyDetection(BaseModel):
    """Anomaly detection result for a single reading."""

    timestamp: datetime = Field(..., description="UTC timestamp of detection")
    machine_id: str = Field(..., description="Machine identifier")
    sensor_id: Optional[str] = Field(None, description="Sensor identifier")

    # Detection results
    is_anomaly: bool = Field(..., description="Whether an anomaly was detected")
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        description="Reconstruction error or anomaly score"
    )
    threshold: float = Field(..., ge=0.0, description="Detection threshold used")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the prediction"
    )

    # Feature contributions (optional)
    feature_contributions: Optional[Dict[str, float]] = Field(
        None,
        description="Contribution of each feature to anomaly score"
    )

    # Model metadata
    model_version: str = Field(..., description="Version of the model used")
    model_type: str = Field(default="autoencoder", description="Type of model")

    def to_kafka_message(self) -> Dict:
        """Convert to Kafka message format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "machine_id": self.machine_id,
            "sensor_id": self.sensor_id,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "feature_contributions": self.feature_contributions or {},
            "model_version": self.model_version,
            "model_type": self.model_type
        }

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:05Z",
                "machine_id": "MACHINE_001",
                "sensor_id": "TEMP_001",
                "anomaly_score": 0.087,
                "threshold": 0.05,
                "is_anomaly": True,
                "confidence": 0.92,
                "feature_contributions": {
                    "temperature": 0.65,
                    "vibration": 0.25,
                    "pressure": 0.05,
                    "power": 0.05
                },
                "model_version": "autoencoder_v1.2.0",
                "model_type": "autoencoder"
            }
        }


class AnomalyDetectionBatch(BaseModel):
    """Batch of anomaly detection results."""

    predictions: List[AnomalyDetection] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of anomaly predictions"
    )
    batch_size: int = Field(..., description="Number of predictions in batch")
    inference_time_ms: float = Field(..., description="Total inference time in milliseconds")

    @property
    def num_anomalies(self) -> int:
        """Count number of detected anomalies."""
        return sum(1 for pred in self.predictions if pred.is_anomaly)

    @property
    def anomaly_rate(self) -> float:
        """Calculate anomaly rate."""
        return self.num_anomalies / self.batch_size if self.batch_size > 0 else 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "timestamp": "2024-01-15T10:30:05Z",
                        "machine_id": "MACHINE_001",
                        "anomaly_score": 0.087,
                        "threshold": 0.05,
                        "is_anomaly": True,
                        "confidence": 0.92,
                        "model_version": "autoencoder_v1.2.0"
                    }
                ],
                "batch_size": 1,
                "inference_time_ms": 15.3
            }
        }
