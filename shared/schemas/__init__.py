"""
Shared Pydantic schemas for IndustrialMind platform.

This module contains all data models used across services for:
- Kafka message schemas
- API request/response models
- Database models
- Configuration models
"""

from .sensor_reading import (
    SensorReading,
    SensorReadingBatch,
    SensorMetadata,
)
from .anomaly import (
    AnomalyDetection,
    AnomalyDetectionBatch,
)
from .maintenance import (
    MaintenancePrediction,
    MaintenancePredictionBatch,
)
from .alerts import (
    Alert,
    AlertBatch,
)

__all__ = [
    # Sensor readings
    "SensorReading",
    "SensorReadingBatch",
    "SensorMetadata",
    # Anomaly detection
    "AnomalyDetection",
    "AnomalyDetectionBatch",
    # Maintenance predictions
    "MaintenancePrediction",
    "MaintenancePredictionBatch",
    # Alerts
    "Alert",
    "AlertBatch",
]
