"""
Predictive maintenance data models.

Defines Pydantic models for maintenance predictions.
"""

from datetime import datetime
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class MaintenancePrediction(BaseModel):
    """Predictive maintenance forecast."""

    timestamp: datetime = Field(..., description="UTC timestamp of prediction")
    machine_id: str = Field(..., description="Machine identifier")

    # Prediction results
    predicted_rul_hours: float = Field(
        ...,
        ge=0.0,
        description="Predicted Remaining Useful Life in hours"
    )
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="95% confidence interval for RUL prediction"
    )

    # Maintenance recommendation
    recommended_action: str = Field(
        ...,
        description="Recommended maintenance action"
    )
    priority: str = Field(
        ...,
        description="Priority level: low, medium, high, critical"
    )
    estimated_downtime_hours: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated maintenance downtime"
    )

    # Model metadata
    model_version: str = Field(..., description="Version of the forecasting model")
    model_type: str = Field(default="transformer", description="Type of model")

    # Additional context
    failure_mode: Optional[str] = Field(
        None,
        description="Predicted failure mode"
    )
    contributing_factors: Optional[List[str]] = Field(
        None,
        description="Factors contributing to prediction"
    )

    @property
    def days_until_maintenance(self) -> float:
        """Convert RUL to days."""
        return self.predicted_rul_hours / 24.0

    @property
    def needs_immediate_attention(self) -> bool:
        """Check if immediate attention is needed."""
        return self.priority in ["high", "critical"] or self.predicted_rul_hours < 24.0

    def to_kafka_message(self) -> dict:
        """Convert to Kafka message format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "machine_id": self.machine_id,
            "predicted_rul_hours": self.predicted_rul_hours,
            "confidence_interval": list(self.confidence_interval),
            "recommended_action": self.recommended_action,
            "priority": self.priority,
            "estimated_downtime_hours": self.estimated_downtime_hours,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "failure_mode": self.failure_mode,
            "contributing_factors": self.contributing_factors or []
        }

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "machine_id": "MACHINE_001",
                "predicted_rul_hours": 48.5,
                "confidence_interval": [42.0, 55.0],
                "recommended_action": "schedule_maintenance",
                "priority": "medium",
                "estimated_downtime_hours": 4.0,
                "model_version": "transformer_v1.0.0",
                "model_type": "transformer",
                "failure_mode": "bearing_degradation",
                "contributing_factors": [
                    "elevated_temperature",
                    "increased_vibration"
                ]
            }
        }


class MaintenancePredictionBatch(BaseModel):
    """Batch of maintenance predictions."""

    predictions: List[MaintenancePrediction] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of maintenance predictions"
    )
    batch_size: int = Field(..., description="Number of predictions")
    inference_time_ms: float = Field(..., description="Total inference time in ms")

    @property
    def critical_machines(self) -> List[str]:
        """Get list of machines needing immediate attention."""
        return [
            pred.machine_id
            for pred in self.predictions
            if pred.needs_immediate_attention
        ]

    @property
    def num_critical(self) -> int:
        """Count critical predictions."""
        return len(self.critical_machines)

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "timestamp": "2024-01-15T10:30:00Z",
                        "machine_id": "MACHINE_001",
                        "predicted_rul_hours": 48.5,
                        "confidence_interval": [42.0, 55.0],
                        "recommended_action": "schedule_maintenance",
                        "priority": "medium",
                        "model_version": "transformer_v1.0.0"
                    }
                ],
                "batch_size": 1,
                "inference_time_ms": 45.2
            }
        }
