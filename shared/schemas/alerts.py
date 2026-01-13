"""
Alert data models.

Defines Pydantic models for system alerts and notifications.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Alert(BaseModel):
    """System alert model."""

    timestamp: datetime = Field(..., description="UTC timestamp of alert")
    machine_id: str = Field(..., description="Machine identifier")

    # Alert classification
    alert_type: str = Field(
        ...,
        description="Type of alert: anomaly, maintenance, system, data_quality"
    )
    severity: str = Field(
        ...,
        description="Severity level: info, warning, error, critical"
    )

    # Alert details
    title: str = Field(..., min_length=1, max_length=200, description="Alert title")
    message: str = Field(..., min_length=1, description="Detailed alert message")

    # Context
    source: str = Field(..., description="Service that generated the alert")
    related_entity_id: Optional[str] = Field(
        None,
        description="Related entity (sensor_id, component_id, etc.)"
    )

    # Alert metadata
    metadata: Optional[Dict] = Field(
        None,
        description="Additional context (scores, thresholds, etc.)"
    )

    # Acknowledgement
    acknowledged: bool = Field(default=False, description="Whether alert is acknowledged")
    acknowledged_by: Optional[str] = Field(
        None,
        description="User who acknowledged the alert"
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="When alert was acknowledged"
    )

    # Resolution
    resolved: bool = Field(default=False, description="Whether issue is resolved")
    resolution_notes: Optional[str] = Field(
        None,
        description="Notes about resolution"
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="When issue was resolved"
    )

    @property
    def is_critical(self) -> bool:
        """Check if alert is critical."""
        return self.severity == "critical"

    @property
    def requires_action(self) -> bool:
        """Check if alert requires action."""
        return self.severity in ["error", "critical"] and not self.acknowledged

    def acknowledge(self, user: str) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()

    def resolve(self, notes: str) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolution_notes = notes
        self.resolved_at = datetime.utcnow()

    def to_kafka_message(self) -> Dict:
        """Convert to Kafka message format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "machine_id": self.machine_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "related_entity_id": self.related_entity_id,
            "metadata": self.metadata or {},
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:05Z",
                "machine_id": "MACHINE_001",
                "alert_type": "anomaly",
                "severity": "warning",
                "title": "Elevated Temperature Detected",
                "message": "Temperature sensor reading (85.2°C) exceeds normal operating range",
                "source": "anomaly-detection-service",
                "related_entity_id": "TEMP_001",
                "metadata": {
                    "anomaly_score": 0.087,
                    "threshold": 0.05,
                    "temperature": 85.2,
                    "normal_range": [60.0, 75.0]
                },
                "acknowledged": False,
                "resolved": False
            }
        }


class AlertBatch(BaseModel):
    """Batch of alerts."""

    alerts: List[Alert] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of alerts"
    )

    @property
    def batch_size(self) -> int:
        """Get number of alerts."""
        return len(self.alerts)

    @property
    def critical_count(self) -> int:
        """Count critical alerts."""
        return sum(1 for alert in self.alerts if alert.is_critical)

    @property
    def unacknowledged_count(self) -> int:
        """Count unacknowledged alerts."""
        return sum(1 for alert in self.alerts if not alert.acknowledged)

    def get_by_severity(self, severity: str) -> List[Alert]:
        """Filter alerts by severity."""
        return [alert for alert in self.alerts if alert.severity == severity]

    def get_by_machine(self, machine_id: str) -> List[Alert]:
        """Filter alerts by machine."""
        return [alert for alert in self.alerts if alert.machine_id == machine_id]

    class Config:
        json_schema_extra = {
            "example": {
                "alerts": [
                    {
                        "timestamp": "2024-01-15T10:30:05Z",
                        "machine_id": "MACHINE_001",
                        "alert_type": "anomaly",
                        "severity": "warning",
                        "title": "Elevated Temperature Detected",
                        "message": "Temperature exceeds normal range",
                        "source": "anomaly-detection-service"
                    }
                ]
            }
        }
