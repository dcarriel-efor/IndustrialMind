"""
Sensor reading data models.

Defines Pydantic models for sensor data throughout the platform.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, ConfigDict


class SensorMetadata(BaseModel):
    """Metadata associated with sensor readings."""

    location: str = Field(..., description="Physical location of the sensor")
    shift: Optional[str] = Field(None, description="Production shift (morning/afternoon/night)")
    operator: Optional[str] = Field(None, description="Operator ID or name")
    production_line: Optional[str] = Field(None, description="Production line identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Factory_A_Line_1",
                "shift": "morning",
                "operator": "OP_001",
                "production_line": "LINE_1"
            }
        }


class SensorReading(BaseModel):
    """Single sensor reading with timestamp and measurements."""

    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of the reading"
    )
    machine_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique identifier for the machine"
    )
    sensor_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique identifier for the sensor"
    )

    # Sensor measurements
    temperature: float = Field(
        ...,
        ge=-50.0,
        le=200.0,
        description="Temperature in Celsius"
    )
    vibration: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        description="Vibration in mm/s"
    )
    pressure: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Pressure in bar"
    )
    power: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Power consumption in kW"
    )

    # Optional metadata
    metadata: Optional[SensorMetadata] = Field(
        None,
        description="Additional metadata about the reading"
    )

    @validator('timestamp')
    def timestamp_not_future(cls, v):
        """Ensure timestamp is not in the future."""
        # Convert to offset-naive for comparison if needed
        now = datetime.now(timezone.utc)
        v_naive = v.replace(tzinfo=None) if v.tzinfo else v
        now_naive = now.replace(tzinfo=None)
        if v_naive > now_naive:
            raise ValueError('Timestamp cannot be in the future')
        return v

    @validator('machine_id', 'sensor_id')
    def ids_valid_format(cls, v):
        """Validate ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('IDs must be alphanumeric with optional _ or -')
        return v

    def to_dict(self) -> Dict:
        """Convert to dictionary for Kafka serialization."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "machine_id": self.machine_id,
            "sensor_id": self.sensor_id,
            "readings": {
                "temperature": self.temperature,
                "vibration": self.vibration,
                "pressure": self.pressure,
                "power": self.power
            }
        }
        if self.metadata:
            data["metadata"] = self.metadata.dict()
        return data

    def to_influx_point(self) -> Dict:
        """Convert to InfluxDB point format."""
        return {
            "measurement": "sensor_readings",
            "tags": {
                "machine_id": self.machine_id,
                "sensor_id": self.sensor_id,
                "location": self.metadata.location if self.metadata else "unknown",
            },
            "fields": {
                "temperature": self.temperature,
                "vibration": self.vibration,
                "pressure": self.pressure,
                "power": self.power
            },
            "time": self.timestamp
        }

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "machine_id": "MACHINE_001",
                "sensor_id": "TEMP_001",
                "temperature": 65.5,
                "vibration": 1.2,
                "pressure": 45.0,
                "power": 250.0,
                "metadata": {
                    "location": "Factory_A_Line_1",
                    "shift": "morning"
                }
            }
        }
    )


class SensorReadingBatch(BaseModel):
    """Batch of sensor readings."""

    readings: List[SensorReading] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of sensor readings"
    )

    @property
    def batch_size(self) -> int:
        """Get the number of readings in the batch."""
        return len(self.readings)

    def to_dict_list(self) -> List[Dict]:
        """Convert all readings to dictionaries."""
        return [reading.to_dict() for reading in self.readings]

    def to_influx_points(self) -> List[Dict]:
        """Convert all readings to InfluxDB points."""
        return [reading.to_influx_point() for reading in self.readings]

    class Config:
        json_schema_extra = {
            "example": {
                "readings": [
                    {
                        "timestamp": "2024-01-15T10:30:00Z",
                        "machine_id": "MACHINE_001",
                        "sensor_id": "TEMP_001",
                        "temperature": 65.5,
                        "vibration": 1.2,
                        "pressure": 45.0,
                        "power": 250.0
                    }
                ]
            }
        }
