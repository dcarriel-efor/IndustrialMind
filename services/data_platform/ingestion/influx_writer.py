"""InfluxDB writer for batch writing sensor readings."""

from typing import List
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import structlog

# Import from shared schemas
from shared.schemas.sensor_reading import SensorReading

logger = structlog.get_logger()


class InfluxDBWriter:
    """Writes sensor readings to InfluxDB with batching."""

    def __init__(self, url: str, token: str, org: str, bucket: str):
        """Initialize InfluxDB writer.

        Args:
            url: InfluxDB URL (e.g., "http://influxdb:8086")
            token: Authentication token
            org: Organization name
            bucket: Bucket name for storing data
        """
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = bucket
        self.org = org

        logger.info(
            "InfluxDB writer initialized",
            url=url,
            org=org,
            bucket=bucket
        )

    def write_batch(self, readings: List[SensorReading]) -> bool:
        """Convert readings to InfluxDB Points and write batch.

        Args:
            readings: List of SensorReading objects to write

        Returns:
            True if write was successful, False otherwise
        """
        if not readings:
            logger.debug("Empty batch, skipping write")
            return True

        points = [self._reading_to_point(r) for r in readings]

        try:
            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=points
            )
            logger.info(
                "Batch written to InfluxDB",
                count=len(points),
                bucket=self.bucket
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to write batch to InfluxDB",
                error=str(e),
                count=len(points),
                bucket=self.bucket
            )
            return False

    def _reading_to_point(self, reading: SensorReading) -> Point:
        """Convert SensorReading to InfluxDB Point format.

        Args:
            reading: SensorReading object

        Returns:
            InfluxDB Point object
        """
        # Parse timestamp - handle both string and datetime objects
        if isinstance(reading.timestamp, str):
            timestamp = datetime.fromisoformat(reading.timestamp.replace('Z', '+00:00'))
        else:
            timestamp = reading.timestamp

        point = (
            Point("sensor_readings")
            .tag("machine_id", reading.machine_id)
            .tag("sensor_id", reading.sensor_id)
            .field("temperature", float(reading.temperature))
            .field("vibration", float(reading.vibration))
            .field("pressure", float(reading.pressure))
            .field("power", float(reading.power))
            .time(timestamp, WritePrecision.NS)
        )

        return point

    def close(self):
        """Close the InfluxDB client gracefully."""
        try:
            self.client.close()
            logger.info("InfluxDB client closed")
        except Exception as e:
            logger.error("Error closing InfluxDB client", error=str(e))
