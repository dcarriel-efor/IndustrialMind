"""Kafka consumer for sensor readings with Pydantic validation."""

import json
from typing import List
from kafka import KafkaConsumer
from pydantic import ValidationError
import structlog

# Import from shared schemas
from shared.schemas.sensor_reading import SensorReading

logger = structlog.get_logger()


class SensorReadingConsumer:
    """Consumes sensor readings from Kafka with validation."""

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str):
        """Initialize Kafka consumer.

        Args:
            bootstrap_servers: Kafka bootstrap servers (e.g., "kafka:29092")
            topic: Topic name to consume from
            group_id: Consumer group ID for offset management
        """
        self.topic = topic
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',  # Start from latest on first run
            enable_auto_commit=False,  # Manual commit after successful write
            max_poll_records=100,  # Batch size
            consumer_timeout_ms=1000  # Return empty after 1 second if no messages
        )
        logger.info(
            "Kafka consumer initialized",
            topic=topic,
            group_id=group_id,
            bootstrap_servers=bootstrap_servers
        )

    def consume_batch(self, timeout_ms: int = 1000) -> List[SensorReading]:
        """Consume batch of messages and validate with Pydantic.

        Args:
            timeout_ms: Timeout for polling in milliseconds

        Returns:
            List of validated SensorReading objects
        """
        messages = self.consumer.poll(timeout_ms=timeout_ms)

        validated_readings = []
        invalid_count = 0

        for topic_partition, records in messages.items():
            for record in records:
                try:
                    # Validate with Pydantic schema
                    reading = SensorReading(**record.value)
                    validated_readings.append(reading)
                except ValidationError as e:
                    invalid_count += 1
                    logger.error(
                        "Invalid message skipped",
                        error=str(e),
                        value=record.value,
                        partition=topic_partition.partition,
                        offset=record.offset
                    )
                    # Skip invalid messages, continue processing
                except Exception as e:
                    invalid_count += 1
                    logger.error(
                        "Unexpected error processing message",
                        error=str(e),
                        partition=topic_partition.partition,
                        offset=record.offset
                    )

        if validated_readings:
            logger.info(
                "Batch consumed",
                valid_count=len(validated_readings),
                invalid_count=invalid_count
            )

        return validated_readings

    def commit(self):
        """Commit offsets after successful processing."""
        try:
            self.consumer.commit()
            logger.debug("Kafka offsets committed")
        except Exception as e:
            logger.error("Failed to commit offsets", error=str(e))
            raise

    def close(self):
        """Close the consumer gracefully."""
        try:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        except Exception as e:
            logger.error("Error closing consumer", error=str(e))
