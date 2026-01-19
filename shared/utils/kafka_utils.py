"""
Kafka utility functions for producing and consuming messages.

This module provides helper functions for creating Kafka producers and consumers
with proper configuration for the IndustrialMind platform.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


def create_kafka_producer(
    bootstrap_servers: Optional[str] = None,
    **kwargs
) -> KafkaProducer:
    """
    Create a Kafka producer with standard configuration.

    Args:
        bootstrap_servers: Kafka bootstrap servers (default: from KAFKA_BOOTSTRAP_SERVERS env)
        **kwargs: Additional KafkaProducer configuration options

    Returns:
        Configured KafkaProducer instance

    Example:
        >>> producer = create_kafka_producer()
        >>> producer.send('sensor-readings', value={'temperature': 25.5})
    """
    if bootstrap_servers is None:
        bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

    # Default configuration
    config = {
        'bootstrap_servers': bootstrap_servers.split(','),
        'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
        'key_serializer': lambda k: k.encode('utf-8') if k else None,
        'acks': 'all',  # Wait for all replicas to acknowledge
        'retries': 3,
        'max_in_flight_requests_per_connection': 1,  # Ensure ordering
        'compression_type': 'gzip',
        'linger_ms': 10,  # Batch messages for 10ms for efficiency
    }

    # Override with user-provided config
    config.update(kwargs)

    try:
        producer = KafkaProducer(**config)
        logger.info(f"Kafka producer created successfully. Bootstrap servers: {bootstrap_servers}")
        return producer
    except KafkaError as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        raise


def create_kafka_consumer(
    topics: list[str],
    group_id: str,
    bootstrap_servers: Optional[str] = None,
    **kwargs
) -> KafkaConsumer:
    """
    Create a Kafka consumer with standard configuration.

    Args:
        topics: List of topics to subscribe to
        group_id: Consumer group ID
        bootstrap_servers: Kafka bootstrap servers (default: from KAFKA_BOOTSTRAP_SERVERS env)
        **kwargs: Additional KafkaConsumer configuration options

    Returns:
        Configured KafkaConsumer instance

    Example:
        >>> consumer = create_kafka_consumer(['sensor-readings'], 'my-consumer-group')
        >>> for message in consumer:
        ...     print(message.value)
    """
    if bootstrap_servers is None:
        bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

    # Default configuration
    config = {
        'bootstrap_servers': bootstrap_servers.split(','),
        'group_id': group_id,
        'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
        'key_deserializer': lambda k: k.decode('utf-8') if k else None,
        'auto_offset_reset': 'latest',  # Start from latest on first run
        'enable_auto_commit': False,  # Manual commit for reliability
        'max_poll_records': 100,  # Batch size
        'session_timeout_ms': 30000,
        'heartbeat_interval_ms': 10000,
    }

    # Override with user-provided config
    config.update(kwargs)

    try:
        consumer = KafkaConsumer(*topics, **config)
        logger.info(f"Kafka consumer created successfully. Topics: {topics}, Group: {group_id}")
        return consumer
    except KafkaError as e:
        logger.error(f"Failed to create Kafka consumer: {e}")
        raise


def send_message(
    producer: KafkaProducer,
    topic: str,
    value: Dict[str, Any],
    key: Optional[str] = None,
    headers: Optional[list] = None
) -> None:
    """
    Send a message to Kafka with error handling.

    Args:
        producer: KafkaProducer instance
        topic: Topic name
        value: Message value (will be JSON serialized)
        key: Optional message key
        headers: Optional message headers

    Raises:
        KafkaError: If message fails to send

    Example:
        >>> producer = create_kafka_producer()
        >>> send_message(producer, 'sensor-readings', {'temperature': 25.5}, key='MACHINE_001')
    """
    try:
        future = producer.send(
            topic,
            value=value,
            key=key,
            headers=headers
        )

        # Block until message is sent (or timeout)
        record_metadata = future.get(timeout=10)

        logger.debug(
            f"Message sent to {topic}. "
            f"Partition: {record_metadata.partition}, "
            f"Offset: {record_metadata.offset}"
        )
    except KafkaError as e:
        logger.error(f"Failed to send message to {topic}: {e}")
        raise


def close_producer(producer: KafkaProducer, timeout: int = 30) -> None:
    """
    Gracefully close Kafka producer.

    Args:
        producer: KafkaProducer instance
        timeout: Timeout in seconds to wait for pending messages
    """
    try:
        producer.flush(timeout=timeout)
        producer.close(timeout=timeout)
        logger.info("Kafka producer closed successfully")
    except Exception as e:
        logger.error(f"Error closing Kafka producer: {e}")
        raise


def close_consumer(consumer: KafkaConsumer) -> None:
    """
    Gracefully close Kafka consumer.

    Args:
        consumer: KafkaConsumer instance
    """
    try:
        consumer.close()
        logger.info("Kafka consumer closed successfully")
    except Exception as e:
        logger.error(f"Error closing Kafka consumer: {e}")
        raise
