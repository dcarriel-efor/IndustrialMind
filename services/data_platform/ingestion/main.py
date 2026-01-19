"""Data Ingestion Service - Kafka to InfluxDB pipeline."""

import os
import signal
import sys
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path

# Add project root to path for shared modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from kafka_consumer import SensorReadingConsumer
from influx_writer import InfluxDBWriter
from shared.utils.logging_config import configure_logging

# Configure centralized logging
logger = configure_logging(
    service_name="ingestion",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    add_context={"version": "0.1.0"}
)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info("Shutdown signal received", signal=sig)
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def write_with_retry(writer: InfluxDBWriter, readings):
    """Write to InfluxDB with exponential backoff retry.

    Args:
        writer: InfluxDBWriter instance
        readings: List of SensorReading objects

    Raises:
        Exception: If all retry attempts fail
    """
    if not writer.write_batch(readings):
        raise Exception("Failed to write batch to InfluxDB")


def main():
    """Main entry point for the ingestion service."""
    # Load configuration from environment variables
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    kafka_topic = os.getenv("KAFKA_TOPIC", "sensor-readings")
    kafka_group_id = os.getenv("KAFKA_GROUP_ID", "ingestion-service")

    influx_url = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
    influx_token = os.getenv("INFLUXDB_TOKEN", "industrialmind-token-123456")
    influx_org = os.getenv("INFLUXDB_ORG", "industrialmind")
    influx_bucket = os.getenv("INFLUXDB_BUCKET", "sensors")

    logger.info(
        "Starting Data Ingestion Service",
        kafka_bootstrap=kafka_bootstrap,
        kafka_topic=kafka_topic,
        kafka_group_id=kafka_group_id,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket
    )

    # Initialize consumer and writer
    consumer = None
    writer = None

    try:
        consumer = SensorReadingConsumer(
            bootstrap_servers=kafka_bootstrap,
            topic=kafka_topic,
            group_id=kafka_group_id
        )

        writer = InfluxDBWriter(
            url=influx_url,
            token=influx_token,
            org=influx_org,
            bucket=influx_bucket
        )

        logger.info("Ingestion service started successfully")

        # Main processing loop
        messages_processed = 0
        batches_processed = 0
        errors = 0

        while not shutdown_requested:
            try:
                # Consume batch (100 messages max, 1 second timeout)
                readings = consumer.consume_batch(timeout_ms=1000)

                if readings:
                    # Write to InfluxDB with retry
                    try:
                        write_with_retry(writer, readings)
                        # Commit Kafka offsets only after successful write
                        consumer.commit()

                        messages_processed += len(readings)
                        batches_processed += 1

                        # Log stats every 100 batches
                        if batches_processed % 100 == 0:
                            logger.info(
                                "Processing stats",
                                messages_processed=messages_processed,
                                batches_processed=batches_processed,
                                errors=errors
                            )

                    except Exception as e:
                        errors += 1
                        logger.error(
                            "Failed to write batch after retries",
                            error=str(e),
                            batch_size=len(readings)
                        )
                        # Don't commit offset - will retry on next poll
                        # Wait a bit before continuing to avoid tight error loop
                        time.sleep(5)

                else:
                    # No messages, brief sleep to avoid busy loop
                    time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                errors += 1
                logger.error("Unexpected error in main loop", error=str(e))
                time.sleep(5)  # Brief pause before continuing

        # Final stats
        logger.info(
            "Ingestion service stopping",
            total_messages=messages_processed,
            total_batches=batches_processed,
            total_errors=errors
        )

    except Exception as e:
        logger.error("Fatal error during initialization", error=str(e))
        sys.exit(1)

    finally:
        # Cleanup
        if consumer:
            consumer.close()
        if writer:
            writer.close()
        logger.info("Ingestion service stopped gracefully")


if __name__ == "__main__":
    main()
