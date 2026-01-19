"""
Data simulator main entry point.

Generates realistic industrial sensor data and publishes to Kafka.
Supports configurable number of machines, sampling interval, and anomaly rate.
"""

import argparse
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from sensor_simulator import SensorSimulator, MachineState
from anomaly_generator import AnomalyGenerator, AnomalyType
from shared.utils.kafka_utils import create_kafka_producer, send_message, close_producer
from shared.utils.logging_config import configure_logging

# Configure centralized logging
logger = configure_logging(
    service_name="simulator",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    add_context={"version": "0.1.0"}
)


class DataSimulator:
    """
    Main data simulator orchestrator.

    Manages multiple machine simulators, anomaly injection, and Kafka publishing.

    Attributes:
        num_machines: Number of machines to simulate
        interval: Sampling interval in seconds
        anomaly_rate: Probability of anomaly injection (0.0-1.0)
        simulators: List of SensorSimulator instances
        anomaly_gen: AnomalyGenerator instance
        producer: Kafka producer
        shutdown_requested: Flag for graceful shutdown
    """

    def __init__(
        self,
        num_machines: int = 5,
        interval: float = 1.0,
        anomaly_rate: float = 0.05,
        kafka_bootstrap_servers: Optional[str] = None
    ):
        """
        Initialize data simulator.

        Args:
            num_machines: Number of machines to simulate (default: 5)
            interval: Sampling interval in seconds (default: 1.0)
            anomaly_rate: Anomaly injection probability (default: 0.05)
            kafka_bootstrap_servers: Kafka bootstrap servers (default: from env)
        """
        self.num_machines = num_machines
        self.interval = interval
        self.anomaly_rate = anomaly_rate
        self.shutdown_requested = False

        # Create simulators for each machine
        self.simulators = [
            SensorSimulator(f"MACHINE_{i:03d}")
            for i in range(1, num_machines + 1)
        ]

        # Create anomaly generator
        self.anomaly_gen = AnomalyGenerator()

        # Create Kafka producer
        self.producer = create_kafka_producer(bootstrap_servers=kafka_bootstrap_servers)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info(
            "data_simulator_initialized",
            num_machines=num_machines,
            interval=interval,
            anomaly_rate=anomaly_rate
        )

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        logger.info("shutdown_signal_received", signal=sig)
        self.shutdown_requested = True

    def run(self):
        """
        Main simulation loop.

        Generates sensor readings for all machines at configured interval,
        injects anomalies based on anomaly_rate, and publishes to Kafka.
        """
        logger.info("data_simulator_starting")

        readings_generated = 0
        anomalies_injected = 0

        try:
            while not self.shutdown_requested:
                for simulator in self.simulators:
                    # Generate base reading
                    reading = simulator.generate_reading()

                    # Inject anomaly with probability
                    if random.random() < self.anomaly_rate:
                        anomaly_type = random.choice(list(AnomalyType))

                        # Apply anomaly to sensor values
                        (
                            reading.temperature,
                            reading.vibration,
                            reading.pressure,
                            reading.power
                        ) = self.anomaly_gen.inject_anomaly(
                            reading.temperature,
                            reading.vibration,
                            reading.pressure,
                            reading.power,
                            anomaly_type
                        )

                        # Note: Metadata is None by design (enriched at ingestion layer)
                        # We track anomalies via logging instead
                        anomalies_injected += 1

                        logger.info(
                            "anomaly_injected",
                            machine_id=reading.machine_id,
                            anomaly_type=anomaly_type.value,
                            temperature=reading.temperature,
                            vibration=reading.vibration
                        )

                    # Publish to Kafka (use model_dump with mode='json' for proper JSON serialization)
                    send_message(
                        self.producer,
                        topic=os.getenv("KAFKA_TOPIC", "sensor-readings"),
                        value=reading.model_dump(mode='json'),
                        key=reading.machine_id
                    )

                    readings_generated += 1

                # Log stats every 100 readings
                if readings_generated % 100 == 0:
                    logger.info(
                        "simulator_stats",
                        readings_generated=readings_generated,
                        anomalies_injected=anomalies_injected,
                        anomaly_rate_actual=anomalies_injected / readings_generated
                    )

                # Sleep until next interval
                time.sleep(self.interval)

        except Exception as e:
            logger.error("simulation_error", error=str(e), exc_info=True)
            raise
        finally:
            self._shutdown()

    def _shutdown(self):
        """Graceful shutdown procedure."""
        logger.info("data_simulator_shutting_down")

        # Close Kafka producer
        close_producer(self.producer)

        logger.info(
            "data_simulator_stopped",
            total_readings=sum(1 for _ in self.simulators)
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Industrial sensor data simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--machines",
        type=int,
        default=5,
        help="Number of machines to simulate"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds"
    )

    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        help="Anomaly injection probability (0.0-1.0)"
    )

    parser.add_argument(
        "--kafka-bootstrap-servers",
        type=str,
        default=None,
        help="Kafka bootstrap servers (default: from KAFKA_BOOTSTRAP_SERVERS env)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if args.machines < 1:
        logger.error("invalid_num_machines", machines=args.machines)
        sys.exit(1)

    if not 0.0 <= args.anomaly_rate <= 1.0:
        logger.error("invalid_anomaly_rate", anomaly_rate=args.anomaly_rate)
        sys.exit(1)

    # Create and run simulator
    simulator = DataSimulator(
        num_machines=args.machines,
        interval=args.interval,
        anomaly_rate=args.anomaly_rate,
        kafka_bootstrap_servers=args.kafka_bootstrap_servers
    )

    simulator.run()


if __name__ == "__main__":
    main()
