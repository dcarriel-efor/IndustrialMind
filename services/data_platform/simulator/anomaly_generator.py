"""
Anomaly generator for injecting realistic failure patterns into sensor readings.

This module provides classes for generating various types of industrial equipment
anomalies to test detection systems.
"""

import random
from enum import Enum
from typing import Optional

import numpy as np

# Import shared schemas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared.schemas.sensor_reading import SensorReading


class AnomalyType(Enum):
    """Types of anomalies that can be injected."""
    SPIKE = "spike"
    DRIFT = "drift"
    CYCLIC = "cyclic"
    MULTI_SENSOR = "multi_sensor"


class AnomalyGenerator:
    """
    Injects various anomaly patterns into sensor readings.

    Supports four types of anomalies:
    - SPIKE: Sudden value jump (e.g., temp +30°C for single reading)
    - DRIFT: Gradual increase over time (cumulative offset)
    - CYCLIC: Repeating sine wave pattern overlay
    - MULTI_SENSOR: Correlated anomaly across multiple sensors

    Attributes:
        drift_offset: Cumulative offset for drift anomalies
        cycle_counter: Counter for cyclic pattern phase
    """

    def __init__(self):
        """Initialize anomaly generator with internal state."""
        self.drift_offset = 0.0
        self.cycle_counter = 0

    def inject_anomaly(
        self,
        temperature: float,
        vibration: float,
        pressure: float,
        power: float,
        anomaly_type: Optional[AnomalyType] = None
    ) -> tuple[float, float, float, float]:
        """
        Inject specified anomaly type into sensor values.

        Args:
            temperature: Temperature value (°C)
            vibration: Vibration value (mm/s)
            pressure: Pressure value (PSI)
            power: Power value (W)
            anomaly_type: Type of anomaly to inject (random if None)

        Returns:
            Tuple of (temperature, vibration, pressure, power) with anomaly applied
        """
        if anomaly_type is None:
            anomaly_type = random.choice(list(AnomalyType))

        if anomaly_type == AnomalyType.SPIKE:
            return self._inject_spike(temperature, vibration, pressure, power)
        elif anomaly_type == AnomalyType.DRIFT:
            return self._inject_drift(temperature, vibration, pressure, power)
        elif anomaly_type == AnomalyType.CYCLIC:
            return self._inject_cyclic(temperature, vibration, pressure, power)
        else:  # MULTI_SENSOR
            return self._inject_multi_sensor(temperature, vibration, pressure, power)

    def _inject_spike(
        self,
        temperature: float,
        vibration: float,
        pressure: float,
        power: float
    ) -> tuple[float, float, float, float]:
        """
        Inject spike anomaly (sudden jump in one sensor).

        Randomly selects one sensor and adds large offset.

        Returns:
            Tuple with spike applied to one sensor
        """
        sensor_choice = random.choice(['temperature', 'vibration', 'pressure', 'power'])

        if sensor_choice == 'temperature':
            temperature += random.uniform(25, 35)  # +25-35°C spike
        elif sensor_choice == 'vibration':
            vibration += random.uniform(1.5, 2.5)  # +1.5-2.5 mm/s spike
        elif sensor_choice == 'pressure':
            pressure += random.uniform(-20, -30)  # -20-30 PSI drop
        else:  # power
            power += random.uniform(150, 250)  # +150-250W spike

        return temperature, vibration, pressure, power

    def _inject_drift(
        self,
        temperature: float,
        vibration: float,
        pressure: float,
        power: float
    ) -> tuple[float, float, float, float]:
        """
        Inject drift anomaly (gradual increase over time).

        Increases internal drift_offset and applies to temperature/vibration.

        Returns:
            Tuple with cumulative drift applied
        """
        # Increment drift offset by 0.5-1.0 per call
        self.drift_offset += random.uniform(0.5, 1.0)

        # Apply drift to temperature and vibration (most sensitive to degradation)
        temperature += self.drift_offset * 0.5  # Temperature drifts slower
        vibration += self.drift_offset * 0.1  # Vibration drifts faster

        return temperature, vibration, pressure, power

    def _inject_cyclic(
        self,
        temperature: float,
        vibration: float,
        pressure: float,
        power: float
    ) -> tuple[float, float, float, float]:
        """
        Inject cyclic anomaly (repeating sine wave pattern).

        Overlays sine wave on vibration reading (typical of bearing issues).

        Returns:
            Tuple with cyclic pattern applied to vibration
        """
        self.cycle_counter += 1

        # Create sine wave with period of ~50 readings (5 Hz at 10 Hz sampling)
        cyclic_offset = np.sin(self.cycle_counter * 0.1) * 0.8

        # Apply primarily to vibration (characteristic of mechanical oscillation)
        vibration += cyclic_offset

        # Minor effect on temperature (friction from oscillation)
        temperature += abs(cyclic_offset) * 2

        return temperature, vibration, pressure, power

    def _inject_multi_sensor(
        self,
        temperature: float,
        vibration: float,
        pressure: float,
        power: float
    ) -> tuple[float, float, float, float]:
        """
        Inject multi-sensor anomaly (correlated across sensors).

        Simulates cascading failure where multiple sensors show issues.

        Returns:
            Tuple with correlated anomalies across multiple sensors
        """
        # Simulate overheating scenario (temp + vibration + power increase)
        temperature += random.uniform(20, 30)
        vibration += random.uniform(1.0, 1.8)
        power += random.uniform(100, 150)

        # Pressure may drop due to system stress
        pressure -= random.uniform(5, 15)

        return temperature, vibration, pressure, power

    def reset_drift(self) -> None:
        """Reset drift offset to zero."""
        self.drift_offset = 0.0

    def reset_cycle(self) -> None:
        """Reset cycle counter to zero."""
        self.cycle_counter = 0

    def reset(self) -> None:
        """Reset all internal state."""
        self.drift_offset = 0.0
        self.cycle_counter = 0.0
