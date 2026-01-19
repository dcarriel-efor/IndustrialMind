"""
Sensor simulator for generating realistic industrial sensor readings.

This module provides classes for simulating industrial equipment sensors with
state-based behavior patterns (NORMAL, DEGRADING, FAILING).
"""

import random
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import ValidationError

# Import shared schemas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared.schemas.sensor_reading import SensorReading


class MachineState(Enum):
    """Machine operational states."""
    NORMAL = "normal"
    DEGRADING = "degrading"
    FAILING = "failing"


class SensorSimulator:
    """
    Generates realistic sensor readings with state-based patterns.

    The simulator generates readings for 4 sensor types:
    - Temperature (°C)
    - Vibration (mm/s)
    - Pressure (PSI)
    - Power (W)

    Sensor values vary based on machine state:
    - NORMAL: Baseline operation with Gaussian noise
    - DEGRADING: Gradual increase in temperature and vibration
    - FAILING: Extreme values with high variance

    Attributes:
        machine_id: Unique machine identifier
        state: Current machine state (NORMAL, DEGRADING, FAILING)
        degradation_level: Degradation progress (0.0 to 1.0)
        sensor_id: Sensor identifier
    """

    # Sensor value ranges for different states
    NORMAL_RANGES = {
        'temperature': (55, 5),    # (mean, std) - 40-70°C range
        'vibration': (1.0, 0.2),   # 0.5-1.5 mm/s range
        'pressure': (50, 5),       # 40-60 PSI range
        'power': (250, 25),        # 200-300W range
    }

    DEGRADING_BASE_INCREASE = {
        'temperature': 20,  # Up to +20°C
        'vibration': 1.0,   # Up to +1.0 mm/s
        'pressure': -10,    # Down to -10 PSI
        'power': 50,        # Up to +50W
    }

    FAILING_RANGES = {
        'temperature': (90, 10),   # >80°C
        'vibration': (3.0, 0.5),   # >2.5 mm/s
        'pressure': (25, 5),       # <30 PSI
        'power': (450, 50),        # >400W
    }

    def __init__(
        self,
        machine_id: str,
        sensor_id: Optional[str] = None,
        initial_state: MachineState = MachineState.NORMAL
    ):
        """
        Initialize sensor simulator.

        Args:
            machine_id: Unique machine identifier
            sensor_id: Sensor identifier (default: machine_id + '_SENSOR')
            initial_state: Initial machine state
        """
        self.machine_id = machine_id
        self.sensor_id = sensor_id or f"{machine_id}_SENSOR"
        self.state = initial_state
        self.degradation_level = 0.0

    def generate_reading(self) -> SensorReading:
        """
        Generate a single sensor reading based on current state.

        Returns:
            SensorReading instance with current sensor values

        Raises:
            ValidationError: If generated reading fails Pydantic validation
        """
        if self.state == MachineState.NORMAL:
            values = self._generate_normal()
        elif self.state == MachineState.DEGRADING:
            values = self._generate_degrading()
        else:  # FAILING
            values = self._generate_failing()

        # Create SensorReading with current timestamp
        reading = SensorReading(
            timestamp=datetime.now(timezone.utc),
            machine_id=self.machine_id,
            sensor_id=self.sensor_id,
            temperature=values['temperature'],
            vibration=values['vibration'],
            pressure=values['pressure'],
            power=values['power'],
            metadata=None  # Metadata will be enriched at ingestion layer
        )

        return reading

    def _generate_normal(self) -> dict:
        """
        Generate normal operation sensor values with Gaussian noise.

        Returns:
            Dictionary with sensor values
        """
        return {
            'temperature': max(0, np.random.normal(*self.NORMAL_RANGES['temperature'])),
            'vibration': max(0, np.random.normal(*self.NORMAL_RANGES['vibration'])),
            'pressure': max(0, np.random.normal(*self.NORMAL_RANGES['pressure'])),
            'power': max(0, np.random.normal(*self.NORMAL_RANGES['power'])),
        }

    def _generate_degrading(self) -> dict:
        """
        Generate degrading state sensor values.

        Values gradually increase based on degradation_level.

        Returns:
            Dictionary with sensor values
        """
        base_values = self._generate_normal()

        # Apply degradation-based increases
        return {
            'temperature': base_values['temperature'] + (
                self.degradation_level * self.DEGRADING_BASE_INCREASE['temperature']
            ),
            'vibration': base_values['vibration'] + (
                self.degradation_level * self.DEGRADING_BASE_INCREASE['vibration']
            ),
            'pressure': base_values['pressure'] + (
                self.degradation_level * self.DEGRADING_BASE_INCREASE['pressure']
            ),
            'power': base_values['power'] + (
                self.degradation_level * self.DEGRADING_BASE_INCREASE['power']
            ),
        }

    def _generate_failing(self) -> dict:
        """
        Generate failing state sensor values.

        Extreme values with high variance.

        Returns:
            Dictionary with sensor values
        """
        return {
            'temperature': max(0, np.random.normal(*self.FAILING_RANGES['temperature'])),
            'vibration': max(0, np.random.normal(*self.FAILING_RANGES['vibration'])),
            'pressure': max(0, np.random.normal(*self.FAILING_RANGES['pressure'])),
            'power': max(0, np.random.normal(*self.FAILING_RANGES['power'])),
        }

    def set_state(self, state: MachineState, degradation_level: float = 0.0) -> None:
        """
        Set machine state and degradation level.

        Args:
            state: New machine state
            degradation_level: Degradation progress (0.0 to 1.0)

        Raises:
            ValueError: If degradation_level is not in [0.0, 1.0]
        """
        if not 0.0 <= degradation_level <= 1.0:
            raise ValueError("degradation_level must be between 0.0 and 1.0")

        self.state = state
        self.degradation_level = degradation_level

    def progress_degradation(self, increment: float = 0.01) -> None:
        """
        Progress degradation level for DEGRADING state.

        Args:
            increment: Amount to increase degradation (default: 0.01)
        """
        if self.state == MachineState.DEGRADING:
            self.degradation_level = min(1.0, self.degradation_level + increment)

            # Transition to FAILING if fully degraded
            if self.degradation_level >= 1.0:
                self.state = MachineState.FAILING
