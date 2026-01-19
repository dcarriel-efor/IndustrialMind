"""
Unit tests for sensor simulator.

Tests sensor value generation, state transitions, and anomaly injection.
Target: >80% code coverage
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from sensor_simulator import SensorSimulator, MachineState
from anomaly_generator import AnomalyGenerator, AnomalyType
from shared.schemas.sensor_reading import SensorReading


class TestSensorSimulator:
    """Test cases for SensorSimulator class."""

    def test_initialization(self):
        """Test simulator initializes with correct defaults."""
        sim = SensorSimulator("MACHINE_001")

        assert sim.machine_id == "MACHINE_001"
        assert sim.sensor_id == "MACHINE_001_SENSOR"
        assert sim.state == MachineState.NORMAL
        assert sim.degradation_level == 0.0

    def test_custom_sensor_id(self):
        """Test simulator accepts custom sensor ID."""
        sim = SensorSimulator("MACHINE_001", sensor_id="CUSTOM_SENSOR")

        assert sim.sensor_id == "CUSTOM_SENSOR"

    def test_initial_state_override(self):
        """Test simulator can start in non-NORMAL state."""
        sim = SensorSimulator("MACHINE_001", initial_state=MachineState.DEGRADING)

        assert sim.state == MachineState.DEGRADING

    def test_normal_reading_in_range(self):
        """Test normal readings fall within expected ranges."""
        sim = SensorSimulator("MACHINE_001")

        # Generate 100 readings to test statistical properties
        readings = [sim._generate_normal() for _ in range(100)]

        for reading in readings:
            # Temperature: 40-70°C (mean 55, std 5) - allow 4 sigma for outliers
            assert 35 <= reading['temperature'] <= 75, f"Temp out of range: {reading['temperature']}"

            # Vibration: 0.5-1.5 mm/s (mean 1.0, std 0.2) - allow 4 sigma
            assert 0.2 <= reading['vibration'] <= 1.8, f"Vibration out of range: {reading['vibration']}"

            # Pressure: 40-60 PSI (mean 50, std 5) - allow 4 sigma
            assert 30 <= reading['pressure'] <= 70, f"Pressure out of range: {reading['pressure']}"

            # Power: 200-300W (mean 250, std 25) - allow 4 sigma
            assert 150 <= reading['power'] <= 350, f"Power out of range: {reading['power']}"

    def test_degrading_reading_higher_than_normal(self):
        """Test degrading readings show increased values."""
        sim = SensorSimulator("MACHINE_001")
        sim.set_state(MachineState.DEGRADING, degradation_level=0.5)

        # Generate reading
        reading = sim._generate_degrading()

        # With 50% degradation, temp should be at least 10°C above normal baseline
        # Normal mean: 55°C, degrading adds up to 20°C * 0.5 = 10°C
        assert reading['temperature'] > 55, f"Degrading temp should be elevated: {reading['temperature']}"

        # Vibration should also be elevated
        assert reading['vibration'] > 1.0, f"Degrading vibration should be elevated: {reading['vibration']}"

    def test_failing_reading_critical(self):
        """Test failing readings show critical values."""
        sim = SensorSimulator("MACHINE_001")
        sim.set_state(MachineState.FAILING)

        # Generate 10 readings to test consistency
        readings = [sim._generate_failing() for _ in range(10)]

        for reading in readings:
            # Temperature should be >80°C (mean 90, std 10) - allow some variance
            assert reading['temperature'] > 65, f"Failing temp should be critical: {reading['temperature']}"

            # Vibration should be >2.0 mm/s (mean 3.0, std 0.5) - allow some variance
            assert reading['vibration'] > 1.8, f"Failing vibration should be critical: {reading['vibration']}"

            # Pressure should be low <35 PSI (mean 25, std 5) - allow some variance
            assert reading['pressure'] < 40, f"Failing pressure should be low: {reading['pressure']}"

            # Power should be high >350W (mean 450, std 50) - allow 4 sigma variance
            assert reading['power'] > 300, f"Failing power should be high: {reading['power']}"

    def test_generate_reading_returns_sensor_reading(self):
        """Test generate_reading returns valid SensorReading instance."""
        sim = SensorSimulator("MACHINE_001")

        reading = sim.generate_reading()

        assert isinstance(reading, SensorReading)
        assert reading.machine_id == "MACHINE_001"
        assert reading.sensor_id == "MACHINE_001_SENSOR"
        assert hasattr(reading, 'timestamp')
        assert hasattr(reading, 'temperature')
        assert hasattr(reading, 'vibration')
        assert hasattr(reading, 'pressure')
        assert hasattr(reading, 'power')

    def test_state_transition(self):
        """Test state can be changed via set_state."""
        sim = SensorSimulator("MACHINE_001")

        assert sim.state == MachineState.NORMAL

        sim.set_state(MachineState.DEGRADING, degradation_level=0.3)
        assert sim.state == MachineState.DEGRADING
        assert sim.degradation_level == 0.3

        sim.set_state(MachineState.FAILING)
        assert sim.state == MachineState.FAILING

    def test_set_state_validates_degradation_level(self):
        """Test set_state rejects invalid degradation levels."""
        sim = SensorSimulator("MACHINE_001")

        # Should raise ValueError for degradation < 0
        with pytest.raises(ValueError, match="degradation_level must be between 0.0 and 1.0"):
            sim.set_state(MachineState.DEGRADING, degradation_level=-0.1)

        # Should raise ValueError for degradation > 1
        with pytest.raises(ValueError, match="degradation_level must be between 0.0 and 1.0"):
            sim.set_state(MachineState.DEGRADING, degradation_level=1.5)

    def test_progress_degradation(self):
        """Test degradation progresses correctly."""
        sim = SensorSimulator("MACHINE_001")
        sim.set_state(MachineState.DEGRADING, degradation_level=0.0)

        # Progress by default increment (0.01)
        sim.progress_degradation()
        assert sim.degradation_level == 0.01

        # Progress by custom increment
        sim.progress_degradation(increment=0.1)
        assert sim.degradation_level == 0.11

    def test_progress_degradation_caps_at_1(self):
        """Test degradation caps at 1.0 and doesn't exceed."""
        sim = SensorSimulator("MACHINE_001")
        sim.set_state(MachineState.DEGRADING, degradation_level=0.95)

        # Progress beyond 1.0
        sim.progress_degradation(increment=0.1)
        assert sim.degradation_level == 1.0

    def test_degradation_transitions_to_failing(self):
        """Test degradation automatically transitions to FAILING at 1.0."""
        sim = SensorSimulator("MACHINE_001")
        sim.set_state(MachineState.DEGRADING, degradation_level=0.95)

        # Progress to 1.0
        sim.progress_degradation(increment=0.1)

        assert sim.state == MachineState.FAILING
        assert sim.degradation_level == 1.0

    def test_progress_degradation_only_in_degrading_state(self):
        """Test progress_degradation only works in DEGRADING state."""
        sim = SensorSimulator("MACHINE_001")

        # Should not progress in NORMAL state
        sim.set_state(MachineState.NORMAL)
        sim.progress_degradation()
        assert sim.degradation_level == 0.0

        # Should not progress in FAILING state
        sim.set_state(MachineState.FAILING)
        sim.progress_degradation()
        assert sim.degradation_level == 0.0


class TestAnomalyGenerator:
    """Test cases for AnomalyGenerator class."""

    def test_initialization(self):
        """Test anomaly generator initializes correctly."""
        gen = AnomalyGenerator()

        assert gen.drift_offset == 0.0
        assert gen.cycle_counter == 0

    def test_inject_spike_anomaly(self):
        """Test spike anomaly increases one sensor significantly."""
        gen = AnomalyGenerator()

        temp, vib, press, power = 55.0, 1.0, 50.0, 250.0

        # Inject spike multiple times to test all sensors get spiked
        spike_detected = {
            'temperature': False,
            'vibration': False,
            'pressure': False,
            'power': False
        }

        for _ in range(50):  # Run multiple times to hit all sensors
            new_temp, new_vib, new_press, new_power = gen._inject_spike(
                temp, vib, press, power
            )

            if new_temp > temp + 20:
                spike_detected['temperature'] = True
            if new_vib > vib + 1.0:
                spike_detected['vibration'] = True
            if new_press < press - 15:
                spike_detected['pressure'] = True
            if new_power > power + 100:
                spike_detected['power'] = True

        # At least one sensor should have been spiked
        assert any(spike_detected.values()), "No spikes detected after 50 attempts"

    def test_inject_drift_anomaly(self):
        """Test drift anomaly accumulates over time."""
        gen = AnomalyGenerator()

        temp, vib, press, power = 55.0, 1.0, 50.0, 250.0

        # First drift
        temp1, vib1, _, _ = gen._inject_drift(temp, vib, press, power)

        # Second drift should be larger
        temp2, vib2, _, _ = gen._inject_drift(temp, vib, press, power)

        # Drift should accumulate
        assert temp2 > temp1, "Temperature drift should accumulate"
        assert vib2 > vib1, "Vibration drift should accumulate"
        assert gen.drift_offset > 0, "Drift offset should increase"

    def test_inject_cyclic_anomaly(self):
        """Test cyclic anomaly creates oscillating pattern."""
        gen = AnomalyGenerator()

        temp, vib, press, power = 55.0, 1.0, 50.0, 250.0

        vibrations = []

        # Collect 100 readings to see cyclic pattern
        for _ in range(100):
            _, new_vib, _, _ = gen._inject_cyclic(temp, vib, press, power)
            vibrations.append(new_vib)

        # Check that vibration oscillates (has both increases and decreases)
        diffs = [vibrations[i+1] - vibrations[i] for i in range(len(vibrations)-1)]
        has_increases = any(d > 0 for d in diffs)
        has_decreases = any(d < 0 for d in diffs)

        assert has_increases and has_decreases, "Cyclic pattern should oscillate"

    def test_inject_multi_sensor_anomaly(self):
        """Test multi-sensor anomaly affects multiple sensors."""
        gen = AnomalyGenerator()

        temp, vib, press, power = 55.0, 1.0, 50.0, 250.0

        new_temp, new_vib, new_press, new_power = gen._inject_multi_sensor(
            temp, vib, press, power
        )

        # All sensors should be affected
        assert new_temp > temp + 15, "Temperature should increase significantly"
        assert new_vib > vib + 0.5, "Vibration should increase"
        assert new_press < press - 3, "Pressure should decrease"
        assert new_power > power + 50, "Power should increase"

    def test_inject_anomaly_random_type(self):
        """Test inject_anomaly with random type selection."""
        gen = AnomalyGenerator()

        temp, vib, press, power = 55.0, 1.0, 50.0, 250.0

        # Should not raise error
        result = gen.inject_anomaly(temp, vib, press, power)

        assert len(result) == 4, "Should return 4 sensor values"

    def test_inject_anomaly_specific_type(self):
        """Test inject_anomaly with specific type."""
        gen = AnomalyGenerator()

        temp, vib, press, power = 55.0, 1.0, 50.0, 250.0

        # Test each anomaly type
        for anomaly_type in AnomalyType:
            result = gen.inject_anomaly(
                temp, vib, press, power,
                anomaly_type=anomaly_type
            )
            assert len(result) == 4, f"Should return 4 values for {anomaly_type}"

    def test_reset_drift(self):
        """Test reset_drift clears drift offset."""
        gen = AnomalyGenerator()

        # Build up drift
        for _ in range(10):
            gen._inject_drift(55.0, 1.0, 50.0, 250.0)

        assert gen.drift_offset > 0

        # Reset
        gen.reset_drift()
        assert gen.drift_offset == 0.0

    def test_reset_cycle(self):
        """Test reset_cycle clears cycle counter."""
        gen = AnomalyGenerator()

        # Advance cycle
        for _ in range(10):
            gen._inject_cyclic(55.0, 1.0, 50.0, 250.0)

        assert gen.cycle_counter > 0

        # Reset
        gen.reset_cycle()
        assert gen.cycle_counter == 0

    def test_reset_all(self):
        """Test reset clears all state."""
        gen = AnomalyGenerator()

        # Build up state
        for _ in range(10):
            gen._inject_drift(55.0, 1.0, 50.0, 250.0)
            gen._inject_cyclic(55.0, 1.0, 50.0, 250.0)

        # Reset
        gen.reset()

        assert gen.drift_offset == 0.0
        assert gen.cycle_counter == 0.0
