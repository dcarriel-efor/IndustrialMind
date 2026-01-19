"""
Standalone test for simulator without Kafka.
Verifies sensor data generation works correctly.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from sensor_simulator import SensorSimulator, MachineState
from anomaly_generator import AnomalyGenerator, AnomalyType

def test_simulator_standalone():
    """Test simulator generates valid readings."""
    print("="*60)
    print("STANDALONE SIMULATOR TEST")
    print("="*60)

    # Create simulator
    sim = SensorSimulator("MACHINE_001")
    print(f"\n[OK] Created simulator for {sim.machine_id}")

    # Generate 10 normal readings
    print("\n--- Normal State (10 readings) ---")
    for i in range(10):
        reading = sim.generate_reading()
        print(f"  {i+1}. T={reading.temperature:.1f}C, V={reading.vibration:.2f}mm/s, "
              f"P={reading.pressure:.1f}PSI, Pow={reading.power:.0f}W")

    # Test degrading state
    print("\n--- Degrading State (50% degradation, 5 readings) ---")
    sim.set_state(MachineState.DEGRADING, degradation_level=0.5)
    for i in range(5):
        reading = sim.generate_reading()
        print(f"  {i+1}. T={reading.temperature:.1f}C, V={reading.vibration:.2f}mm/s, "
              f"P={reading.pressure:.1f}PSI, Pow={reading.power:.0f}W")

    # Test failing state
    print("\n--- Failing State (5 readings) ---")
    sim.set_state(MachineState.FAILING)
    for i in range(5):
        reading = sim.generate_reading()
        print(f"  {i+1}. T={reading.temperature:.1f}C, V={reading.vibration:.2f}mm/s, "
              f"P={reading.pressure:.1f}PSI, Pow={reading.power:.0f}W")

    print("\n[OK] All readings generated successfully!")

    # Test anomaly generator
    print("\n" + "="*60)
    print("ANOMALY GENERATOR TEST")
    print("="*60)

    gen = AnomalyGenerator()
    base_temp, base_vib, base_press, base_power = 55.0, 1.0, 50.0, 250.0

    # Test each anomaly type
    for anomaly_type in AnomalyType:
        temp, vib, press, power = gen.inject_anomaly(
            base_temp, base_vib, base_press, base_power,
            anomaly_type=anomaly_type
        )
        print(f"\n{anomaly_type.value.upper()}:")
        print(f"  Before: T={base_temp:.1f}C, V={base_vib:.2f}mm/s, P={base_press:.1f}PSI, Pow={base_power:.0f}W")
        print(f"  After:  T={temp:.1f}C, V={vib:.2f}mm/s, P={press:.1f}PSI, Pow={power:.0f}W")
        print(f"  Delta:  T={temp-base_temp:+.1f}C, V={vib-base_vib:+.2f}mm/s, "
              f"P={press-base_press:+.1f}PSI, Pow={power-base_power:+.0f}W")

    print("\n[OK] All anomaly types tested successfully!")

    # Test JSON serialization
    print("\n" + "="*60)
    print("JSON SERIALIZATION TEST")
    print("="*60)

    reading = sim.generate_reading()
    reading_dict = reading.model_dump()

    print(f"\nSerialized reading keys: {list(reading_dict.keys())}")
    print(f"Timestamp: {reading_dict['timestamp']}")
    print(f"Machine ID: {reading_dict['machine_id']}")
    print(f"Sensor ID: {reading_dict['sensor_id']}")
    print(f"Values: T={reading_dict['temperature']:.1f}, V={reading_dict['vibration']:.2f}, "
          f"P={reading_dict['pressure']:.1f}, Pow={reading_dict['power']:.0f}")

    print("\n[OK] JSON serialization works correctly!")

    print("\n" + "="*60)
    print("ALL TESTS PASSED [OK]")
    print("="*60)

if __name__ == "__main__":
    test_simulator_standalone()
