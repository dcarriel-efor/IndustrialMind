# Week 1 Exhaustive Implementation Plan

## Plan Status: Ready for Execution ✅

**Date Created**: 2026-01-13
**Project**: IndustrialMind - Industrial AI Platform
**Month**: 1 (Foundation & Data Pipeline)
**Week**: 1 (Data Pipeline Foundation)

---

## Context & Current State

### What's Already Complete (Infrastructure files created):
- ✅ [docker-compose.yml](../../docker-compose.yml) with 9 infrastructure services configured
- ✅ [Makefile](../../Makefile) with common Docker Compose commands
- ✅ [.env](../../.env) and [.env.example](../../.env.example) (Azure configuration, ports: MLflow 5011, Grafana 3011)
- ✅ Shared Pydantic schemas: [sensor_reading.py](../../shared/schemas/sensor_reading.py), [anomaly.py](../../shared/schemas/anomaly.py), [maintenance.py](../../shared/schemas/maintenance.py), [alerts.py](../../shared/schemas/alerts.py)
- ✅ Infrastructure configs: [Prometheus config](../../infrastructure/prometheus/prometheus.yml), [PostgreSQL init script](../../infrastructure/postgres/init-multiple-dbs.sh)
- ✅ Complete documentation: [ARCHITECTURE_SCHEMA.md](../../docs/ARCHITECTURE_SCHEMA.md), [CREDENTIALS.md](../../docs/CREDENTIALS.md), [LAUNCH_CHECKLIST.md](../../LAUNCH_CHECKLIST.md)
- ✅ Project folder structure: services/, shared/, tests/, docs/, infrastructure/

### Infrastructure Services Running (9 containers ready):
1. Zookeeper (coordination) - port 2181
2. Kafka (event streaming) - ports 9092, 29092
3. InfluxDB (time-series storage) - port 8086
4. PostgreSQL (relational database) - port 5432
5. Redis (caching) - port 6379
6. MLflow (ML tracking) - port 5011
7. Prometheus (metrics) - port 9090
8. Grafana (dashboards) - port 3011
9. Docker Network: industrialmind-network

### Week 1 Goal (from [project_based_roadmap.md](../../project_based_roadmap.md)):
Build the foundation data pipeline with 3 application services:
1. **Data Simulator** - Generate realistic industrial sensor data → Kafka
2. **Data Ingestion** - Consume from Kafka → Write to InfluxDB
3. **Streamlit Dashboard** - Query InfluxDB → Visualize real-time

---

## Day-by-Day Implementation Breakdown

### 🗓️ DAY 1-2: Data Simulator Service

**Objective**: Create Python service that generates realistic industrial sensor data and publishes to Kafka.

#### File Structure to Create

```
services/data_platform/simulator/
├── __init__.py
├── main.py                      # Entry point with CLI args
├── sensor_simulator.py          # Core simulator logic
├── anomaly_generator.py         # Anomaly injection
├── config.py                    # Configuration management
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image
└── tests/
    ├── __init__.py
    └── test_simulator.py        # Unit tests (>80% coverage)
```

#### Implementation Details

**1. sensor_simulator.py**

Core classes and methods:
```python
from enum import Enum
from dataclasses import dataclass
import numpy as np
from datetime import datetime

class MachineState(Enum):
    NORMAL = "normal"
    DEGRADING = "degrading"
    FAILING = "failing"

class SensorSimulator:
    """Generates realistic sensor readings with state-based patterns"""

    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.state = MachineState.NORMAL
        self.degradation_level = 0.0  # 0.0 to 1.0

    def generate_reading(self) -> SensorReading:
        """Generate single sensor reading based on current state"""
        if self.state == MachineState.NORMAL:
            return self._generate_normal()
        elif self.state == MachineState.DEGRADING:
            return self._generate_degrading()
        else:
            return self._generate_failing()

    def _generate_normal(self) -> SensorReading:
        """Normal operation ranges with Gaussian noise"""
        temp = np.random.normal(55, 5)  # 40-70°C
        vibration = np.random.normal(1.0, 0.2)  # 0.5-1.5 mm/s
        pressure = np.random.normal(50, 5)  # 40-60 PSI
        power = np.random.normal(250, 25)  # 200-300W
        return SensorReading(...)

    def _generate_degrading(self) -> SensorReading:
        """Degrading: gradual increase in temp/vibration"""
        base_temp = 55 + (self.degradation_level * 20)  # 55→75°C
        base_vib = 1.0 + (self.degradation_level * 1.0)  # 1.0→2.0 mm/s
        # Add noise and return

    def _generate_failing(self) -> SensorReading:
        """Failing: extreme values, high variance"""
        temp = np.random.normal(90, 10)  # >80°C
        vibration = np.random.normal(3.0, 0.5)  # >2.5 mm/s
        # Return critical values
```

**Sensor Value Ranges**:
| Sensor | Normal | Degrading | Failing |
|--------|--------|-----------|---------|
| Temperature | 40-70°C | 70-85°C | >85°C |
| Vibration | 0.5-1.5 mm/s | 1.5-2.5 mm/s | >2.5 mm/s |
| Pressure | 40-60 PSI | 30-40 or 60-70 PSI | <30 or >70 PSI |
| Power | 200-300W | 300-400W | >400W |

**2. anomaly_generator.py**

Inject realistic anomaly patterns:
```python
class AnomalyGenerator:
    """Injects various anomaly patterns into sensor readings"""

    ANOMALY_TYPES = ["spike", "drift", "cyclic", "multi_sensor"]

    def inject_anomaly(self, reading: SensorReading, anomaly_type: str):
        """Inject specified anomaly type into reading"""
        if anomaly_type == "spike":
            # Sudden value jump (e.g., temp +30°C for 1 reading)
            reading.temperature += 30
        elif anomaly_type == "drift":
            # Gradual increase over time (tracked internally)
            self.drift_offset += 0.5
            reading.temperature += self.drift_offset
        elif anomaly_type == "cyclic":
            # Repeating pattern (e.g., sine wave overlay)
            reading.vibration += np.sin(self.cycle_counter * 0.1) * 0.5
        elif anomaly_type == "multi_sensor":
            # Correlated anomaly (temp + vibration together)
            reading.temperature += 20
            reading.vibration += 1.5
        return reading
```

**3. main.py**

Entry point with configuration:
```python
import argparse
from kafka_utils import create_kafka_producer
from sensor_simulator import SensorSimulator
import structlog

logger = structlog.get_logger()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--interval", type=float, default=1.0)  # seconds
    parser.add_argument("--anomaly-rate", type=float, default=0.05)  # 5%
    args = parser.parse_args()

    # Create Kafka producer
    producer = create_kafka_producer()

    # Create simulators for each machine
    simulators = [
        SensorSimulator(f"MACHINE_{i:03d}")
        for i in range(1, args.machines + 1)
    ]

    logger.info("Starting simulator", machines=args.machines, interval=args.interval)

    while True:
        for sim in simulators:
            reading = sim.generate_reading()

            # Inject anomaly with probability
            if random.random() < args.anomaly_rate:
                anomaly_type = random.choice(ANOMALY_TYPES)
                reading = inject_anomaly(reading, anomaly_type)

            # Publish to Kafka
            producer.send("sensor-readings", value=reading.to_dict())

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
```

**4. Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command (can be overridden in docker-compose)
CMD ["python", "main.py", "--machines", "5", "--interval", "1", "--anomaly-rate", "0.05"]
```

**5. requirements.txt**

```
kafka-python==2.0.2
pydantic==2.5.0
numpy==1.26.0
python-dotenv==1.0.0
structlog==23.2.0
```

**6. tests/test_simulator.py**

```python
import pytest
from sensor_simulator import SensorSimulator, MachineState

def test_normal_reading_in_range():
    sim = SensorSimulator("MACHINE_001")
    reading = sim._generate_normal()

    assert 40 <= reading.temperature <= 70
    assert 0.5 <= reading.vibration <= 1.5
    assert 40 <= reading.pressure <= 60
    assert 200 <= reading.power <= 300

def test_state_transition():
    sim = SensorSimulator("MACHINE_001")
    assert sim.state == MachineState.NORMAL

    sim.state = MachineState.DEGRADING
    reading = sim.generate_reading()
    assert reading.temperature > 55  # Degrading temps higher

def test_anomaly_injection():
    # Test spike anomaly
    # Test drift anomaly
    # Test multi-sensor anomaly
    pass
```

**7. Update docker-compose.yml**

Uncomment and configure:
```yaml
  data-simulator:
    build:
      context: ./services/data_platform/simulator
      dockerfile: Dockerfile
    container_name: industrialmind-data-simulator
    depends_on:
      kafka:
        condition: service_healthy
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:29092
      KAFKA_TOPIC: sensor-readings
    env_file:
      - .env
    restart: unless-stopped
```

#### Day 1-2 Success Criteria

- [ ] Simulator generates 1000+ readings/minute (5 machines × 1 reading/sec × 60 sec = 300 minimum)
- [ ] Readings follow realistic patterns (Gaussian noise, proper ranges)
- [ ] Anomalies injected at 5% rate
- [ ] Successfully publishes to Kafka topic `sensor-readings`
- [ ] Uses existing `SensorReading` Pydantic schema from [shared/schemas/](../../shared/schemas/)
- [ ] Graceful shutdown on SIGTERM
- [ ] Logs structured output (JSON format)
- [ ] Unit tests pass with >80% coverage
- [ ] Container builds successfully: `docker build -t industrialmind-simulator .`
- [ ] Runs via docker-compose: `docker-compose up data-simulator`

---

### 🗓️ DAY 3-4: Data Ingestion Service

**Objective**: Create Python service that consumes from Kafka and writes to InfluxDB.

#### File Structure to Create

```
services/data_platform/ingestion/
├── __init__.py
├── main.py                      # Entry point
├── kafka_consumer.py            # Kafka consumer logic
├── influx_writer.py             # InfluxDB batch writer
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container image
└── tests/
    ├── __init__.py
    └── test_ingestion.py        # Unit + integration tests
```

#### Implementation Details

**1. kafka_consumer.py**

```python
from kafka import KafkaConsumer
from shared.schemas.sensor_reading import SensorReading
import structlog

logger = structlog.get_logger()

class SensorReadingConsumer:
    """Consumes sensor readings from Kafka with validation"""

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',  # Start from latest on first run
            enable_auto_commit=False,  # Manual commit after successful write
            max_poll_records=100  # Batch size
        )
        logger.info("Kafka consumer initialized", topic=topic, group_id=group_id)

    def consume_batch(self, timeout_ms=1000) -> List[SensorReading]:
        """Consume batch of messages, validate with Pydantic"""
        messages = self.consumer.poll(timeout_ms=timeout_ms)

        validated_readings = []
        for topic_partition, records in messages.items():
            for record in records:
                try:
                    # Validate with Pydantic schema
                    reading = SensorReading(**record.value)
                    validated_readings.append(reading)
                except ValidationError as e:
                    logger.error("Invalid message", error=str(e), value=record.value)
                    # Skip invalid messages, continue processing

        return validated_readings

    def commit(self):
        """Commit offsets after successful processing"""
        self.consumer.commit()
```

**2. influx_writer.py**

```python
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import structlog

logger = structlog.get_logger()

class InfluxDBWriter:
    """Writes sensor readings to InfluxDB with batching"""

    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = bucket
        self.org = org
        logger.info("InfluxDB writer initialized", url=url, bucket=bucket)

    def write_batch(self, readings: List[SensorReading]):
        """Convert readings to InfluxDB Points and write batch"""
        points = [self._reading_to_point(r) for r in readings]

        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            logger.info("Wrote batch to InfluxDB", count=len(points))
            return True
        except Exception as e:
            logger.error("Failed to write to InfluxDB", error=str(e))
            return False

    def _reading_to_point(self, reading: SensorReading) -> Point:
        """Convert SensorReading to InfluxDB Point format"""
        return (
            Point("sensor_readings")
            .tag("machine_id", reading.machine_id)
            .tag("sensor_id", reading.sensor_id)
            .field("temperature", reading.temperature)
            .field("vibration", reading.vibration)
            .field("pressure", reading.pressure)
            .field("power", reading.power)
            .time(reading.timestamp, WritePrecision.NS)
        )
```

**3. main.py**

```python
from kafka_consumer import SensorReadingConsumer
from influx_writer import InfluxDBWriter
from tenacity import retry, stop_after_attempt, wait_exponential
import signal
import sys

# Graceful shutdown handler
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logger.info("Shutdown signal received")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def write_with_retry(writer, readings):
    """Write to InfluxDB with exponential backoff retry"""
    if not writer.write_batch(readings):
        raise Exception("Failed to write batch")

def main():
    # Load config from environment
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    influx_url = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
    influx_token = os.getenv("INFLUXDB_TOKEN", "industrialmind-token-123456")

    # Initialize consumer and writer
    consumer = SensorReadingConsumer(
        bootstrap_servers=kafka_bootstrap,
        topic="sensor-readings",
        group_id="ingestion-service"
    )

    writer = InfluxDBWriter(
        url=influx_url,
        token=influx_token,
        org="industrialmind",
        bucket="sensors"
    )

    logger.info("Ingestion service started")

    while not shutdown_requested:
        # Consume batch (100 messages max, 1 second timeout)
        readings = consumer.consume_batch(timeout_ms=1000)

        if readings:
            # Write to InfluxDB with retry
            try:
                write_with_retry(writer, readings)
                # Commit Kafka offsets only after successful write
                consumer.commit()
            except Exception as e:
                logger.error("Failed to write batch after retries", error=str(e))
                # Don't commit offset - will retry on next poll

    logger.info("Ingestion service stopped gracefully")

if __name__ == "__main__":
    main()
```

**4. requirements.txt**

```
kafka-python==2.0.2
influxdb-client==1.38.0
pydantic==2.5.0
python-dotenv==1.0.0
structlog==23.2.0
tenacity==8.2.3
```

**5. Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

**6. tests/test_ingestion.py**

```python
import pytest
from unittest.mock import Mock, patch
from kafka_consumer import SensorReadingConsumer
from influx_writer import InfluxDBWriter

def test_valid_message_deserialization():
    # Test Pydantic validation works
    pass

def test_invalid_message_skipped():
    # Test that invalid messages are logged and skipped
    pass

def test_batch_write_to_influxdb():
    # Test batch writing
    pass

def test_offset_commit_after_write():
    # Test Kafka offset only committed after successful write
    pass

@pytest.mark.integration
def test_end_to_end_kafka_to_influx(kafka_container, influxdb_container):
    # Integration test with TestContainers
    # Produce message to Kafka
    # Run ingestion
    # Query InfluxDB to verify data
    pass
```

**7. Update docker-compose.yml**

```yaml
  data-ingestion:
    build:
      context: ./services/data_platform/ingestion
      dockerfile: Dockerfile
    container_name: industrialmind-data-ingestion
    depends_on:
      kafka:
        condition: service_healthy
      influxdb:
        condition: service_healthy
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:29092
      KAFKA_TOPIC: sensor-readings
      INFLUXDB_URL: http://influxdb:8086
      INFLUXDB_TOKEN: industrialmind-token-123456
      INFLUXDB_ORG: industrialmind
      INFLUXDB_BUCKET: sensors
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 10s
      timeout: 5s
      retries: 5
```

#### Day 3-4 Success Criteria

- [ ] Consumes from `sensor-readings` Kafka topic
- [ ] Validates messages using `SensorReading` Pydantic schema
- [ ] Writes to InfluxDB `sensors` bucket
- [ ] Batch size: 100 points per write
- [ ] Latency: <1 second from Kafka message to InfluxDB write
- [ ] Invalid messages logged and skipped (no crash)
- [ ] Kafka offsets committed only after successful write
- [ ] Retries with exponential backoff on InfluxDB failures
- [ ] Graceful shutdown on SIGTERM
- [ ] Structured logging (JSON)
- [ ] Unit tests + 1 integration test passing (>80% coverage)
- [ ] Verified data in InfluxDB UI (http://localhost:8086)

---

### 🗓️ DAY 5-6: Streamlit Dashboard

**Objective**: Create real-time visualization dashboard for sensor data.

#### File Structure to Create

```
services/dashboard/
├── __init__.py
├── app.py                       # Main Streamlit app
├── pages/
│   ├── 1_📊_Real_Time_Monitor.py
│   ├── 2_📈_Historical_Analysis.py
│   └── 3_🏭_Machine_Health.py
├── components/
│   ├── __init__.py
│   ├── influx_client.py        # InfluxDB query helpers
│   └── charts.py               # Reusable Plotly charts
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container image
└── tests/
    ├── __init__.py
    └── test_dashboard.py        # Unit tests
```

#### Implementation Details

See the complete detailed implementation in the [full plan file](C:\Users\diego\.claude\plans\glittery-munching-tide.md) lines 1607-2003.

Key components:
- **influx_client.py**: InfluxDB query helpers with Streamlit caching
- **charts.py**: Plotly chart components (time series, gauges, multi-sensor)
- **app.py**: Main page with real-time monitoring
- **pages/**: Historical analysis, machine health views

#### Day 5-6 Success Criteria

- [ ] Dashboard accessible at http://localhost:8501
- [ ] Displays all available machines in dropdown
- [ ] Real-time charts update every 2 seconds (when auto-refresh enabled)
- [ ] All 4 sensor types visualized (temperature, vibration, pressure, power)
- [ ] Metric cards show current values
- [ ] Gauge charts show status (green/yellow/red zones)
- [ ] Historical analysis page works (date range picker)
- [ ] CSV download functionality works
- [ ] Machine health page shows all machines with health scores
- [ ] Handles empty data gracefully (no crashes)
- [ ] Multi-page navigation works
- [ ] Responsive layout (looks good on different screen sizes)
- [ ] Unit tests for query functions (>80% coverage)

---

### 🗓️ DAY 7: Integration Testing, Documentation & Release

**Objective**: Ensure everything works end-to-end, comprehensive testing, and complete Week 1 documentation.

#### Tasks

**1. End-to-End Integration Testing**

Create comprehensive integration test suite in `tests/integration/`:
- conftest.py (pytest fixtures, Docker Compose setup)
- test_simulator_to_kafka.py
- test_kafka_to_influxdb.py
- test_end_to_end.py

**2. Performance Testing**

Create `scripts/load_test.py`:
- Simulate 10 machines × 10 readings/sec = 100 readings/sec
- Run for 10 minutes
- Measure: Kafka lag, InfluxDB write latency, Memory/CPU usage
- Verify: No dropped messages

**3. Update All Documentation**

- README.md - Add Week 1 deliverables section
- docs/WEEK_1_SUMMARY.md - Create detailed breakdown
- Update LAUNCH_CHECKLIST.md with Week 1 verifications

**4. Code Quality Checks**

```bash
# Linting
ruff check services/ --fix

# Type checking
mypy services/data_platform/simulator/
mypy services/data_platform/ingestion/
mypy services/dashboard/

# Format check
black services/

# Security scan
bandit -r services/ -ll

# Dependency check
safety check
```

**5. Git Commits (Proper Structure)**

6 commits planned:
1. feat(simulator): implement industrial sensor data simulator
2. feat(ingestion): implement Kafka to InfluxDB ingestion service
3. feat(dashboard): implement Streamlit real-time monitoring dashboard
4. test: add end-to-end integration tests for Week 1 pipeline
5. docs: complete Week 1 documentation and demo
6. chore: enable data-simulator, data-ingestion, and dashboard services

All commits include `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`

**6. Tag Release v0.1.0-week1**

```bash
git tag -a v0.1.0-week1 -m "Week 1 Deliverable: Foundation Data Pipeline

Features:
✅ Industrial sensor data simulator (5 machines, 4 sensors)
✅ Kafka streaming infrastructure (sensor-readings topic)
✅ InfluxDB time-series storage (sensors bucket)
✅ Streamlit real-time dashboard (3 pages)
✅ Complete integration tests (35 tests, 81% coverage)

Performance:
- Throughput: 1200+ readings/minute
- Latency: <3 seconds end-to-end
- Reliability: 0 errors in 10-minute load test

Tech Stack:
- Python 3.11
- Apache Kafka
- InfluxDB 2.7
- Streamlit 1.29
- Docker Compose

Architecture:
Simulator → Kafka → Ingestion → InfluxDB → Dashboard
"

git push origin main --tags
```

#### Day 7 Success Criteria

- [ ] All integration tests pass (5 tests)
- [ ] All unit tests pass (35 tests across 3 services)
- [ ] Code coverage >80% (target: 81%)
- [ ] Load test: 10 minutes sustained @ 100 readings/sec, zero errors
- [ ] Performance targets met (see table above)
- [ ] Code quality checks pass (ruff, mypy, black, bandit)
- [ ] README.md updated with Week 1 section
- [ ] WEEK_1_SUMMARY.md created with detailed breakdown
- [ ] Demo script works end-to-end
- [ ] Git commits properly structured with Co-Authored-By
- [ ] Tagged release: v0.1.0-week1
- [ ] Screenshots captured for portfolio
- [ ] All documentation accurate and complete

---

## Final Verification Checklist

### Infrastructure (9 containers)
- [ ] Zookeeper running and healthy
- [ ] Kafka running and healthy
- [ ] InfluxDB running and healthy
- [ ] PostgreSQL running and healthy
- [ ] Redis running and healthy
- [ ] MLflow running and healthy
- [ ] Prometheus running and healthy
- [ ] Grafana running and healthy
- [ ] All on industrialmind-network

### Application Services (3 containers)
- [ ] data-simulator running and producing
- [ ] data-ingestion running and consuming
- [ ] dashboard running and serving on :8501

### Kafka
- [ ] Topic `sensor-readings` created
- [ ] Topic has messages (check: `kafka-console-consumer`)
- [ ] Consumer group `ingestion-service` registered
- [ ] No lag (consumer keeping up with producer)

### InfluxDB
- [ ] Bucket `sensors` exists
- [ ] Data visible in InfluxDB UI (http://localhost:8086)
- [ ] Query returns recent data
- [ ] No write errors in logs

### Dashboard
- [ ] Accessible at http://localhost:8501
- [ ] All machines appear in dropdown
- [ ] Charts display correctly
- [ ] Auto-refresh works
- [ ] No JavaScript errors in browser console
- [ ] All 3 pages load correctly

### Data Flow
- [ ] Simulator logs show "Produced message"
- [ ] Kafka has messages in topic
- [ ] Ingestion logs show "Wrote batch to InfluxDB"
- [ ] InfluxDB has data
- [ ] Dashboard queries InfluxDB successfully
- [ ] End-to-end latency <5 seconds

### Testing
- [ ] Unit tests: 35 passing, 0 failing
- [ ] Integration tests: 5 passing, 0 failing
- [ ] Coverage: >80% for all services
- [ ] Load test: 10 minutes sustained, 0 errors

### Performance
- [ ] Throughput: >1000 readings/minute ✅ Target: 1200/min
- [ ] Kafka → InfluxDB lag: <1 second ✅ Target: ~500ms
- [ ] Dashboard refresh: <500ms ✅ Target: ~300ms
- [ ] Memory usage: <2GB ✅ Target: ~1.8GB
- [ ] CPU usage: <50% ✅ Target: ~35%

### Code Quality
- [ ] Ruff linting passes
- [ ] Mypy type checking passes
- [ ] Black formatting applied
- [ ] Bandit security scan passes
- [ ] No hardcoded credentials
- [ ] Proper error handling
- [ ] Structured logging throughout

### Documentation
- [ ] README.md updated
- [ ] WEEK_1_SUMMARY.md created
- [ ] Architecture diagrams included
- [ ] Known limitations documented
- [ ] Next steps outlined
- [ ] Screenshots captured

### Git & Release
- [ ] 6 clean commits with descriptive messages
- [ ] All commits have Co-Authored-By
- [ ] No secrets in commit history
- [ ] .gitignore configured
- [ ] Tagged: v0.1.0-week1
- [ ] Pushed to remote with tags

---

## Dependencies Summary

### Simulator
```
kafka-python==2.0.2
pydantic==2.5.0
numpy==1.26.0
python-dotenv==1.0.0
structlog==23.2.0
```

### Ingestion
```
kafka-python==2.0.2
influxdb-client==1.38.0
pydantic==2.5.0
python-dotenv==1.0.0
structlog==23.2.0
tenacity==8.2.3
```

### Dashboard
```
streamlit==1.29.0
influxdb-client==1.38.0
plotly==5.18.0
pandas==2.1.3
python-dotenv==1.0.0
```

### Testing (requirements-dev.txt)
```
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0
testcontainers==3.7.1
ruff==0.1.6
mypy==1.7.1
black==23.11.0
bandit==1.7.5
safety==2.3.5
```

---

## Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Simulator throughput | >1000 readings/min | Count Kafka messages |
| Kafka → InfluxDB lag | <1 second | Compare timestamps |
| Dashboard page load | <500ms | Browser DevTools |
| End-to-end latency | <5 seconds | Simulator → Dashboard |
| Memory (all 12 containers) | <2GB | `docker stats` |
| CPU (average) | <50% | `docker stats` |
| Test coverage | >80% | `pytest --cov` |
| Tests passing | 100% | `pytest -v` |

---

## Week 1 Summary

### What Will Be Achieved

**Services Built**: 3 production-ready microservices
1. **Data Simulator** (350 lines of code)
   - Realistic sensor value generation
   - State-based patterns (NORMAL, DEGRADING, FAILING)
   - 4 anomaly types
   - Kafka producer integration

2. **Data Ingestion** (280 lines of code)
   - Kafka consumer with validation
   - Batch InfluxDB writes
   - Error handling and retries
   - Offset management

3. **Streamlit Dashboard** (420 lines of code)
   - 3-page multi-page app
   - Real-time visualization
   - Historical analysis
   - Machine health monitoring

**Infrastructure**: 9 Docker containers orchestrated
- Kafka, InfluxDB, PostgreSQL, Redis, MLflow, Prometheus, Grafana, Zookeeper, Network

**Testing**: 40 comprehensive tests
- 35 unit tests (81% coverage)
- 5 integration tests
- 1 load test (10 minutes sustained)

**Documentation**: 4 comprehensive documents
- README updated
- WEEK_1_SUMMARY created
- Architecture diagrams
- Troubleshooting guide

### Estimated Effort

| Task | Estimated Hours |
|------|----------------|
| Day 1-2: Simulator | 14 hours |
| Day 3-4: Ingestion | 14 hours |
| Day 5-6: Dashboard | 14 hours |
| Day 7: Testing & Docs | 7 hours |
| **Total** | **49 hours** |

### Skills Demonstrated

- ✅ Python microservices architecture
- ✅ Event-driven architecture (Kafka)
- ✅ Time-series database optimization
- ✅ Real-time data visualization
- ✅ Docker containerization
- ✅ Testing (unit + integration)
- ✅ Error handling and resilience
- ✅ Structured logging
- ✅ Git workflow
- ✅ Technical documentation

### Next Steps (Week 2)

From [project_based_roadmap.md](../../project_based_roadmap.md):
1. Enhanced simulator with degradation curves
2. Advanced dashboard with statistical analysis
3. API documentation
4. Prepare for Month 2: PyTorch anomaly detection model

---

## Appendix: Complete File Tree

```
IndustrialMind/
├── services/
│   ├── data_platform/
│   │   ├── simulator/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── sensor_simulator.py
│   │   │   ├── anomaly_generator.py
│   │   │   ├── config.py
│   │   │   ├── requirements.txt
│   │   │   ├── Dockerfile
│   │   │   └── tests/
│   │   │       ├── __init__.py
│   │   │       └── test_simulator.py
│   │   └── ingestion/
│   │       ├── __init__.py
│   │       ├── main.py
│   │       ├── kafka_consumer.py
│   │       ├── influx_writer.py
│   │       ├── config.py
│   │       ├── requirements.txt
│   │       ├── Dockerfile
│   │       └── tests/
│   │           ├── __init__.py
│   │           └── test_ingestion.py
│   └── dashboard/
│       ├── __init__.py
│       ├── app.py
│       ├── pages/
│       │   ├── 1_📊_Real_Time_Monitor.py
│       │   ├── 2_📈_Historical_Analysis.py
│       │   └── 3_🏭_Machine_Health.py
│       ├── components/
│       │   ├── __init__.py
│       │   ├── influx_client.py
│       │   └── charts.py
│       ├── config.py
│       ├── requirements.txt
│       ├── Dockerfile
│       └── tests/
│           ├── __init__.py
│           └── test_dashboard.py
├── tests/
│   └── integration/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_simulator_to_kafka.py
│       ├── test_kafka_to_influxdb.py
│       └── test_end_to_end.py
├── scripts/
│   ├── demo.py
│   ├── load_test.py
│   └── populate_demo_data.py
├── docs/
│   ├── WEEK_1_SUMMARY.md
│   ├── ARCHITECTURE_SCHEMA.md
│   ├── CREDENTIALS.md
│   └── LAUNCH_CHECKLIST.md
├── docker-compose.yml
├── Makefile
├── .env
├── .env.example
├── README.md
└── requirements-dev.txt
```

**Total New Files**: 34 files
**Total Lines of Code**: ~2500 lines (excluding tests and docs)
**Total Test Count**: 40 tests

---

**End of Week 1 Implementation Plan**

**Reference**: Full detailed plan available at [C:\Users\diego\.claude\plans\glittery-munching-tide.md](C:\Users\diego\.claude\plans\glittery-munching-tide.md)
