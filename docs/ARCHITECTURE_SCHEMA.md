# IndustrialMind - Architecture Schema & Component Guide

**Version**: 0.1.0-dev (Month 1 - Foundation Phase)
**Last Updated**: 2026-01-12
**Status**: Infrastructure Layer Complete ✅

---

## 📋 Table of Contents

1. [High-Level Architecture Overview](#high-level-architecture-overview)
2. [Infrastructure Layer (Current)](#infrastructure-layer-current)
3. [Application Layer (To Be Built)](#application-layer-to-be-built)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Component Details & Limitations](#component-details--limitations)
6. [What's Built vs What's Coming](#whats-built-vs-whats-coming)

---

## 🏗️ High-Level Architecture Overview

### The Complete System (12-Month Vision)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INDUSTRIALMIND PLATFORM                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: CLIENT TIER                                                         │
│ ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐            │
│ │   Streamlit     │  │   Next.js       │  │  Mobile App      │            │
│ │   Dashboard     │  │   Frontend      │  │  (Future)        │            │
│ │   (Month 2)     │  │   (Month 7)     │  │  (Month 11)      │            │
│ └─────────────────┘  └─────────────────┘  └──────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: API GATEWAY (Month 7+)                                             │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │  Kong/NGINX - Authentication, Rate Limiting, Routing                 │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: APPLICATION SERVICES                                                │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Data         │  │ ML Training  │  │ ML Inference │  │ Knowledge    │  │
│  │ Simulator    │  │ Service      │  │ APIs         │  │ Graph API    │  │
│  │ (Month 1)    │  │ (Month 2)    │  │ (Month 2)    │  │ (Month 5)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Data         │  │ RAG          │  │ LLM          │  │ Alert &      │  │
│  │ Ingestion    │  │ Service      │  │ Fine-tuning  │  │ Notification │  │
│  │ (Month 1)    │  │ (Month 6)    │  │ (Month 7)    │  │ (Month 3)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: EVENT STREAMING (✅ ACTIVE NOW)                                    │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │                     Apache Kafka + Zookeeper                          │   │
│ │  Topics: sensor-readings, anomaly-detected, predictions, alerts       │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: DATA STORAGE (✅ ACTIVE NOW)                                       │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  InfluxDB    │  │ PostgreSQL   │  │    Redis     │  │   Neo4J      │  │
│  │ Time Series  │  │  Relational  │  │   Cache      │  │    Graph     │  │
│  │   ✅ NOW     │  │   ✅ NOW     │  │   ✅ NOW     │  │  (Month 5)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                                              │
│  ┌──────────────┐                                                           │
│  │  ChromaDB    │                                                           │
│  │   Vector     │                                                           │
│  │  (Month 6)   │                                                           │
│  └──────────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6: MLOPS PLATFORM (✅ ACTIVE NOW)                                     │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │  MLflow - Experiment Tracking, Model Registry, Model Serving         │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │  DVC - Data Version Control (Month 3)                                │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 7: OBSERVABILITY (✅ ACTIVE NOW)                                      │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │  Prometheus (Metrics) + Grafana (Dashboards) + Loki (Logs) + Jaeger │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 8: CLOUD INFRASTRUCTURE (Month 9+)                                    │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │  Azure Kubernetes Service (AKS) + Azure ML + Azure Storage           │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
✅ = Currently running in Docker containers
🔨 = To be built (future months)
```

---

## 🐳 Infrastructure Layer (Current)

### What You Have Running NOW

These are the **9 Docker containers** currently configured and ready to start:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCKER HOST (Your Computer)                       │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │         Network: industrialmind-network                     │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │  EVENT STREAMING LAYER                                │  │   │
│  │  │  ┌─────────────────┐  ┌──────────────────────────┐   │  │   │
│  │  │  │  Zookeeper      │  │  Kafka                    │   │  │   │
│  │  │  │  Port: 2181     │  │  Ports: 9092/29092       │   │  │   │
│  │  │  │  Purpose:       │  │  Purpose:                 │   │  │   │
│  │  │  │  - Coordinates  │  │  - Message broker         │   │  │   │
│  │  │  │    Kafka        │  │  - Event streaming        │   │  │   │
│  │  │  │  - Manages      │  │  - Pub/sub system         │   │  │   │
│  │  │  │    cluster      │  │                           │   │  │   │
│  │  │  └─────────────────┘  └──────────────────────────┘   │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │  DATA STORAGE LAYER                                   │  │   │
│  │  │  ┌─────────────────┐  ┌──────────────────────────┐   │  │   │
│  │  │  │  InfluxDB       │  │  PostgreSQL               │   │  │   │
│  │  │  │  Port: 8086     │  │  Port: 5432              │   │  │   │
│  │  │  │  Purpose:       │  │  Purpose:                 │   │  │   │
│  │  │  │  - Store sensor │  │  - Metadata storage       │   │  │   │
│  │  │  │    time series  │  │  - MLflow backend         │   │  │   │
│  │  │  │  - Query data   │  │  - 3 databases:           │   │  │   │
│  │  │  │  - Downsampling │  │    * industrialmind       │   │  │   │
│  │  │  │                 │  │    * mlflow               │   │  │   │
│  │  │  │                 │  │    * airflow              │   │  │   │
│  │  │  └─────────────────┘  └──────────────────────────┘   │  │   │
│  │  │                                                        │  │   │
│  │  │  ┌─────────────────┐                                  │  │   │
│  │  │  │  Redis          │                                  │  │   │
│  │  │  │  Port: 6379     │                                  │  │   │
│  │  │  │  Purpose:       │                                  │  │   │
│  │  │  │  - Caching      │                                  │  │   │
│  │  │  │  - Session store│                                  │  │   │
│  │  │  │  - Rate limiting│                                  │  │   │
│  │  │  └─────────────────┘                                  │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │  MLOPS LAYER                                          │  │   │
│  │  │  ┌─────────────────┐                                  │  │   │
│  │  │  │  MLflow         │                                  │  │   │
│  │  │  │  Port: 5011     │                                  │  │   │
│  │  │  │  Purpose:       │                                  │  │   │
│  │  │  │  - Track        │                                  │  │   │
│  │  │  │    experiments  │                                  │  │   │
│  │  │  │  - Log metrics  │                                  │  │   │
│  │  │  │  - Model        │                                  │  │   │
│  │  │  │    registry     │                                  │  │   │
│  │  │  │  - Artifact     │                                  │  │   │
│  │  │  │    storage      │                                  │  │   │
│  │  │  └─────────────────┘                                  │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │  OBSERVABILITY LAYER                                  │  │   │
│  │  │  ┌─────────────────┐  ┌──────────────────────────┐   │  │   │
│  │  │  │  Prometheus     │  │  Grafana                  │   │  │   │
│  │  │  │  Port: 9090     │  │  Port: 3011              │   │  │   │
│  │  │  │  Purpose:       │  │  Purpose:                 │   │  │   │
│  │  │  │  - Collect      │  │  - Visualize metrics      │   │  │   │
│  │  │  │    metrics      │  │  - Create dashboards      │   │  │   │
│  │  │  │  - Time series  │  │  - Alerting               │   │  │   │
│  │  │  │    monitoring   │  │  - Query Prometheus       │   │  │   │
│  │  │  │  - Alerting     │  │                           │   │  │   │
│  │  │  └─────────────────┘  └──────────────────────────┘   │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  All containers connected via: industrialmind-network               │
│  Data persisted in: Docker volumes (survives restarts)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Component Details & Limitations

### 1. Zookeeper
**Container**: `industrialmind-zookeeper`
**Port**: 2181
**Image**: `confluentinc/cp-zookeeper:7.5.0`

**Purpose**:
- Coordinate Kafka brokers
- Manage Kafka cluster metadata
- Leader election for Kafka partitions
- Store Kafka configuration

**What It Does**:
- Runs as a service that Kafka depends on
- You won't interact with it directly
- It's required for Kafka to function

**Limitations**:
- Single node setup (not HA)
- No authentication (dev only)
- Stores data in Docker volume

**Data Storage**:
- Volume: `industrialmind_zookeeper-data`
- Volume: `industrialmind_zookeeper-logs`

---

### 2. Kafka
**Container**: `industrialmind-kafka`
**Ports**: 9092 (external), 29092 (internal)
**Image**: `confluentinc/cp-kafka:7.5.0`

**Purpose**:
- **Event streaming backbone** of the platform
- Acts as message broker between services
- Decouples data producers from consumers
- Provides fault-tolerant message storage

**What It Does**:
1. Receives sensor readings from Data Simulator
2. Stores messages in topics (queues)
3. Allows multiple consumers to read messages
4. Guarantees message order within partitions
5. Retains messages for replay

**How You'll Use It**:
```python
# Produce messages (Data Simulator)
producer.send('sensor-readings', sensor_data)

# Consume messages (Data Ingestion)
consumer = KafkaConsumer('sensor-readings')
for message in consumer:
    process(message.value)
```

**Topics Created** (via `make kafka-topics`):
1. `sensor-readings` - Raw sensor data stream
2. `anomaly-detected` - Detected anomalies
3. `maintenance-predictions` - RUL predictions
4. `alerts` - System alerts

**Limitations**:
- Single broker (not HA)
- 3 partitions per topic (moderate parallelism)
- Replication factor: 1 (no redundancy)
- No authentication/encryption
- Data retained for 7 days by default

**Data Storage**:
- Volume: `industrialmind_kafka-data`
- Size: Grows with message volume

---

### 3. InfluxDB
**Container**: `industrialmind-influxdb`
**Port**: 8086
**Image**: `influxdb:2.7`

**Purpose**:
- **Primary time-series database**
- Optimized for sensor data storage
- Fast writes and queries
- Automatic data downsampling

**What It Does**:
1. Stores all sensor readings with timestamps
2. Organizes data by measurement, tags, fields
3. Provides SQL-like query language (Flux)
4. Retention policies for data lifecycle
5. Continuous queries for aggregation

**Data Structure**:
```
Bucket: sensors
  Measurement: sensor_readings
    Tags: machine_id, sensor_id, location
    Fields: temperature, vibration, pressure, power
    Timestamp: nanosecond precision
```

**How You'll Use It**:
```python
# Write data (Data Ingestion Service)
from influxdb_client import InfluxDBClient

point = {
    "measurement": "sensor_readings",
    "tags": {"machine_id": "MACHINE_001"},
    "fields": {
        "temperature": 65.5,
        "vibration": 1.2
    },
    "time": datetime.utcnow()
}
write_api.write(bucket="sensors", record=point)

# Query data (Dashboard, ML Training)
query = 'from(bucket:"sensors") |> range(start: -1h)'
result = query_api.query(query)
```

**Limitations**:
- Single instance (not clustered)
- No replication
- Limited to available RAM/disk
- No authentication required (dev only)

**Data Storage**:
- Volume: `industrialmind_influxdb-data`
- Size: Grows with time-series data
- Expected: ~1GB per month of continuous data

**Access**:
- UI: http://localhost:8086
- Credentials: admin / password123
- Token: industrialmind-token-123456

---

### 4. PostgreSQL
**Container**: `industrialmind-postgres`
**Port**: 5432
**Image**: `postgres:15-alpine`

**Purpose**:
- **Relational database** for structured data
- Stores metadata, not time-series
- Backend for MLflow
- Application state storage

**Databases**:
1. **industrialmind** - Main application DB
   - Equipment metadata
   - Sensor configurations
   - Anomaly events
   - Alert history
   - User data (future)

2. **mlflow** - MLflow tracking backend
   - Experiment metadata
   - Run parameters
   - Metrics
   - Model registry

3. **airflow** - Workflow orchestration (future)
   - DAG definitions
   - Task history

**Schema** (industrialmind database):
```sql
-- Equipment metadata
equipment (id, machine_id, type, location, ...)

-- Sensor metadata
sensors (id, sensor_id, machine_id, sensor_type, ...)

-- Anomaly events
anomaly_events (id, timestamp, machine_id, anomaly_score, ...)

-- ML Models
models (id, model_name, version, mlflow_run_id, ...)

-- Alerts
alerts (id, timestamp, machine_id, alert_type, severity, ...)
```

**How You'll Use It**:
```python
# SQLAlchemy ORM
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://admin:password123@localhost:5432/industrialmind'
)

# Query equipment
equipment = session.query(Equipment).filter_by(machine_id='MACHINE_001').first()
```

**Limitations**:
- Single instance (not HA)
- No replication
- Connection limit: ~100 concurrent
- No connection pooling configured

**Data Storage**:
- Volume: `industrialmind_postgres-data`
- Size: Metadata only, small footprint

**Access**:
```bash
# CLI access
docker exec -it industrialmind-postgres psql -U admin -d industrialmind

# Connection string
postgresql://admin:password123@localhost:5432/industrialmind
```

---

### 5. Redis
**Container**: `industrialmind-redis`
**Port**: 6379
**Image**: `redis:7-alpine`

**Purpose**:
- **In-memory cache** for fast lookups
- Session storage
- Rate limiting
- Temporary data storage

**What It Does**:
1. Caches recent ML predictions
2. Stores API rate limit counters
3. Caches database query results
4. Stores session data
5. Pub/sub messaging (optional)

**How You'll Use It**:
```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# Cache prediction
r.setex(
    f'prediction:{machine_id}',
    300,  # 5 minute TTL
    json.dumps(prediction_result)
)

# Get cached prediction
cached = r.get(f'prediction:{machine_id}')
```

**Use Cases**:
1. **ML Inference Caching**:
   - Cache recent predictions
   - Avoid redundant model calls
   - Reduce inference latency

2. **API Rate Limiting**:
   - Track requests per user/IP
   - Implement sliding window limits

3. **Session Storage**:
   - Store user sessions
   - Authentication tokens (future)

**Limitations**:
- Single instance (not clustered)
- No persistence configured (data lost on restart)
- Memory-limited (constrained by Docker)
- No authentication (dev only)

**Data Storage**:
- Volume: `industrialmind_redis-data`
- In-memory + optional persistence
- Ephemeral by nature

---

### 6. MLflow
**Container**: `industrialmind-mlflow`
**Port**: 5011
**Image**: `ghcr.io/mlflow/mlflow:v2.8.1`

**Purpose**:
- **MLOps platform** for ML lifecycle
- Experiment tracking
- Model registry
- Model versioning
- Artifact storage

**What It Does**:
1. **Experiment Tracking**:
   - Log hyperparameters
   - Record metrics (loss, accuracy, F1)
   - Track training duration
   - Compare runs

2. **Model Registry**:
   - Version models
   - Stage models (staging/production)
   - Model lineage tracking
   - Metadata storage

3. **Artifact Storage**:
   - Store model files (.pt, .pkl)
   - Store plots/visualizations
   - Store training logs
   - Dataset snapshots

**How You'll Use It**:
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5011")
mlflow.set_experiment("anomaly-detection")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 64)

    # Train model
    model = train_autoencoder(...)

    # Log metrics
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("val_loss", val_loss)

    # Log model
    mlflow.pytorch.log_model(model, "model")
```

**Directory Structure**:
```
/mlflow/artifacts/
  ├── 0/  (experiment_id)
  │   ├── run_id_1/
  │   │   ├── artifacts/
  │   │   │   └── model/
  │   │   │       ├── model.pt
  │   │   │       └── requirements.txt
  │   │   └── metrics/
  │   └── run_id_2/
  └── 1/  (another experiment)
```

**Limitations**:
- Single instance
- Local artifact storage (not S3/Azure)
- No authentication
- Limited to Docker volume size

**Data Storage**:
- Backend: PostgreSQL (`mlflow` database)
- Artifacts: Volume `industrialmind_mlflow-artifacts`
- Expected size: Models ~100MB-1GB each

**Access**:
- UI: http://localhost:5011
- No login required
- Browse experiments, compare runs

---

### 7. Prometheus
**Container**: `industrialmind-prometheus`
**Port**: 9090
**Image**: `prom/prometheus:v2.47.0`

**Purpose**:
- **Metrics collection** and storage
- Time-series monitoring database
- Alerting engine
- Service health monitoring

**What It Does**:
1. **Scrapes Metrics**:
   - Polls services every 15 seconds
   - Collects metrics from `/metrics` endpoints
   - Stores time-series data

2. **Metrics Types**:
   - Counter: Cumulative (requests_total)
   - Gauge: Point-in-time (memory_usage)
   - Histogram: Distributions (latency_ms)
   - Summary: Quantiles

3. **Querying**:
   - PromQL query language
   - Aggregations, rates, percentiles
   - Grafana visualization

**Metrics You'll Collect**:
```python
# Application metrics (when services are built)
from prometheus_client import Counter, Histogram

# Request counter
requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

# Latency histogram
inference_latency = Histogram(
    'ml_inference_latency_seconds',
    'ML model inference latency',
    ['model']
)

# Usage
requests_total.labels('POST', '/predict', '200').inc()
inference_latency.labels('autoencoder').observe(0.045)
```

**How You'll Use It**:
```promql
# Query examples in Prometheus UI
rate(api_requests_total[5m])
histogram_quantile(0.99, ml_inference_latency_seconds)
up{job="anomaly-api"}
```

**Limitations**:
- Single instance
- Local storage only
- 15 day retention by default
- No HA or clustering

**Data Storage**:
- Volume: `industrialmind_prometheus-data`
- Size: Grows with metrics (~1GB/month)

**Access**:
- UI: http://localhost:9090
- Graph metrics, check targets
- Test queries

---

### 8. Grafana
**Container**: `industrialmind-grafana`
**Port**: 3011
**Image**: `grafana/grafana:10.1.0`

**Purpose**:
- **Visualization platform**
- Dashboard creation
- Metric exploration
- Alerting UI

**What It Does**:
1. **Dashboards**:
   - Create visual dashboards
   - Graph time-series data
   - Multiple data sources
   - Template variables

2. **Data Sources** (to be configured):
   - Prometheus (metrics)
   - InfluxDB (sensor data)
   - PostgreSQL (metadata)
   - Loki (logs, future)

3. **Panels**:
   - Time series graphs
   - Stat panels
   - Tables
   - Heatmaps

**Dashboards You'll Create**:
1. **System Health Dashboard**:
   - Service uptime
   - Resource usage (CPU, memory)
   - Request rates
   - Error rates

2. **ML Performance Dashboard**:
   - Model inference latency
   - Prediction distribution
   - Anomaly detection rate
   - Model accuracy over time

3. **Business Metrics Dashboard**:
   - Anomalies detected per hour
   - Equipment health scores
   - Maintenance scheduling
   - Alert statistics

4. **Sensor Data Dashboard**:
   - Real-time sensor readings
   - Historical trends
   - Anomaly markers
   - Equipment comparison

**How You'll Use It**:
1. Access http://localhost:3011
2. Login: admin / admin
3. Add data sources (Prometheus, InfluxDB)
4. Create/import dashboards
5. Set up alerts

**Limitations**:
- Single instance
- No SSO/LDAP (dev only)
- Limited to configured data sources

**Data Storage**:
- Volume: `industrialmind_grafana-data`
- Stores: Dashboards, settings, datasources

---

### 9. Neo4J (Month 5+, Currently Disabled)
**Purpose**: Graph database for equipment relationships

**When Enabled**:
- Port: 7474 (UI), 7687 (Bolt)
- Store equipment topology
- Relationship queries
- Root cause analysis

**Disabled Reason**: Not needed until Month 5 when you build the knowledge graph feature.

---

### 10. ChromaDB (Month 6+, Currently Disabled)
**Purpose**: Vector database for RAG system

**When Enabled**:
- Port: 8000
- Store document embeddings
- Similarity search
- LLM context retrieval

**Disabled Reason**: Not needed until Month 6 when you build the RAG assistant.

---

## 🔄 Data Flow Diagrams

### Flow 1: Real-Time Sensor Data (To Be Built)

```
┌─────────────────┐
│  Data Simulator │  ← YOU WILL BUILD THIS (Month 1, Week 1)
│  (Python)       │
└────────┬────────┘
         │ produce
         ▼
┌─────────────────┐
│  Kafka Topic    │  ✅ READY NOW (created via make kafka-topics)
│ sensor-readings │
└────────┬────────┘
         │
         ├─────────┬────────────┐
         │         │            │
         ▼         ▼            ▼
┌──────────┐  ┌────────┐  ┌──────────┐
│ InfluxDB │  │ ML API │  │Dashboard │  ← YOU WILL BUILD THESE
│ Consumer │  │(Real   │  │(Stream)  │
└──────────┘  │time)   │  └──────────┘
              └────┬───┘
                   │ if anomaly
                   ▼
         ┌──────────────────┐
         │  Kafka Topic     │  ✅ READY NOW
         │ anomaly-detected │
         └──────────────────┘
                   │
                   ├──────────┐
                   ▼          ▼
            ┌─────────┐  ┌────────┐
            │ Alert   │  │ Store  │  ← YOU WILL BUILD THESE
            │ Service │  │ in PG  │
            └─────────┘  └────────┘
```

**Status**:
- ✅ Kafka infrastructure ready
- ✅ InfluxDB ready to receive data
- 🔨 Data Simulator - TO BUILD (Week 1)
- 🔨 Data Ingestion Service - TO BUILD (Week 1)
- 🔨 ML API - TO BUILD (Month 2)
- 🔨 Dashboard - TO BUILD (Week 1)

---

### Flow 2: ML Training Pipeline (To Be Built)

```
┌──────────────────┐
│ Manual Trigger / │  ← YOU CONTROL THIS
│ Scheduled Job    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Training Service │  ← YOU WILL BUILD THIS (Month 2)
│ (Python)         │
└────────┬─────────┘
         │
         ├─ Query training data ──→ ┌──────────┐
         │                           │ InfluxDB │  ✅ READY NOW
         │                           └──────────┘
         │
         ├─ Version data ──→ ┌────┐
         │                    │DVC │  (Month 3)
         │                    └────┘
         │
         ├─ Train PyTorch model
         │  (autoencoder/transformer)
         │
         ├─ Log experiment ──→ ┌────────┐
         │                      │ MLflow │  ✅ READY NOW
         │                      │ - Params│
         │                      │ - Metrics│
         │                      └────────┘
         │
         ├─ Evaluate on validation set
         │
         ├─ Register model ──→ ┌────────────────┐
         │                      │ MLflow Registry│  ✅ READY NOW
         │                      │ - Version 1.0  │
         │                      │ - Version 1.1  │
         │                      └────────────────┘
         │
         └─ Deploy model ──→ ┌────────┐
                             │ ML API │  (Month 2)
                             └────────┘
```

**Status**:
- ✅ MLflow ready for experiment tracking
- ✅ InfluxDB ready for training data
- 🔨 Training scripts - TO BUILD (Month 2)
- 🔨 DVC setup - TO BUILD (Month 3)

---

## ✅ What's Built vs 🔨 What's Coming

### ✅ Infrastructure Complete (NOW)

**What's Running**:
1. ✅ 9 Docker containers configured
2. ✅ Docker network created
3. ✅ Docker volumes for persistence
4. ✅ All services health-checked
5. ✅ Kafka topics defined
6. ✅ PostgreSQL databases created
7. ✅ Port mappings configured
8. ✅ Environment variables set
9. ✅ Prometheus scrape config
10. ✅ Grafana ready for datasources

**What You Can Do Now**:
- ✅ Start infrastructure: `make up`
- ✅ Access all UIs (InfluxDB, MLflow, Grafana)
- ✅ View metrics in Prometheus
- ✅ Check logs: `make logs`
- ✅ Connect to databases
- ✅ Produce/consume Kafka messages manually

---

### 🔨 Application Layer (To Be Built)

**Week 1 (This Week)**:
1. 🔨 Data Simulator
   - Generate realistic sensor data
   - Inject anomalies
   - Produce to Kafka

2. 🔨 Data Ingestion Service
   - Consume from Kafka
   - Write to InfluxDB
   - Error handling

3. 🔨 Basic Streamlit Dashboard
   - Connect to InfluxDB
   - Display real-time sensor charts
   - Show data statistics

**Month 2**:
4. 🔨 PyTorch Autoencoder
   - Model architecture
   - Training script with MLflow
   - Evaluation

5. 🔨 ML Inference API
   - FastAPI service
   - Load model from MLflow
   - /predict endpoint

**Month 3+**:
6. 🔨 MLOps enhancements
7. 🔨 Transformer model
8. 🔨 Knowledge graph
9. 🔨 RAG system
10. 🔨 LLM fine-tuning
11. 🔨 Kubernetes deployment
12. 🔨 Azure migration

---

## 📊 Resource Requirements

### Current Infrastructure

**CPU**:
- Minimum: 4 cores
- Recommended: 8 cores
- Per container: ~0.5-1 core under load

**Memory**:
- Minimum: 8GB allocated to Docker
- Recommended: 12GB
- Breakdown:
  - Kafka: 2GB
  - InfluxDB: 1.5GB
  - PostgreSQL: 512MB
  - MLflow: 512MB
  - Prometheus: 512MB
  - Grafana: 256MB
  - Redis: 256MB
  - Zookeeper: 256MB
  - Overhead: 1GB

**Disk**:
- Minimum: 20GB free
- Expected growth:
  - Sensor data: ~1GB/month (InfluxDB)
  - ML models: ~500MB/month (MLflow)
  - Metrics: ~500MB/month (Prometheus)
  - Kafka: ~1GB/month
  - Logs: ~500MB/month
  - **Total: ~3.5GB/month**

**Network**:
- All inter-container communication internal
- External: Only web UIs and your applications

---

## 🎯 Summary

### What You Have:
1. **Complete infrastructure layer** - 9 services ready
2. **Event streaming** - Kafka topics created
3. **Data storage** - InfluxDB, PostgreSQL, Redis ready
4. **MLOps platform** - MLflow ready for experiments
5. **Observability** - Prometheus & Grafana ready
6. **Documentation** - All credentials and architecture explained

### What You Need to Build (Week 1):
1. **Data Simulator** - Python script to generate sensor data
2. **Data Ingestion** - Python service to consume Kafka → InfluxDB
3. **Dashboard** - Streamlit app to visualize data

### Architecture Evolution:
- **Month 1-3**: Modular monolith (all services in Docker Compose)
- **Month 4-6**: Extract ML services, add knowledge graph
- **Month 7-12**: Microservices + Kubernetes + Azure

---

**Questions to understand better?** Let me know what's unclear!

**Ready to start building?** Week 1 starts with the Data Simulator.
