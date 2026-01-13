# IndustrialMind Architecture Design Plan

## Status: Phase 2 - Architecture Design

### User Request
Design the overall system architecture for IndustrialMind Industrial AI Platform.

---

## Requirements Summary (from exploration)

### Core System Requirements
- **Real-time anomaly detection** (<100ms p99 latency)
- **Predictive maintenance** (time-to-failure forecasting)
- **Knowledge graph** of equipment relationships (1000+ nodes)
- **LLM-powered assistant** for technicians (RAG system)
- **Full MLOps pipeline** (tracking, registry, deployment, monitoring)

### Technical Constraints
- Fixed tech stack: Kafka, InfluxDB, PyTorch, Neo4J, MLflow, K8s, AWS
- 12-month phased approach (learning → production)
- Budget: Free tiers only
- Target: Portfolio-grade, production-ready code

### Non-Functional Requirements
- F1 score >0.90 for anomaly detection
- >80% test coverage
- Inference latency <100ms (p99)
- 1000+ sensor readings/minute throughput
- Horizontal scalability
- Production observability

---

## Architectural Approaches to Consider

### Approach 1: Monolithic Start, Progressive Decomposition
**Philosophy**: Start simple, decompose as complexity grows

**Structure**:
- Month 1-2: Single Python app with modules
- Month 3-4: Split into 3-4 services
- Month 5-8: Full microservices (8+ services)
- Month 9-12: Cloud-native with K8s

**Pros**:
- Faster initial development
- Easier debugging early on
- Natural learning progression
- Refactoring teaches architecture

**Cons**:
- Risk of tight coupling early
- May need significant refactoring
- Harder to demonstrate microservices skills initially

---

### Approach 2: Microservices from Day 1
**Philosophy**: Production architecture from the start

**Structure**:
```
industrialmind/
├── data-simulator/          # Service 1: Data generation
├── data-ingestion/          # Service 2: Kafka → InfluxDB
├── ml-inference-api/        # Service 3: Model serving
├── knowledge-graph-api/     # Service 4: Neo4J queries
├── llm-assistant-api/       # Service 5: RAG system
├── dashboard-frontend/      # Service 6: UI
├── mlops-service/          # Service 7: Training orchestration
└── monitoring/             # Service 8: Metrics export
```

**Pros**:
- Clear service boundaries from start
- Portfolio demonstrates microservices expertise
- Easier to scale individual components
- True production architecture

**Cons**:
- Higher initial complexity
- Slower initial development
- More DevOps overhead early
- May overcomplicate simple features

---

### Approach 3: Hybrid - Domain-Driven Services
**Philosophy**: Services based on bounded contexts, not technical layers

**Structure**:
```
Core Services (Month 1-4):
├── data-platform/          # Simulator + Ingestion + Storage
├── ml-platform/            # Training + Inference + Registry
└── dashboard/              # Visualization

Advanced Services (Month 5-12):
├── knowledge-platform/     # Neo4J + ChromaDB + RAG
├── deployment-platform/    # K8s + Terraform + CI/CD
└── observability/         # Monitoring + Alerting
```

**Pros**:
- Balanced complexity
- Services align with learning phases
- Clear domain boundaries
- Easier to reason about

**Cons**:
- Requires good domain understanding upfront
- May still need refactoring
- Less granular than pure microservices

---

---

## FINAL RECOMMENDED ARCHITECTURE

### Status: ✅ Complete - Ready for Implementation

---

## Executive Summary

**Recommended Approach**: **Progressive Decomposition Strategy**

Start with a well-structured modular monolith that naturally evolves into microservices as you learn and add complexity. This approach:
- Maximizes learning (you refactor as you grow)
- Minimizes initial complexity (faster to first results)
- Demonstrates architectural evolution (valuable for portfolio)
- Aligns perfectly with the 12-month phased roadmap

---

## Architecture Evolution Timeline

### Phase 1: Foundation (Months 1-3)
**Structure**: Modular monolith with clear bounded contexts

```
industrialmind/
├── docker-compose.yml              # All services in one compose file
├── services/
│   ├── data_platform/              # Data simulation + ingestion
│   │   ├── simulator/              # Sensor data generator
│   │   ├── ingestion/              # Kafka → InfluxDB consumer
│   │   └── api/                    # Data query API (FastAPI)
│   ├── ml_platform/                # ML training + inference
│   │   ├── models/                 # PyTorch model definitions
│   │   ├── training/               # Training pipeline
│   │   ├── inference/              # Serving API (FastAPI)
│   │   └── mlops/                  # MLflow tracking
│   └── dashboard/                  # Streamlit visualization
├── shared/
│   ├── schemas/                    # Kafka schemas (Avro/JSON)
│   ├── utils/                      # Common utilities
│   └── config/                     # Shared configuration
└── tests/                          # Comprehensive test suite
```

**Deployment**: Docker Compose locally
**Data Flow**: Simple and direct
**Focus**: Learning core ML and data engineering

### Phase 2: Hybrid Services (Months 4-6)
**Structure**: Extract high-value services

```
industrialmind/
├── docker-compose.yml
├── services/
│   ├── data-simulator/             # Extracted: Independent service
│   ├── data-ingestion/             # Extracted: Independent service
│   ├── ml-inference-api/           # Extracted: Model serving
│   ├── ml-training-service/        # Extracted: Training orchestration
│   ├── dashboard/                  # Streamlit UI
│   └── core-api/                   # Main business logic API
├── infrastructure/
│   ├── kafka/                      # Kafka cluster config
│   ├── influxdb/                   # InfluxDB config
│   └── mlflow/                     # MLflow server
└── tests/
```

**Deployment**: Docker Compose + Kubernetes (minikube)
**Data Flow**: Event-driven with Kafka
**Focus**: Service boundaries, API contracts, MLOps

### Phase 3: Full Microservices + Cloud (Months 7-12)
**Structure**: Complete microservices architecture

```
industrialmind/
├── services/
│   ├── data-simulator/             # Sensor data generation
│   ├── data-ingestion/             # Kafka → InfluxDB
│   ├── anomaly-detection-api/      # Real-time anomaly ML
│   ├── predictive-maintenance-api/ # Forecasting ML
│   ├── knowledge-graph-api/        # Neo4J queries
│   ├── rag-service/                # RAG + LLM
│   ├── llm-finetuning-service/     # Model fine-tuning
│   ├── alert-notification/         # Alerting system
│   ├── dashboard-frontend/         # Next.js UI
│   └── api-gateway/                # Kong/NGINX
├── infrastructure/
│   ├── terraform/                  # AWS EKS deployment
│   │   ├── vpc.tf
│   │   ├── eks.tf
│   │   ├── rds.tf
│   │   └── elasticache.tf
│   ├── kubernetes/                 # K8s manifests
│   │   ├── base/                   # Kustomize base
│   │   └── overlays/               # Env-specific
│   └── helm/                       # Helm charts
└── observability/
    ├── prometheus/
    ├── grafana/
    ├── loki/
    └── jaeger/
```

**Deployment**: AWS EKS with Terraform
**Data Flow**: Event-driven + CQRS
**Focus**: Scalability, observability, production operations

---

## High-Level System Architecture

### Architecture Diagram (Logical View)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT TIER                                │
│  [Streamlit Dashboard] → [Next.js Frontend] → [Mobile App (Future)]  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        API GATEWAY                                   │
│              [Kong/NGINX] - Auth, Rate Limiting, Routing             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
┌──────▼──────┐    ┌──────────▼─────────┐    ┌───────▼────────┐
│   Core API  │    │  ML Inference APIs  │    │ Knowledge APIs │
│   Service   │    │  - Anomaly Detect   │    │  - Neo4J       │
│  (FastAPI)  │    │  - Pred. Maint.     │    │  - RAG Service │
└──────┬──────┘    └──────────┬──────────┘    └───────┬────────┘
       │                      │                        │
┌──────▼──────────────────────▼────────────────────────▼─────────┐
│                     EVENT STREAMING TIER                        │
│                  [Apache Kafka Cluster]                         │
│  Topics: sensor-readings, anomaly-detected, predictions, alerts │
└─────┬────────────────────────────────────────────────┬──────────┘
      │                                                 │
┌─────▼──────────┐                           ┌─────────▼─────────┐
│  DATA TIER     │                           │   MLOPS TIER      │
│  - InfluxDB    │                           │   - MLflow        │
│  - PostgreSQL  │                           │   - DVC           │
│  - Redis       │                           │   - Feast         │
│  - Neo4J       │                           │   - Model Registry│
│  - ChromaDB    │                           └───────────────────┘
└────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────┐
│                    OBSERVABILITY TIER                           │
│  [Prometheus] [Grafana] [Loki] [Jaeger] [Alertmanager]         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Service Responsibilities

### 1. Data Simulator Service
**Purpose**: Generate realistic sensor data for development/testing
**Tech**: Python + Kafka Producer
**Responsibilities**:
- Simulate 4 sensors (temperature, vibration, pressure, power)
- Generate normal and anomalous patterns
- Configurable failure scenarios
- Produce to Kafka topic `sensor-readings`

**Key Files**:
- `services/data_platform/simulator/sensor_simulator.py`
- `services/data_platform/simulator/anomaly_generator.py`

### 2. Data Ingestion Service
**Purpose**: Consume sensor data from Kafka → InfluxDB
**Tech**: Python + Kafka Consumer + InfluxDB Client
**Responsibilities**:
- Subscribe to `sensor-readings` topic
- Validate and transform messages
- Write to InfluxDB with proper tags/fields
- Handle backpressure and retries

**Key Files**:
- `services/data_platform/ingestion/kafka_consumer.py`
- `services/data_platform/ingestion/influx_writer.py`

### 3. Anomaly Detection Service
**Purpose**: Real-time anomaly detection using autoencoder
**Tech**: FastAPI + PyTorch + Redis (caching)
**Responsibilities**:
- Load trained autoencoder model
- Provide `/predict` and `/predict/batch` endpoints
- Return anomaly score + threshold comparison
- Publish anomalies to `anomaly-detected` topic
- Cache recent predictions

**Key Files**:
- `services/ml_platform/inference/anomaly_api.py`
- `services/ml_platform/models/autoencoder.py`
- Uses skill: `Skills/api_design/fastapi_ml_service.md`

### 4. Predictive Maintenance Service
**Purpose**: Forecast time-to-failure using transformer
**Tech**: FastAPI + PyTorch
**Responsibilities**:
- Load trained transformer model
- Provide `/forecast` endpoint
- Return RUL (Remaining Useful Life) predictions
- Publish predictions to `maintenance-predictions` topic

**Key Files**:
- `services/ml_platform/inference/forecasting_api.py`
- `services/ml_platform/models/transformer_forecaster.py`

### 5. Knowledge Graph Service
**Purpose**: Equipment relationships and graph queries
**Tech**: FastAPI + Neo4J
**Responsibilities**:
- CRUD for equipment/component nodes
- Relationship management (CONTAINS, AFFECTS, etc.)
- Graph traversal queries
- Root cause analysis queries

**Key Files**:
- `services/knowledge_platform/graph_api.py`
- `services/knowledge_platform/neo4j_client.py`

### 6. RAG Service
**Purpose**: LLM-powered assistant with retrieval
**Tech**: FastAPI + LangChain + ChromaDB + OpenAI API
**Responsibilities**:
- Embed and store maintenance documents
- Vector similarity search
- Augmented generation with retrieved context
- Conversation history management

**Key Files**:
- `services/knowledge_platform/rag_api.py`
- `services/knowledge_platform/vector_store.py`

### 7. Alert & Notification Service
**Purpose**: Process events and trigger alerts
**Tech**: Python + Kafka Consumer + SMTP/Slack
**Responsibilities**:
- Subscribe to `anomaly-detected`, `maintenance-predictions`
- Apply alerting rules
- Send notifications (email, Slack, PagerDuty)
- Deduplicate alerts

**Key Files**:
- `services/core_api/alerting/alert_processor.py`
- `services/core_api/alerting/notification_engine.py`

### 8. ML Training Service
**Purpose**: Orchestrate model training pipelines
**Tech**: Python + MLflow + DVC
**Responsibilities**:
- Trigger training on schedule or API call
- Load data from InfluxDB
- Train models with hyperparameter tuning
- Log experiments to MLflow
- Register best models
- Deploy to serving layer

**Key Files**:
- `services/ml_platform/training/train_autoencoder.py`
- `services/ml_platform/training/train_transformer.py`
- Uses skill: `Skills/pytorch/training_loop.md`

### 9. Dashboard Service
**Purpose**: Visualization and monitoring UI
**Tech**: Streamlit (Months 1-4) → Next.js (Months 5-12)
**Responsibilities**:
- Real-time sensor data visualization
- Anomaly timeline
- Equipment health dashboard
- Model performance metrics

**Key Files**:
- `services/dashboard/app.py` (Streamlit)
- `services/dashboard-frontend/` (Next.js)

---

## Data Flow Architectures

### Flow 1: Real-Time Anomaly Detection

```
[Sensor/Simulator]
      │
      │ produce
      ▼
[Kafka: sensor-readings]
      │
      ├─────┬─────────────┐
      │     │             │
      ▼     ▼             ▼
 [InfluxDB] [Anomaly API] [Dashboard]
 (storage)   (inference)   (realtime)
              │
              │ if anomaly
              ▼
      [Kafka: anomaly-detected]
              │
              ├───────────┐
              ▼           ▼
         [Alert Svc]  [Knowledge Graph]
         (notify)     (root cause)
```

### Flow 2: Batch Training Pipeline

```
[Manual Trigger / Schedule]
      │
      ▼
[ML Training Service]
      │
      ├─ Load data ─→ [InfluxDB]
      │
      ├─ Version data ─→ [DVC]
      │
      ├─ Train model (PyTorch)
      │
      ├─ Log experiment ─→ [MLflow]
      │
      ├─ Evaluate on val set
      │
      ├─ Register if better ─→ [MLflow Model Registry]
      │
      └─ Deploy to serving ─→ [Anomaly Detection API]
```

### Flow 3: RAG Query Flow

```
[User Question via Dashboard]
      │
      ▼
[RAG Service]
      │
      ├─ Embed query ─→ [ChromaDB]
      │                    │
      │                    ├─ Vector search
      │                    └─ Return top-k docs
      │
      ├─ Graph query ─→ [Neo4J]
      │                   │
      │                   └─ Get related equipment
      │
      ├─ Build context (docs + graph)
      │
      └─ Generate answer ─→ [OpenAI API / Fine-tuned LLM]
                              │
                              └─ Return answer
```

---

## Data Schemas

### Kafka Topics

#### Topic: `sensor-readings`
```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "machine_id": "MACHINE_001",
  "sensor_id": "TEMP_001",
  "readings": {
    "temperature": 65.5,
    "vibration": 1.2,
    "pressure": 45.0,
    "power": 250.0
  },
  "metadata": {
    "location": "Factory_A_Line_1",
    "shift": "morning"
  }
}
```

#### Topic: `anomaly-detected`
```json
{
  "timestamp": "2024-01-15T10:30:05.000Z",
  "machine_id": "MACHINE_001",
  "anomaly_score": 0.087,
  "threshold": 0.05,
  "is_anomaly": true,
  "confidence": 0.92,
  "feature_contributions": {
    "temperature": 0.65,
    "vibration": 0.25,
    "pressure": 0.05,
    "power": 0.05
  },
  "model_version": "autoencoder_v1.2.0"
}
```

#### Topic: `maintenance-predictions`
```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "machine_id": "MACHINE_001",
  "predicted_rul_hours": 48.5,
  "confidence_interval": [42.0, 55.0],
  "recommended_action": "schedule_maintenance",
  "priority": "medium",
  "model_version": "transformer_v1.0.0"
}
```

### PostgreSQL Schema

```sql
-- Equipment metadata
CREATE TABLE equipment (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(50) UNIQUE NOT NULL,
    type VARCHAR(50) NOT NULL,
    location VARCHAR(100),
    installation_date DATE,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active'
);

-- Sensor metadata
CREATE TABLE sensors (
    id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(50) UNIQUE NOT NULL,
    machine_id VARCHAR(50) REFERENCES equipment(machine_id),
    sensor_type VARCHAR(50) NOT NULL,
    unit VARCHAR(20),
    normal_range_min FLOAT,
    normal_range_max FLOAT
);

-- Anomaly events
CREATE TABLE anomaly_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    machine_id VARCHAR(50) REFERENCES equipment(machine_id),
    anomaly_score FLOAT NOT NULL,
    is_confirmed BOOLEAN DEFAULT FALSE,
    root_cause TEXT,
    resolution TEXT,
    resolved_at TIMESTAMPTZ
);

-- ML Models
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50),
    registered_at TIMESTAMPTZ DEFAULT NOW(),
    mlflow_run_id VARCHAR(100),
    metrics JSONB,
    status VARCHAR(20) DEFAULT 'staging'
);

-- Alerts
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    machine_id VARCHAR(50),
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    message TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ
);
```

### Neo4J Graph Schema

```cypher
// Nodes
CREATE (e:Equipment {
  machine_id: 'MACHINE_001',
  type: 'CNC_Mill',
  location: 'Factory_A'
})

CREATE (c:Component {
  component_id: 'SPINDLE_001',
  type: 'Spindle',
  criticality: 'high'
})

CREATE (s:Sensor {
  sensor_id: 'TEMP_001',
  type: 'Temperature',
  location: 'spindle_bearing'
})

// Relationships
CREATE (e)-[:CONTAINS]->(c)
CREATE (c)-[:HAS_SENSOR]->(s)
CREATE (c1)-[:AFFECTS {impact: 'high'}]->(c2)
CREATE (e)-[:SIMILAR_TO {similarity: 0.85}]->(e2)
```

---

## Deployment Architecture

### Local Development (Docker Compose)

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Event Streaming
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on: [zookeeper]
    ports: ["9092:9092"]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092

  # Time Series Database
  influxdb:
    image: influxdb:2.7
    ports: ["8086:8086"]
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: password123
      DOCKER_INFLUXDB_INIT_ORG: industrialmind
      DOCKER_INFLUXDB_INIT_BUCKET: sensors

  # Relational Database
  postgres:
    image: postgres:15-alpine
    ports: ["5432:5432"]
    environment:
      POSTGRES_DB: industrialmind
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: password123

  # Caching
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  # Graph Database
  neo4j:
    image: neo4j:5.12
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/password123

  # Vector Database
  chromadb:
    image: chromadb/chroma:0.4.15
    ports: ["8000:8000"]

  # MLOps
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.1
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0 --backend-store-uri postgresql://admin:password123@postgres:5432/mlflow

  # Application Services
  data-simulator:
    build: ./services/data_platform/simulator
    depends_on: [kafka]
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092

  data-ingestion:
    build: ./services/data_platform/ingestion
    depends_on: [kafka, influxdb]
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      INFLUXDB_URL: http://influxdb:8086

  anomaly-api:
    build: ./services/ml_platform/inference
    ports: ["8001:8000"]
    depends_on: [redis, kafka]
    environment:
      MODEL_PATH: /models/autoencoder_best.pt
      REDIS_URL: redis://redis:6379

  dashboard:
    build: ./services/dashboard
    ports: ["8501:8501"]
    depends_on: [influxdb, postgres]

  # Observability
  prometheus:
    image: prom/prometheus:v2.47.0
    ports: ["9090:9090"]
    volumes:
      - ./infrastructure/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:10.1.0
    ports: ["3000:3000"]
    depends_on: [prometheus]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
```

### Production (AWS EKS with Terraform)

**Critical Infrastructure Files**:

1. `infrastructure/terraform/main.tf`
2. `infrastructure/terraform/eks.tf`
3. `infrastructure/terraform/vpc.tf`
4. `infrastructure/terraform/rds.tf`
5. `infrastructure/kubernetes/base/anomaly-detection-deployment.yaml`
6. `infrastructure/helm/industrialmind/values.yaml`

---

## Technology Justifications

### Why Kafka over RabbitMQ/AWS Kinesis?
- **Event sourcing**: Natural fit for time-series sensor data
- **Replay capability**: Can replay events for model retraining
- **Scalability**: Horizontal scaling with partitions
- **Industry standard**: Most enterprises use Kafka
- **Learning value**: More marketable skill
- **Free**: Runs locally without costs

### Why PyTorch over TensorFlow?
- **User's weak point**: Explicitly mentioned as needing improvement
- **Research focus**: Better for custom architectures (autoencoders, transformers)
- **Python-first**: More Pythonic API
- **Growing adoption**: Gaining enterprise traction
- **Portfolio value**: Demonstrates modern ML skills

### Why FastAPI over Flask/Django?
- **Performance**: Async support for ML serving
- **Auto docs**: OpenAPI/Swagger out of box
- **Type safety**: Pydantic validation
- **Modern**: Industry moving toward FastAPI
- **ML-friendly**: Excellent for model serving

### Why Progressive Decomposition?
- **Learning curve**: Manageable complexity growth
- **Portfolio story**: Shows architectural evolution
- **Practical**: Mimics real-world growth
- **Flexibility**: Can adjust based on learnings
- **Time-efficient**: Faster initial progress

---

## Scalability & Resilience Patterns

### Horizontal Scaling Strategy
- **Stateless services**: All APIs are stateless
- **Kafka partitioning**: Partition by machine_id
- **Database sharding**: Shard InfluxDB by time range
- **Caching**: Redis for hot data
- **Load balancing**: NGINX/Kong with round-robin

### Circuit Breaker Pattern
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_ml_inference(data):
    response = requests.post("http://anomaly-api/predict", json=data)
    return response.json()
```

### Retry with Exponential Backoff
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def write_to_influxdb(point):
    influx_client.write_api().write(bucket="sensors", record=point)
```

### Bulkhead Pattern
- **Resource isolation**: Separate thread pools per service
- **Database connection pools**: Limited connections per service
- **Rate limiting**: Per-service rate limits

---

## Observability Strategy

### Metrics (Prometheus)
**Service-level metrics**:
- Request rate (requests/sec)
- Error rate (errors/sec)
- Latency (p50, p95, p99)
- Saturation (CPU, memory, connections)

**ML-specific metrics**:
- `anomaly_detection_latency_ms` (histogram)
- `anomaly_score_distribution` (histogram)
- `model_prediction_count` (counter)
- `false_positive_rate` (gauge)
- `data_drift_score` (gauge)

### Logging (Loki)
**Structured logging**:
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "anomaly_detected",
    machine_id="MACHINE_001",
    anomaly_score=0.087,
    threshold=0.05,
    model_version="v1.2.0"
)
```

### Tracing (Jaeger)
- End-to-end request tracing
- Span: sensor reading → Kafka → inference → alert
- Performance bottleneck identification

### Alerting (Alertmanager)
**Critical alerts**:
- Service down (>5min)
- Error rate >5%
- p99 latency >100ms
- Model drift detected
- Data quality issues

### Dashboards (Grafana)
1. **System Health**: All services status, resource usage
2. **ML Performance**: Model metrics, prediction distribution
3. **Business Metrics**: Anomalies detected, maintenance scheduled
4. **Data Quality**: Missing data, outliers, schema violations
5. **SLO Dashboard**: Availability, latency, throughput SLOs

---

## Critical Files to Create (Month 1 - Week 1)

### Infrastructure Setup
1. `docker-compose.yml` - Complete local environment
2. `Makefile` - Common commands (up, down, logs, test)
3. `.env.example` - Environment variable template
4. `README.md` - Setup and run instructions

### Project Structure
5. `services/data_platform/simulator/sensor_simulator.py` - Data generator
6. `services/data_platform/ingestion/kafka_consumer.py` - Kafka → InfluxDB
7. `services/ml_platform/models/autoencoder.py` - Model definition
8. `services/dashboard/app.py` - Basic Streamlit dashboard

### Shared Code
9. `shared/schemas/sensor_reading.py` - Pydantic models
10. `shared/utils/kafka_utils.py` - Kafka producer/consumer helpers
11. `shared/config/settings.py` - Configuration management

### Testing
12. `tests/conftest.py` - Pytest fixtures
13. `tests/test_simulator.py` - Simulator tests
14. `tests/test_ingestion.py` - Ingestion tests

### Documentation
15. `docs/ARCHITECTURE.md` - This document (copy from plan)
16. `docs/SETUP.md` - Development environment setup
17. `docs/API.md` - API documentation

---

## Verification & Testing Strategy

### End-to-End Verification Flow

**Step 1: Infrastructure Up**
```bash
make docker-up
# Verify all services healthy
docker-compose ps
# All services should show "healthy" or "running"
```

**Step 2: Generate Sample Data**
```bash
# Start simulator
python services/data_platform/simulator/sensor_simulator.py
# Should see: "Produced 100 sensor readings to Kafka"
```

**Step 3: Verify Data Flow**
```bash
# Check Kafka topic
kafka-console-consumer --bootstrap-server localhost:9092 --topic sensor-readings --from-beginning
# Should see JSON sensor readings

# Check InfluxDB
influx query 'from(bucket:"sensors") |> range(start: -1h) |> limit(n: 10)'
# Should see 10 recent sensor readings
```

**Step 4: Dashboard Verification**
```bash
# Open Streamlit
open http://localhost:8501
# Should see real-time sensor charts updating
```

**Step 5: Run Tests**
```bash
pytest tests/ -v --cov=services --cov-report=html
# Should achieve >80% coverage
# All tests should pass
```

### Success Criteria for Month 1
- [ ] Docker Compose brings up all infrastructure services
- [ ] Simulator generates realistic sensor data
- [ ] Data flows from Kafka → InfluxDB correctly
- [ ] Dashboard displays real-time sensor charts
- [ ] Tests pass with >80% coverage
- [ ] README allows someone else to run the project

---

## Implementation Priorities (First Week)

### Day 1-2: Infrastructure
1. Create `docker-compose.yml` with all services
2. Create `Makefile` for common commands
3. Test full stack comes up cleanly
4. Document setup in README

### Day 3-4: Data Pipeline
1. Implement sensor simulator
2. Implement Kafka → InfluxDB ingestion
3. Test end-to-end data flow
4. Add monitoring/logging

### Day 5-6: Dashboard
1. Create basic Streamlit app
2. Connect to InfluxDB
3. Display real-time sensor charts
4. Add basic filtering

### Day 7: Testing & Documentation
1. Write integration tests
2. Achieve >80% coverage
3. Complete setup documentation
4. Tag v0.1.0 release

---

## Risk Mitigation

### Risk 1: Overengineering Early
**Mitigation**: Start with simplest working version, resist adding "nice-to-haves"

### Risk 2: Local Resource Constraints
**Mitigation**: Use lightweight configurations, stop unused containers, add resource limits

### Risk 3: Integration Complexity
**Mitigation**: Test each integration independently before combining

### Risk 4: Time Management
**Mitigation**: Timebox tasks, use skills as templates, ask for help when stuck

---

## Next Steps After Plan Approval

1. **Create project structure** using recommended folder layout
2. **Set up Docker Compose** with all infrastructure services
3. **Implement data simulator** (Week 1 priority)
4. **Begin Month 1, Week 1 tasks** from ORGANIZATIONAL_TASKS.md
5. **Use existing skills** from Skills/ folder as templates
6. **Commit frequently** with descriptive messages
7. **Test continuously** - don't let technical debt accumulate

---

## Summary

This architecture provides:
- ✅ **Clear learning path**: Complexity grows with your skills
- ✅ **Portfolio value**: Demonstrates architectural evolution
- ✅ **Production patterns**: Real-world practices throughout
- ✅ **Technology coverage**: All required tech stack included
- ✅ **Scalability**: Designed for growth from day 1
- ✅ **Observability**: Full monitoring/logging from start
- ✅ **Testability**: >80% coverage achievable
- ✅ **Budget-friendly**: Runs entirely on free tiers

**Estimated Time to First Working System**: 1 week
**Estimated Time to Production-Grade Platform**: 12 months (as planned)

