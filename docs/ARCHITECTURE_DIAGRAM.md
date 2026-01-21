# IndustrialMind Architecture Diagram

## System Overview

```mermaid
graph TB
    subgraph "Data Sources"
        SIM[Sensor Simulator<br/>5 machines × 4 sensors<br/>1200+ readings/min]
    end

    subgraph "Streaming Layer"
        KAFKA[Apache Kafka<br/>Topic: sensor-readings<br/>Partitioned by machine_id]
    end

    subgraph "Data Platform Services"
        ING[Ingestion Service<br/>Batch: 100 points<br/>Validation & Transform]
        API_DP[Data Query API<br/>FastAPI<br/>Future]
    end

    subgraph "Storage Layer"
        INFLUX[(InfluxDB<br/>Time-Series DB<br/>30-day retention)]
        POSTGRES[(PostgreSQL<br/>Metadata & MLflow)]
        REDIS[(Redis<br/>Cache Layer)]
    end

    subgraph "ML Platform Services"
        TRAIN[Training Pipeline<br/>PyTorch Models<br/>Month 2+]
        INFER[Inference API<br/>FastAPI<br/>< 100ms p99]
        MLFLOW[MLflow Server<br/>Experiment Tracking<br/>Model Registry]
        DVC[DVC<br/>Data Versioning]
    end

    subgraph "ML Models"
        AE[Autoencoder<br/>Anomaly Detection<br/>F1 > 0.90]
        TRANS[Transformer-LSTM<br/>RUL Forecasting]
        ENSEMBLE[Ensemble Models<br/>Multi-task Learning]
    end

    subgraph "Knowledge & AI Services"
        NEO4J[(Neo4J<br/>Equipment Graph<br/>1000+ nodes)]
        CHROMA[(ChromaDB<br/>Vector Store<br/>Embeddings)]
        RAG[RAG Service<br/>LangChain<br/>Technician Queries]
    end

    subgraph "Frontend Applications"
        DASH_ST[Streamlit Dashboard<br/>Real-time Monitoring<br/>Auto-refresh 2s]
        DASH_NEXT[Next.js Dashboard<br/>Advanced UI<br/>Future]
        API_REST[REST API<br/>External Integration]
    end

    subgraph "Observability Stack"
        PROM[Prometheus<br/>Metrics Collection<br/>:9090]
        GRAF[Grafana<br/>Dashboards & Alerts<br/>:3011]
        LOKI[Loki<br/>Log Aggregation<br/>:3100]
        PROMTAIL[Promtail<br/>Log Shipper]
    end

    subgraph "Deployment & Orchestration"
        DOCKER[Docker Compose<br/>Local Development<br/>12 Services]
        K8S[Kubernetes/EKS<br/>Production<br/>HPA + Monitoring]
        TERRA[Terraform<br/>Infrastructure as Code<br/>AWS Resources]
    end

    %% Data Flow
    SIM -->|Publish| KAFKA
    KAFKA -->|Consume| ING
    ING -->|Write Batch| INFLUX
    ING -->|Logs| PROMTAIL

    %% Query Flow
    INFLUX -->|Query| API_DP
    INFLUX -->|Query| DASH_ST
    API_DP -->|Cached| REDIS

    %% ML Training Flow
    INFLUX -->|Training Data| TRAIN
    TRAIN -->|Experiments| MLFLOW
    TRAIN -->|Data Version| DVC
    MLFLOW -->|Store| POSTGRES
    TRAIN -->|Register| AE
    TRAIN -->|Register| TRANS
    TRAIN -->|Register| ENSEMBLE

    %% ML Inference Flow
    KAFKA -->|Real-time| INFER
    INFER -->|Load Model| AE
    INFER -->|Load Model| TRANS
    INFER -->|Predict| DASH_ST
    INFER -->|Predict| API_REST

    %% Knowledge Graph Flow
    INFLUX -->|Historical| NEO4J
    NEO4J -->|Relationships| RAG
    TRAIN -->|Embeddings| CHROMA
    CHROMA -->|Semantic Search| RAG
    RAG -->|Answers| DASH_ST

    %% Monitoring Flow
    ING -.->|Metrics| PROM
    INFER -.->|Metrics| PROM
    TRAIN -.->|Metrics| PROM
    KAFKA -.->|Metrics| PROM
    PROM -->|Query| GRAF
    PROMTAIL -->|Ship| LOKI
    LOKI -->|Query| GRAF

    %% Deployment
    DOCKER -->|Container| SIM
    DOCKER -->|Container| ING
    DOCKER -->|Container| DASH_ST
    K8S -->|Orchestrate| INFER
    K8S -->|Orchestrate| TRAIN
    TERRA -->|Provision| K8S

    %% Styling
    classDef dataSource fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    classDef streaming fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef storage fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef ml fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef frontend fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef observ fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    classDef deploy fill:#e0e0e0,stroke:#424242,stroke-width:2px

    class SIM dataSource
    class KAFKA streaming
    class INFLUX,POSTGRES,REDIS,NEO4J,CHROMA storage
    class TRAIN,INFER,MLFLOW,AE,TRANS,ENSEMBLE,RAG,DVC ml
    class DASH_ST,DASH_NEXT,API_REST,API_DP frontend
    class PROM,GRAF,LOKI,PROMTAIL observ
    class DOCKER,K8S,TERRA deploy
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant Sim as Sensor Simulator
    participant K as Kafka
    participant Ing as Ingestion
    participant DB as InfluxDB
    participant ML as ML Inference
    participant Dash as Dashboard
    participant Obs as Prometheus

    Note over Sim,Dash: Real-time Data Pipeline (< 3s end-to-end)

    Sim->>K: Publish reading (1/sec × 5 machines)
    K->>Ing: Consume batch (max 100)
    Ing->>Ing: Validate with Pydantic
    Ing->>DB: Write batch
    Ing->>Obs: Export metrics

    par Real-time Display
        Dash->>DB: Query last 10 minutes
        DB-->>Dash: Return time-series
        Dash->>Dash: Render charts (2s refresh)
    and ML Inference
        K->>ML: Stream for prediction
        ML->>ML: Run autoencoder
        ML->>Dash: Send anomaly alert
        ML->>Obs: Export inference metrics
    end

    Note over Sim,Dash: Latency: Sim→Kafka (10ms), Kafka→DB (30ms), DB→Dash (150ms)
```

## ML Training Pipeline

```mermaid
graph LR
    subgraph "Data Preparation"
        RAW[Raw Sensor Data<br/>InfluxDB]
        FEAT[Feature Engineering<br/>Windows, Aggregations]
        SPLIT[Train/Val/Test Split<br/>80/10/10]
    end

    subgraph "Model Training"
        LOAD[DataLoader<br/>Batch: 32<br/>Workers: 4]
        MODEL[PyTorch Model<br/>Autoencoder/Transformer]
        OPT[Optimizer<br/>AdamW + Scheduler]
        LOSS[Loss Function<br/>MSE/BCE/Custom]
    end

    subgraph "Experiment Tracking"
        MLFLOW_TRACK[MLflow Tracking<br/>Params, Metrics, Artifacts]
        MLFLOW_REG[Model Registry<br/>Staging → Production]
    end

    subgraph "Evaluation & Deployment"
        EVAL[Evaluation<br/>F1, Precision, Recall<br/>RMSE, MAE]
        TEST[Test Set Performance<br/>Confusion Matrix]
        DEPLOY[Deploy to Inference<br/>ONNX Export]
    end

    RAW --> FEAT
    FEAT --> SPLIT
    SPLIT --> LOAD
    LOAD --> MODEL
    MODEL --> OPT
    OPT --> LOSS
    LOSS -->|Backward| MODEL
    MODEL --> MLFLOW_TRACK
    MLFLOW_TRACK --> EVAL
    EVAL --> TEST
    TEST -->|Pass Threshold| MLFLOW_REG
    MLFLOW_REG --> DEPLOY

    classDef data fill:#e3f2fd,stroke:#1976d2
    classDef train fill:#e8f5e9,stroke:#388e3c
    classDef track fill:#fff3e0,stroke:#f57c00
    classDef eval fill:#fce4ec,stroke:#c2185b

    class RAW,FEAT,SPLIT data
    class LOAD,MODEL,OPT,LOSS train
    class MLFLOW_TRACK,MLFLOW_REG track
    class EVAL,TEST,DEPLOY eval
```

## Microservices Communication

```mermaid
graph TB
    subgraph "Port Mapping"
        P8501[":8501<br/>Streamlit"]
        P3000[":3000<br/>Next.js"]
        P8000[":8000<br/>FastAPI Data"]
        P8001[":8001<br/>FastAPI ML"]
        P9092[":9092<br/>Kafka"]
        P8086[":8086<br/>InfluxDB"]
        P5432[":5432<br/>PostgreSQL"]
        P6379[":6379<br/>Redis"]
        P7687[":7687<br/>Neo4J"]
        P5011[":5011<br/>MLflow"]
        P9090[":9090<br/>Prometheus"]
        P3011[":3011<br/>Grafana"]
        P3100[":3100<br/>Loki"]
    end

    P8501 -->|HTTP| P8000
    P8501 -->|HTTP| P8001
    P8501 -->|Query| P8086
    P8000 -->|Query| P8086
    P8000 -->|Cache| P6379
    P8001 -->|Load Model| P5011
    P8001 -->|Consume| P9092
    P8001 -->|Query Graph| P7687
    P5011 -->|Store| P5432
    P3011 -->|Query| P9090
    P3011 -->|Query| P3100

    style P8501 fill:#e1bee7
    style P3000 fill:#e1bee7
    style P8000 fill:#c5e1a5
    style P8001 fill:#c5e1a5
    style P9092 fill:#fff59d
    style P8086 fill:#b3e5fc
    style P5432 fill:#b3e5fc
    style P6379 fill:#b3e5fc
    style P7687 fill:#b3e5fc
    style P5011 fill:#ffccbc
    style P9090 fill:#dce775
    style P3011 fill:#dce775
    style P3100 fill:#dce775
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development"
        DEV[Local Development<br/>Docker Compose<br/>12 Services]
    end

    subgraph "CI/CD Pipeline"
        GIT[Git Push]
        GHA[GitHub Actions<br/>Test + Build]
        REG[Container Registry<br/>ECR/GCR]
    end

    subgraph "Staging"
        STAGE_K8S[Kubernetes Staging<br/>2 Replicas]
        STAGE_TEST[Integration Tests<br/>Performance Tests]
    end

    subgraph "Production - AWS"
        ALB[Application Load Balancer]
        EKS[EKS Cluster<br/>Multi-AZ]

        subgraph "Worker Nodes"
            POD1[Inference Pod 1<br/>HPA: 2-10 replicas]
            POD2[Inference Pod 2]
            POD3[Dashboard Pod]
        end

        RDS[(RDS PostgreSQL<br/>Multi-AZ)]
        ELASTICACHE[(ElastiCache Redis<br/>Cluster Mode)]
        MSK[Amazon MSK<br/>Kafka Managed]
        S3[S3 Bucket<br/>Model Artifacts<br/>Training Data]
        CW[CloudWatch<br/>Logs + Metrics]
    end

    DEV --> GIT
    GIT --> GHA
    GHA -->|Build Image| REG
    REG --> STAGE_K8S
    STAGE_K8S --> STAGE_TEST
    STAGE_TEST -->|Approved| ALB
    ALB --> EKS
    EKS --> POD1
    EKS --> POD2
    EKS --> POD3
    POD1 --> RDS
    POD1 --> ELASTICACHE
    POD1 --> MSK
    POD1 --> S3
    POD2 --> RDS
    POD2 --> ELASTICACHE
    POD3 --> RDS
    EKS -.->|Logs/Metrics| CW

    classDef dev fill:#e8f5e9,stroke:#388e3c
    classDef ci fill:#fff3e0,stroke:#f57c00
    classDef stage fill:#e1f5ff,stroke:#0288d1
    classDef prod fill:#fce4ec,stroke:#c2185b
    classDef aws fill:#fff9c4,stroke:#f57c00

    class DEV dev
    class GIT,GHA,REG ci
    class STAGE_K8S,STAGE_TEST stage
    class ALB,EKS,POD1,POD2,POD3 prod
    class RDS,ELASTICACHE,MSK,S3,CW aws
```

## Technology Stack Summary

```mermaid
mindmap
  root((IndustrialMind<br/>Tech Stack))
    Data Layer
      Apache Kafka 7.5
      InfluxDB 2.7
      PostgreSQL 15
      Redis 7
      Neo4J 5.12
      ChromaDB
    ML/AI
      PyTorch 2.0+
      Transformers
      LangChain
      ONNX
      scikit-learn
    MLOps
      MLflow 2.8
      DVC
      Weights & Biases
      TensorBoard
    APIs
      FastAPI
      Pydantic
      Streamlit 1.29
      Next.js
    Observability
      Prometheus 2.47
      Grafana 10.1
      Loki 2.9
      Promtail
      Jaeger
    Infrastructure
      Docker
      Kubernetes
      Helm
      Terraform
      GitHub Actions
    Languages
      Python 3.11
      TypeScript
      SQL
      HCL Terraform
```

## Component Status Matrix

| Component | Status | Coverage | Performance | Documentation |
|-----------|--------|----------|-------------|---------------|
| **Sensor Simulator** | ✅ Deployed | 85% | 1200+/min | ✅ Complete |
| **Kafka Streaming** | ✅ Deployed | N/A | <10ms latency | ✅ Complete |
| **Ingestion Service** | ✅ Deployed | 78% | 30ms writes | ✅ Complete |
| **InfluxDB Storage** | ✅ Deployed | N/A | 150ms queries | ✅ Complete |
| **Streamlit Dashboard** | ✅ Deployed | 82% | 2s refresh | ✅ Complete |
| **Observability Stack** | ✅ Deployed | N/A | Real-time | ✅ Complete |
| **PyTorch Autoencoder** | 🏗️ Month 2 | Target 90% | Target <100ms | 📝 In Progress |
| **Transformer Model** | 📅 Month 3 | Target 90% | Target <100ms | 📋 Planned |
| **MLflow Integration** | 📅 Month 3 | N/A | N/A | 📋 Planned |
| **FastAPI Inference** | 📅 Month 4 | Target 85% | Target <50ms | 📋 Planned |
| **Neo4J Graph** | 📅 Month 5 | N/A | <500ms | 📋 Planned |
| **RAG System** | 📅 Month 6 | Target 80% | <2s | 📋 Planned |
| **LLM Fine-tuning** | 📅 Month 7 | N/A | <1s | 📋 Planned |
| **Kubernetes Deploy** | 📅 Month 8 | N/A | Auto-scale | 📋 Planned |

**Legend**: ✅ Deployed | 🏗️ In Development | 📅 Planned | 📝 Documenting | 📋 Planning

## Performance Metrics

```mermaid
graph LR
    subgraph "Data Pipeline"
        A[Simulator<br/>1200+ readings/min] -->|<10ms| B[Kafka]
        B -->|<100ms| C[Ingestion<br/>Batch 100]
        C -->|~30ms| D[InfluxDB]
        D -->|~150ms| E[Dashboard<br/>2s refresh]
    end

    subgraph "ML Inference Future"
        F[Real-time Stream] -->|<50ms| G[Model Inference<br/>p99 <100ms]
        G -->|<20ms| H[Alert Service]
    end

    subgraph "Target SLAs"
        I[End-to-end: <3s<br/>Uptime: 99.9%<br/>Throughput: 10K+ req/min]
    end

    style A fill:#c8e6c9
    style E fill:#ffccbc
    style G fill:#b3e5fc
    style I fill:#fff9c4
```

---

## Quick Navigation

- **Main README**: [README.md](../README.md)
- **Detailed Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Logging Implementation**: [LOGGING_IMPLEMENTATION.md](LOGGING_IMPLEMENTATION.md)
- **Project Objectives**: [PROJECT_OBJECTIVES.md](PROJECT_OBJECTIVES.md)
- **Setup Guide**: [SETUP.md](SETUP.md) (if exists)

---

## Contributing

This is a portfolio project, but feedback and suggestions are welcome! Please see the main README for contact information.

---

**Last Updated**: 2026-01-19 | **Project Phase**: Month 1, Week 1-2 Complete | **Version**: v0.1.0
