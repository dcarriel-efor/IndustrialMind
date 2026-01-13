# IndustrialMind - Credentials & Access Guide

**⚠️ IMPORTANT**: This file contains development credentials. **DO NOT commit actual production credentials to Git!**

**Status**: Development Environment
**Last Updated**: 2026-01-12

---

## 🔐 Service Credentials

### InfluxDB
**URL**: http://localhost:8086
**Type**: Web UI + API

| Field | Value |
|-------|-------|
| Username | `admin` |
| Password | `password123` |
| Organization | `industrialmind` |
| Default Bucket | `sensors` |
| API Token | `industrialmind-token-123456` |

**First Login**:
1. Navigate to http://localhost:8086
2. Login with username/password
3. Token is auto-configured during setup

---

### PostgreSQL
**URL**: localhost:5432
**Type**: Database

| Field | Value |
|-------|-------|
| Host | `localhost` (or `postgres` from Docker network) |
| Port | `5432` |
| Username | `admin` |
| Password | `password123` |
| Default Database | `industrialmind` |
| MLflow Database | `mlflow` |
| Airflow Database | `airflow` |

**Connection String**:
```
postgresql://admin:password123@localhost:5432/industrialmind
```

**Connect via CLI**:
```bash
# From host
docker exec -it industrialmind-postgres psql -U admin -d industrialmind

# From another container
psql -h postgres -U admin -d industrialmind
```

---

### Redis
**URL**: localhost:6379
**Type**: In-memory cache

| Field | Value |
|-------|-------|
| Host | `localhost` (or `redis` from Docker network) |
| Port | `6379` |
| Password | None (no auth in dev) |
| Database | `0` |

**Connection String**:
```
redis://localhost:6379
```

**Connect via CLI**:
```bash
docker exec -it industrialmind-redis redis-cli
```

---

### MLflow
**URL**: http://localhost:5011
**Type**: Web UI + API

| Field | Value |
|-------|-------|
| Tracking URI | `http://localhost:5011` |
| Backend Store | PostgreSQL (`mlflow` database) |
| Artifact Store | `/mlflow/artifacts` (Docker volume) |

**Access**:
- No authentication required in dev environment
- Navigate to http://localhost:5011
- All experiments and runs visible immediately

**Python Connection**:
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5011")
```

---

### Grafana
**URL**: http://localhost:3011
**Type**: Web UI

| Field | Value |
|-------|-------|
| Username | `admin` |
| Password | `admin` |
| Root URL | `http://localhost:3011` |

**First Login**:
1. Navigate to http://localhost:3011
2. Login with admin/admin
3. Grafana will prompt you to change password (can skip for dev)
4. Dashboard is empty initially - will be configured later

---

### Prometheus
**URL**: http://localhost:9090
**Type**: Web UI + API

| Field | Value |
|-------|-------|
| Web UI | `http://localhost:9090` |
| API Endpoint | `http://localhost:9090/api/v1/query` |

**Access**:
- No authentication required
- Navigate to http://localhost:9090
- View metrics, targets, and alerts

---

### Kafka
**URL**: localhost:9092 (external), kafka:29092 (internal)
**Type**: Message Broker

| Field | Value |
|-------|-------|
| Bootstrap Servers (External) | `localhost:9092` |
| Bootstrap Servers (Internal) | `kafka:29092` |
| Zookeeper | `localhost:2181` |

**Access**:
- No authentication required in dev
- Use Kafka CLI tools or client libraries

**Example Consumer**:
```bash
docker exec -it industrialmind-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic sensor-readings \
  --from-beginning
```

---

### Neo4J (Month 5+, currently commented out)
**URL**: http://localhost:7474 (UI), bolt://localhost:7687 (Bolt)
**Type**: Graph Database

| Field | Value |
|-------|-------|
| Username | `neo4j` |
| Password | `password123` |
| Bolt URI | `bolt://localhost:7687` |
| HTTP URI | `http://localhost:7474` |

---

### ChromaDB (Month 5+, currently commented out)
**URL**: http://localhost:8000
**Type**: Vector Database

| Field | Value |
|-------|-------|
| API Endpoint | `http://localhost:8000` |
| Authentication | None (dev mode) |

---

## 🐳 Container Architecture Explained

### How Containers Work

Your infrastructure runs as **9 separate Docker containers**, each in isolation but networked together:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Computer (Host)                      │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │         Docker Network: industrialmind-network      │   │
│  │                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │   │
│  │  │ Container 1  │  │ Container 2  │  │Container3│ │   │
│  │  │  Zookeeper   │  │    Kafka     │  │ InfluxDB │ │   │
│  │  │   :2181      │  │ :9092/:29092 │  │  :8086   │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │   │
│  │                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │   │
│  │  │ Container 4  │  │ Container 5  │  │Container6│ │   │
│  │  │  PostgreSQL  │  │    Redis     │  │ MLflow   │ │   │
│  │  │   :5432      │  │   :6379      │  │  :5011   │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │   │
│  │                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ Container 7  │  │ Container 8  │               │   │
│  │  │  Prometheus  │  │   Grafana    │               │   │
│  │  │   :9090      │  │   :3011      │               │   │
│  │  └──────────────┘  └──────────────┘               │   │
│  │                                                      │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  Port Mappings (Host → Container):                         │
│  - localhost:8086  → InfluxDB:8086                         │
│  - localhost:5011  → MLflow:5000                           │
│  - localhost:3011  → Grafana:3000                          │
│  - localhost:9090  → Prometheus:9090                       │
│  - localhost:5432  → PostgreSQL:5432                       │
│  - localhost:6379  → Redis:6379                            │
│  - localhost:9092  → Kafka:9092                            │
│  - localhost:2181  → Zookeeper:2181                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts:

1. **Separate Containers**:
   - Each service (Kafka, InfluxDB, etc.) runs in its **own isolated container**
   - Each container is like a mini-virtual machine with its own filesystem
   - Containers can't see each other's files unless explicitly shared

2. **Docker Network**:
   - All containers are connected via `industrialmind-network`
   - Containers can talk to each other using **container names** (e.g., `kafka`, `postgres`)
   - Example: MLflow container connects to PostgreSQL using `postgres:5432`

3. **Port Mapping**:
   - Containers have **internal ports** (inside Docker)
   - These are **mapped** to **external ports** (on your computer)
   - Example: Grafana runs on port 3000 inside its container
   - But you access it at `localhost:3011` (external)

4. **Data Persistence (Volumes)**:
   - Each container has **persistent storage** via Docker volumes
   - Data survives even if you stop/restart containers
   - Volumes: `influxdb-data`, `postgres-data`, `kafka-data`, etc.
   - Located in Docker's volume directory (managed by Docker)

5. **Where Data Lives**:
   - **Inside Docker volumes**: Not directly on your filesystem
   - View volumes: `docker volume ls`
   - Inspect volume: `docker volume inspect industrialmind_influxdb-data`
   - Data persists until you run `make clean` (which deletes volumes)

### Container Communication Examples:

**From Your Code (Python) → Service**:
```python
# You use localhost + external port
influx_client = InfluxDBClient(url="http://localhost:8086", token="...")
```

**From One Container → Another Container**:
```python
# Inside a container, use container name + internal port
influx_client = InfluxDBClient(url="http://influxdb:8086", token="...")
```

**MLflow → PostgreSQL**:
```bash
# MLflow container connects to PostgreSQL container
postgresql://admin:password123@postgres:5432/mlflow
#                                 ^^^^^^^^
#                              container name
```

---

## 📂 Where Are Resources Stored?

### 1. **Container Filesystems** (Temporary)
- Each container has its own filesystem
- **Deleted** when container is removed
- Not directly accessible from host

### 2. **Docker Volumes** (Persistent)
These survive container restarts:

| Volume Name | Purpose | Container |
|-------------|---------|-----------|
| `industrialmind_zookeeper-data` | Zookeeper state | zookeeper |
| `industrialmind_zookeeper-logs` | Zookeeper logs | zookeeper |
| `industrialmind_kafka-data` | Kafka messages | kafka |
| `industrialmind_influxdb-data` | Time-series data | influxdb |
| `industrialmind_influxdb-config` | InfluxDB config | influxdb |
| `industrialmind_postgres-data` | All databases | postgres |
| `industrialmind_redis-data` | Redis persistence | redis |
| `industrialmind_mlflow-artifacts` | ML model files | mlflow |
| `industrialmind_prometheus-data` | Metrics history | prometheus |
| `industrialmind_grafana-data` | Dashboards/settings | grafana |

**View Volumes**:
```bash
docker volume ls | grep industrialmind
```

**Inspect Volume Location**:
```bash
docker volume inspect industrialmind_influxdb-data
# Shows actual path on disk (Docker manages this)
```

### 3. **Project Directory** (Your Code)
These are on your filesystem and **mounted** into containers:

| Path | Mounted To | Purpose |
|------|------------|---------|
| `./infrastructure/prometheus/prometheus.yml` | prometheus container | Config |
| `./infrastructure/grafana/*` | grafana container | Config |
| `./infrastructure/postgres/*.sh` | postgres container | Init scripts |

---

## 🔍 Accessing Container Data

### View Container Logs
```bash
# All services
make logs

# Specific service
docker logs industrialmind-kafka
docker logs industrialmind-influxdb -f  # Follow logs
```

### Execute Commands Inside Containers
```bash
# PostgreSQL
docker exec -it industrialmind-postgres psql -U admin -d industrialmind

# Redis
docker exec -it industrialmind-redis redis-cli

# Kafka
docker exec -it industrialmind-kafka kafka-topics --list --bootstrap-server localhost:9092

# Check files inside container
docker exec -it industrialmind-influxdb ls /var/lib/influxdb2
```

### Copy Files To/From Containers
```bash
# Copy from container to host
docker cp industrialmind-influxdb:/var/lib/influxdb2/backup.tar ./backup.tar

# Copy from host to container
docker cp ./config.yml industrialmind-prometheus:/etc/prometheus/config.yml
```

---

## 🚨 Security Notes

### Development Environment
- **All passwords are in plain text** - This is OK for local development
- **No encryption** between services
- **No firewall rules** - All ports accessible locally

### ⚠️ For Production (Later):
- [ ] Change all default passwords
- [ ] Enable authentication on all services
- [ ] Use secrets management (Azure Key Vault)
- [ ] Enable TLS/SSL for all connections
- [ ] Use strong, randomly generated passwords
- [ ] Implement network policies
- [ ] Enable audit logging

---

## 📋 Quick Reference Card

### Common Ports
```
8086  → InfluxDB UI/API
5432  → PostgreSQL
6379  → Redis
9092  → Kafka (external)
5011  → MLflow UI
3011  → Grafana UI
9090  → Prometheus UI
2181  → Zookeeper
```

### Common Usernames
```
admin → InfluxDB, PostgreSQL, Grafana
neo4j → Neo4J (when enabled)
```

### Common Passwords
```
password123 → Most services
```

### Container Names
```
industrialmind-zookeeper
industrialmind-kafka
industrialmind-influxdb
industrialmind-postgres
industrialmind-redis
industrialmind-mlflow
industrialmind-prometheus
industrialmind-grafana
```

---

## 🆘 Troubleshooting Access

### Can't Access Service UI

1. **Check if container is running**:
   ```bash
   docker ps | grep industrialmind
   ```

2. **Check if port is mapped correctly**:
   ```bash
   docker port industrialmind-grafana
   ```

3. **Check container logs**:
   ```bash
   docker logs industrialmind-grafana
   ```

4. **Verify port not in use**:
   ```bash
   netstat -ano | findstr "3011"  # Windows
   lsof -i :3011                  # Mac/Linux
   ```

### Connection Refused

- Wait 30-60 seconds after `make up` - services need time to initialize
- Check health status: `docker ps` (should show "healthy")
- Restart specific service: `docker restart industrialmind-<service>`

---

**Last Updated**: 2026-01-12
**Environment**: Development
**Security Level**: Low (Dev Only)
