# IndustrialMind Infrastructure Launch Checklist

**Date**: 2026-01-12
**Version**: 0.1.0-dev

This checklist ensures all prerequisites are met before launching the infrastructure.

---

## ✅ Pre-Launch Checklist

### 1. System Requirements

- [ ] **Docker Desktop** installed and running (>= 20.10)
  ```bash
  docker --version
  docker-compose --version
  ```

- [ ] **Docker Resources Allocated**
  - Memory: At least 8GB available for Docker
  - CPU: At least 4 cores recommended
  - Disk: At least 20GB free space

- [ ] **Python** installed (>= 3.11)
  ```bash
  python --version
  ```

- [ ] **Git** installed
  ```bash
  git --version
  ```

### 2. Port Availability

Verify these ports are not in use by other services:

- [ ] **2181** - Zookeeper
- [ ] **9092** - Kafka (external)
- [ ] **29092** - Kafka (internal)
- [ ] **8086** - InfluxDB
- [ ] **5432** - PostgreSQL
- [ ] **6379** - Redis
- [ ] **5011** - MLflow (changed from 5000)
- [ ] **9090** - Prometheus
- [ ] **3011** - Grafana (changed from 3000)

Check ports on Windows:
```powershell
netstat -ano | findstr "2181 9092 8086 5432 6379 5011 9090 3011"
```

Check ports on Linux/Mac:
```bash
netstat -tuln | grep -E '2181|9092|8086|5432|6379|5011|9090|3011'
```

### 3. Project Files

- [x] **docker-compose.yml** - Infrastructure configuration
- [x] **Makefile** - Common commands
- [x] **.env** - Environment variables (copied from .env.example)
- [x] **infrastructure/prometheus/prometheus.yml** - Prometheus config
- [x] **infrastructure/postgres/init-multiple-dbs.sh** - PostgreSQL init script
- [x] **README.md** - Documentation
- [x] **docs/ARCHITECTURE.md** - Architecture documentation

### 4. Directory Structure

- [x] `services/` - Application services directory
- [x] `shared/schemas/` - Pydantic models
- [x] `tests/` - Test directory
- [x] `infrastructure/` - Infrastructure configs
- [x] `docs/` - Documentation

---

## 🚀 Launch Steps

### Step 1: Start Infrastructure

```bash
# Start all services
make up

# Expected output:
# - Creating network "industrialmind-network"
# - Creating volume "industrialmind_zookeeper-data"
# - Creating volume "industrialmind_kafka-data"
# - Creating volume "industrialmind_influxdb-data"
# - Creating volume "industrialmind_postgres-data"
# - Creating volume "industrialmind_redis-data"
# - Creating volume "industrialmind_mlflow-artifacts"
# - Creating volume "industrialmind_prometheus-data"
# - Creating volume "industrialmind_grafana-data"
# - Creating industrialmind-zookeeper
# - Creating industrialmind-postgres
# - Creating industrialmind-redis
# - Creating industrialmind-prometheus
# - Creating industrialmind-kafka
# - Creating industrialmind-influxdb
# - Creating industrialmind-mlflow
# - Creating industrialmind-grafana
```

### Step 2: Verify Services Health

```bash
# Check all services are running
make ps

# Expected output: All services should show "Up" or "healthy"
```

Verify each service individually:

#### Zookeeper
```bash
docker exec industrialmind-zookeeper zkServer.sh status
# Expected: Mode: standalone
```

#### Kafka
```bash
docker exec industrialmind-kafka kafka-broker-api-versions --bootstrap-server localhost:9092
# Expected: List of API versions
```

#### InfluxDB
```bash
curl http://localhost:8086/health
# Expected: {"name":"influxdb","message":"ready for queries and writes","status":"pass"}
```

#### PostgreSQL
```bash
docker exec industrialmind-postgres psql -U admin -d industrialmind -c "SELECT 1;"
# Expected: 1 (1 row)
```

#### Redis
```bash
docker exec industrialmind-redis redis-cli ping
# Expected: PONG
```

#### MLflow
```bash
curl http://localhost:5011/health
# Expected: {"status":"ok"}
```

#### Prometheus
```bash
curl http://localhost:9090/-/healthy
# Expected: Prometheus is Healthy.
```

#### Grafana
```bash
curl http://localhost:3011/api/health
# Expected: {"commit":"...","database":"ok","version":"..."}
```

### Step 3: Create Kafka Topics

```bash
# Create all required Kafka topics
make kafka-topics

# Expected output:
# Created topic sensor-readings.
# Created topic anomaly-detected.
# Created topic maintenance-predictions.
# Created topic alerts.
```

Verify topics were created:
```bash
docker exec industrialmind-kafka kafka-topics --list --bootstrap-server localhost:9092
# Expected:
# alerts
# anomaly-detected
# maintenance-predictions
# sensor-readings
```

### Step 4: Verify Service UIs

Open in browser and verify:

- [ ] **InfluxDB**: http://localhost:8086
  - Login: admin / password123
  - Should see InfluxDB UI
  - Verify "sensors" bucket exists

- [ ] **MLflow**: http://localhost:5011
  - Should see MLflow Tracking UI
  - No experiments yet (expected)

- [ ] **Grafana**: http://localhost:3011
  - Login: admin / admin
  - Should see Grafana home page
  - May prompt to change password (can skip for dev)

- [ ] **Prometheus**: http://localhost:9090
  - Should see Prometheus UI
  - Check Status → Targets (some may be down - OK for now)

---

## 🔍 Troubleshooting

### Service Won't Start

**Check logs**:
```bash
# All services
make logs

# Specific service
docker logs industrialmind-kafka
docker logs industrialmind-influxdb
docker logs industrialmind-mlflow
```

**Common issues**:

1. **Port already in use**
   - Solution: Stop conflicting service or change port in docker-compose.yml

2. **Insufficient memory**
   - Solution: Increase Docker memory allocation in Docker Desktop settings
   - Recommended: 8GB minimum

3. **PostgreSQL won't start**
   - Check: `docker logs industrialmind-postgres`
   - Solution: May need to remove volume and recreate
   ```bash
   make down
   docker volume rm industrialmind_postgres-data
   make up
   ```

4. **Kafka connection refused**
   - Wait 30 seconds after starting - Kafka takes time to initialize
   - Check: `docker logs industrialmind-kafka`

### Health Check Failures

If services show "unhealthy":

```bash
# Check specific service health
docker inspect industrialmind-kafka | grep -A 10 Health

# Restart specific service
docker restart industrialmind-kafka
```

### Clean Start

If you need to start fresh:

```bash
# WARNING: This removes all data
make clean

# Then start again
make up
make kafka-topics
```

---

## ✅ Success Criteria

Infrastructure is ready when:

- [x] All 9 services are running (Up/healthy)
- [ ] All ports are accessible
- [ ] All service UIs load correctly
- [ ] Kafka topics are created
- [ ] No error logs in any service
- [ ] InfluxDB "sensors" bucket exists
- [ ] PostgreSQL "industrialmind" and "mlflow" databases exist
- [ ] MLflow UI loads at http://localhost:5011
- [ ] Grafana UI loads at http://localhost:3011

---

## 📋 Post-Launch

Once infrastructure is running:

1. **Run System Test**
   ```bash
   # Write a test point to InfluxDB
   curl -X POST http://localhost:8086/api/v2/write?org=industrialmind&bucket=sensors \
     -H "Authorization: Token industrialmind-token-123456" \
     -H "Content-Type: text/plain" \
     --data-raw "sensor_readings,machine_id=TEST_001 temperature=25.5"

   # Query it back
   curl -X POST http://localhost:8086/api/v2/query?org=industrialmind \
     -H "Authorization: Token industrialmind-token-123456" \
     -H "Content-Type: application/vnd.flux" \
     --data 'from(bucket:"sensors") |> range(start: -1h) |> filter(fn: (r) => r["machine_id"] == "TEST_001")'
   ```

2. **Next Steps**: Begin Week 1 implementation
   - Implement data simulator
   - Implement data ingestion service
   - Create basic dashboard

---

## 🆘 Need Help?

**Check logs**: `make logs-f`
**Restart services**: `make restart`
**Stop services**: `make down`
**Clean restart**: `make clean && make up`

**Documentation**:
- [README.md](./README.md) - Getting started
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) - System architecture
- [Makefile](./Makefile) - Available commands

---

**Status**: Ready to Launch ✅
**Date**: 2026-01-12
**Configuration**: MLflow on :5011, Grafana on :3011, Azure cloud target
