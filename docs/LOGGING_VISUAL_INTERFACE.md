# Logging Visual Interface - Current State & Roadmap

**Last Updated**: 2026-01-19
**Current Status**: ❌ No visual interface (command line only)
**Planned**: ✅ Grafana + Loki integration (Phase 2)

---

## Current State: Command Line Access

### Available Now

Logs are currently accessible **only via Docker CLI**:

```bash
# Basic log viewing
docker compose logs dashboard                    # All logs
docker compose logs dashboard -f                 # Follow mode (real-time)
docker compose logs dashboard --tail 100         # Last 100 lines
docker compose logs dashboard --since 5m         # Last 5 minutes

# Filtering logs
docker compose logs dashboard | grep error       # Find errors
docker compose logs dashboard | grep MACHINE_001 # Specific machine

# Multiple services
docker compose logs dashboard simulator ingestion  # View multiple services
```

### Limitations

❌ **No visual interface**
❌ **No historical search**
❌ **No aggregation or analytics**
❌ **No alerting**
❌ **Manual command-line queries only**

---

## Quick Visual Solution: Grafana (Already Running!)

**Good news**: Grafana is already running at http://localhost:3011

You can access it now, but it's currently only monitoring infrastructure metrics (via Prometheus), not application logs.

### Current Grafana Setup

```
URL: http://localhost:3011
Username: admin
Password: admin (change on first login)

Currently Connected:
✅ Prometheus (infrastructure metrics)
❌ Loki (logs) - NOT YET CONFIGURED
```

---

## Phase 2 Implementation: Add Visual Log Interface

### Architecture After Phase 2

```
┌─────────────────────────────────────────────────────────────┐
│                  Application Logs (JSON)                     │
│  Dashboard, Simulator, Ingestion Services                    │
└────────────────────────┬────────────────────────────────────┘
                         │ stdout (Docker logs)
                         │
                         v
                  ┌──────────────┐
                  │   Promtail   │  ← Log collector
                  │              │  (reads Docker logs)
                  └──────┬───────┘
                         │
                         │ Ships logs to
                         │
                         v
                  ┌──────────────┐
                  │     Loki     │  ← Log aggregation database
                  │              │  (stores & indexes logs)
                  └──────┬───────┘
                         │
                         │ Query logs
                         │
                         v
                  ┌──────────────┐
                  │   Grafana    │  ← Visual interface
                  │              │  (already running!)
                  │  Port: 3011  │
                  └──────────────┘
```

### What You'll Be Able to Do

After Phase 2 implementation:

✅ **Visual Log Viewer** - Browse logs in Grafana UI
✅ **Search & Filter** - Full-text search across all logs
✅ **Time-based Queries** - "Show me logs from last hour"
✅ **Service Filtering** - View logs by service (dashboard, simulator, etc.)
✅ **Error Tracking** - Dedicated error log dashboard
✅ **Alerting** - Get notified when errors spike
✅ **Historical Analysis** - Query old logs (7-day retention)
✅ **Performance Metrics** - Query timing charts from logs

### Example Grafana Log Queries (After Phase 2)

```logql
# All dashboard logs
{service="dashboard"}

# Only errors
{service="dashboard"} |= "error"

# Specific machine queries
{service="dashboard"} | json | machine_id="MACHINE_001"

# Slow queries (hypothetical - if we log timing)
{service="dashboard"} | json | duration > 100ms

# Error rate over time
rate({service="dashboard"} |= "error" [5m])
```

---

## Quick Setup Guide: Enable Visual Logs (Phase 2)

### Step 1: Add Loki to docker-compose.yml

Add this service:

```yaml
loki:
  image: grafana/loki:2.9.0
  container_name: industrialmind-loki
  ports:
    - "3100:3100"
  command: -config.file=/etc/loki/local-config.yaml
  volumes:
    - ./infrastructure/loki/loki-config.yml:/etc/loki/local-config.yaml
    - loki-data:/loki
  networks:
    - default
  healthcheck:
    test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:3100/ready"]
    interval: 10s
    timeout: 5s
    retries: 5

promtail:
  image: grafana/promtail:2.9.0
  container_name: industrialmind-promtail
  volumes:
    - ./infrastructure/promtail/promtail-config.yml:/etc/promtail/config.yml
    - /var/lib/docker/containers:/var/lib/docker/containers:ro
    - /var/run/docker.sock:/var/run/docker.sock
  command: -config.file=/etc/promtail/config.yml
  depends_on:
    - loki
  networks:
    - default
```

### Step 2: Create Loki Configuration

File: `infrastructure/loki/loki-config.yml`

```yaml
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

limits_config:
  retention_period: 168h  # 7 days

table_manager:
  retention_deletes_enabled: true
  retention_period: 168h
```

### Step 3: Create Promtail Configuration

File: `infrastructure/promtail/promtail-config.yml`

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'logstream'
      - source_labels: ['__meta_docker_container_label_com_docker_compose_service']
        target_label: 'service'
```

### Step 4: Configure Grafana Data Source

1. Go to http://localhost:3011
2. Login (admin/admin)
3. Go to Configuration → Data Sources
4. Click "Add data source"
5. Select "Loki"
6. Set URL: `http://loki:3100`
7. Click "Save & Test"

### Step 5: Start Everything

```bash
# Start Loki and Promtail
docker compose up -d loki promtail

# Verify they're running
docker compose ps loki promtail

# Check Loki is receiving logs
docker compose logs promtail --tail 20
```

### Step 6: View Logs in Grafana

1. Go to Grafana: http://localhost:3011
2. Click "Explore" (compass icon in left sidebar)
3. Select "Loki" from data source dropdown
4. Query examples:
   - `{service="dashboard"}` - All dashboard logs
   - `{service="dashboard"} |= "error"` - Only errors
   - `{service="dashboard"} | json | machine_id="MACHINE_001"` - Specific machine

---

## Alternative: Quick Docker Desktop Logs Viewer

If you're using **Docker Desktop**, you already have a basic visual log viewer:

1. Open Docker Desktop
2. Click on "Containers" in the left sidebar
3. Find `industrialmind-dashboard` container
4. Click on it
5. Go to "Logs" tab
6. You'll see logs with basic filtering

**Limitations**:
- ❌ No advanced search
- ❌ No cross-service queries
- ❌ No historical analysis
- ❌ No alerting

---

## Temporary Solution: Log to File + Tail

While waiting for Grafana/Loki, you can create a continuous log viewer:

### Option 1: Simple Tail Window

```bash
# In a separate terminal, keep this running:
docker compose logs dashboard -f | tee dashboard.log

# Now you have:
# - Real-time logs in terminal
# - Historical logs in dashboard.log file
# - Can search the file: grep error dashboard.log
```

### Option 2: Filter for JSON Only

```bash
# Only see structured logs (not Streamlit internal logs)
docker compose logs dashboard -f | grep '^{'
```

### Option 3: Pretty-print JSON Logs (requires jq)

```bash
# Install jq first (if not installed):
# Windows: choco install jq
# Mac: brew install jq
# Linux: apt-get install jq

# Then run:
docker compose logs dashboard -f | grep '^{' | jq '.'
```

This will format JSON logs nicely:
```json
{
  "event": "machines_queried",
  "count": 5,
  "machines": [
    "MACHINE_001",
    "MACHINE_002",
    "MACHINE_003",
    "MACHINE_004",
    "MACHINE_005"
  ],
  "timestamp": "2026-01-19T10:42:20.963744Z",
  "level": "info"
}
```

---

## Comparison: Current vs. Phase 2

| Feature | Current (CLI) | Phase 2 (Grafana + Loki) |
|---------|---------------|--------------------------|
| **Access Method** | Command line | Web UI (Grafana) |
| **Real-time Logs** | ✅ `docker compose logs -f` | ✅ Live tail in Grafana |
| **Search Logs** | ❌ Manual grep | ✅ Full-text search UI |
| **Filter by Service** | ❌ Manual | ✅ Dropdown selection |
| **Time Range Queries** | ❌ Limited (`--since`) | ✅ Visual time picker |
| **Error Tracking** | ❌ Manual grep | ✅ Dedicated error dashboard |
| **Alerting** | ❌ None | ✅ Grafana alerts |
| **Historical Search** | ❌ Container restart loses logs | ✅ 7-day retention |
| **Cross-Service Queries** | ❌ Complex manual work | ✅ Single query across all services |
| **Charts & Analytics** | ❌ None | ✅ Log rate charts, error % |
| **Export Logs** | ❌ Manual copy-paste | ✅ CSV/JSON export |

---

## Recommended Next Steps

### Option 1: Continue with CLI (Simple, Already Working)

**Pros**:
- ✅ Already implemented
- ✅ No additional infrastructure
- ✅ Fast and lightweight

**Cons**:
- ❌ No visual interface
- ❌ Manual queries only
- ❌ No historical analysis

**Good for**: Small projects, development, quick debugging

---

### Option 2: Implement Phase 2 (Professional Setup)

**Pros**:
- ✅ Professional log management
- ✅ Visual interface
- ✅ Historical analysis
- ✅ Alerting capability
- ✅ Production-ready

**Cons**:
- ❌ Additional infrastructure (Loki, Promtail)
- ❌ ~1-2 hours setup time
- ❌ More resources (~500MB RAM)

**Good for**: Production systems, team environments, long-term monitoring

**Estimated Time**:
- Loki/Promtail setup: 1 hour
- Grafana configuration: 30 minutes
- Create dashboards: 30 minutes
- **Total: ~2 hours**

---

## Summary

**Current State**:
```
✅ Structured logging implemented (Phase 1 complete)
✅ Logs available via: docker compose logs dashboard
❌ No visual interface yet
❌ Manual CLI queries only
```

**To Get Visual Interface**:
```
Option A: Wait for Phase 2 implementation (~2 hours work)
Option B: Use Docker Desktop logs tab (basic viewing)
Option C: Use temporary tail + grep solution (current best option)
```

**Recommended**:
If you need logs frequently, **implement Phase 2 now** (Grafana + Loki).
If you only occasionally check logs, **continue with CLI** for now.

---

## Quick Reference: Essential Log Commands

```bash
# Most useful commands for current CLI access:

# Real-time errors only
docker compose logs dashboard -f | grep error

# Real-time JSON logs only (clean output)
docker compose logs dashboard -f | grep '^{'

# Recent errors (last 10 minutes)
docker compose logs dashboard --since 10m | grep error

# Specific machine queries
docker compose logs dashboard | grep "MACHINE_001"

# Count errors
docker compose logs dashboard | grep error | wc -l

# View all services logs together
docker compose logs -f dashboard simulator ingestion
```

---

**Need help implementing Phase 2?** Let me know and I can guide you through the Loki + Grafana setup step by step!
