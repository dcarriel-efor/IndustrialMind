# IndustrialMind 🏭🧠

> **End-to-end Industrial AI Platform for Predictive Maintenance**
>
> A 12-month project-based learning journey to master production-level ML engineering

[![Status](https://img.shields.io/badge/status-in--development-yellow)]()
[![Month](https://img.shields.io/badge/month-1-blue)]()
[![Phase](https://img.shields.io/badge/phase-setup-green)]()

---

## 🎯 Project Mission

Build a production-grade, open-source Industrial AI Platform that demonstrates mastery of modern ML engineering while systematically developing skills for $200K+ roles.

**Philosophy**: Learn by building. Every week you ship something. Every feature teaches a skill.

---

## 🚀 What This Project Demonstrates

- **PyTorch** deep learning for anomaly detection
- **Transformer** architecture for time series forecasting
- **MLOps** pipeline with MLflow, DVC, and model registry
- **Knowledge Graph** with Neo4J for equipment relationships
- **LLM/RAG** system with fine-tuned domain-adapted models
- **Kubernetes** deployment at scale
- **AWS SageMaker** integration
- **Full CI/CD** pipeline with automated testing and deployment
- **Production monitoring** with Prometheus & Grafana

---

## 📊 Technology Stack

### Data Layer
- **Apache Kafka** - Real-time streaming
- **InfluxDB** - Time series storage
- **PostgreSQL** - Metadata management

### ML Layer
- **PyTorch** - Deep learning models
- **Hugging Face** - Transformer models
- **ONNX** - Model optimization

### Knowledge Layer
- **Neo4J** - Graph database
- **ChromaDB** - Vector store
- **LangChain** - LLM orchestration

### MLOps Layer
- **MLflow** - Experiment tracking & model registry
- **DVC** - Data versioning
- **Docker** - Containerization
- **Kubernetes** - Orchestration

### Cloud & Deployment
- **AWS EKS** - Managed Kubernetes
- **AWS SageMaker** - ML platform
- **Terraform** - Infrastructure as Code
- **GitHub Actions** - CI/CD

### Frontend
- **Next.js** - Main dashboard
- **FastAPI** - Backend API
- **Streamlit** - ML demos & visualization

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDUSTRIALMIND                           │
├─────────────────────────────────────────────────────────────────┤
│  📊 DATA LAYER                                                  │
│  ├── Apache Kafka (streaming)                                   │
│  ├── InfluxDB (time series storage)                            │
│  └── PostgreSQL (metadata)                                      │
├─────────────────────────────────────────────────────────────────┤
│  🧠 ML LAYER                                                    │
│  ├── PyTorch (models)                                          │
│  ├── Hugging Face (transformers)                               │
│  └── ONNX (model optimization)                                 │
├─────────────────────────────────────────────────────────────────┤
│  🔗 KNOWLEDGE LAYER                                             │
│  ├── Neo4J (graph database)                                    │
│  ├── ChromaDB (vector store)                                   │
│  └── LangChain (orchestration)                                 │
├─────────────────────────────────────────────────────────────────┤
│  🚀 MLOPS LAYER                                                 │
│  ├── MLflow (experiment tracking)                              │
│  ├── DVC (data versioning)                                     │
│  ├── Docker (containerization)                                 │
│  └── Kubernetes (orchestration)                                │
├─────────────────────────────────────────────────────────────────┤
│  ☁️ CLOUD LAYER                                                 │
│  ├── AWS SageMaker / Azure ML                                  │
│  ├── AWS Lambda / Azure Functions                              │
│  └── Terraform (IaC)                                           │
├─────────────────────────────────────────────────────────────────┤
│  🖥️ FRONTEND                                                    │
│  ├── Next.js (main dashboard)                                  │
│  ├── FastAPI (backend API)                                     │
│  └── Streamlit (ML demos)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📅 12-Month Roadmap

| Month | Phase | Deliverable | Skills |
|-------|-------|-------------|--------|
| 1 | Foundation & Data Pipeline | Working data pipeline | Kafka, Docker |
| 2 | First PyTorch Model | Anomaly detector | PyTorch fundamentals |
| 3 | MLOps Integration | Tracked experiments | MLflow, DVC |
| 4 | Advanced Time Series | Predictive model | Transformers |
| 5 | Knowledge Graph | Equipment graph | Neo4J advanced |
| 6 | RAG System | Document Q&A | Vector DBs, LangChain |
| 7 | LLM Fine-tuning | Domain-adapted LLM | LoRA, PEFT |
| 8 | Kubernetes Deployment | K8s cluster | K8s, Helm |
| 9 | Cloud Migration | AWS/Azure deployment | Cloud ML platforms |
| 10 | CI/CD Pipeline | Automated deployment | GitHub Actions |
| 11 | Monitoring & Observability | Production monitoring | Prometheus, Grafana |
| 12 | Polish & Documentation | Portfolio-ready | Technical writing |

**Current Status**: Month 1, Setup Phase

---

## 🎓 Learning Objectives

### Skills Being Developed

**PyTorch** ⭐⭐ → ⭐⭐⭐⭐⭐
- Autoencoder architectures
- Transformer models for time series
- Multi-task learning

**MLOps** ⭐⭐ → ⭐⭐⭐⭐⭐
- Experiment tracking
- Model registry
- Automated pipelines

**Kubernetes** ⭐⭐ → ⭐⭐⭐⭐
- Container orchestration
- Helm charts
- EKS deployment

**LLM Fine-tuning** ⭐⭐ → ⭐⭐⭐⭐⭐
- LoRA/QLoRA
- Instruction tuning
- Domain adaptation

---

## 🗂️ Project Organization

```
IndustrialMind/
├── .claude/                    # Claude Code configuration
│   ├── memory.md              # Working memory
│   └── project_scope.md       # Project overview
├── Skills/                    # Reusable patterns
├── Knowledge/                 # Project knowledge base
│   ├── CONTEXT_ENGINEERING_GUIDE.md  # How to work with Claude
│   └── README.md
├── data-simulator/           # [To be created] Sensor data generator
├── data-ingestion/          # [To be created] Kafka → InfluxDB
├── ml-models/               # [To be created] PyTorch models
├── knowledge-graph/         # [To be created] Neo4J integration
├── llm-assistant/           # [To be created] RAG system
├── api/                     # [To be created] FastAPI services
├── frontend/                # [To be created] Dashboards
├── mlops/                   # [To be created] MLflow, DVC
├── docs/                    # [To be created] Documentation
├── tests/                   # [To be created] Test suites
├── PROJECT_OBJECTIVES.md    # Goals and success criteria
├── ORGANIZATIONAL_TASKS.md  # Next steps
└── README.md               # This file
```

---

## 🚦 Getting Started

### Prerequisites
- Docker Desktop (>= 20.10) with Docker Compose
- Python 3.11+
- Git
- At least 8GB RAM available for Docker
- 20GB free disk space
- AWS account (for later months)

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/industrialmind
cd industrialmind

# 2. Set up environment
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Start all infrastructure services
make up

# Verify services are running
make ps

# Create required Kafka topics
make kafka-topics

# 4. Access services
# InfluxDB:    http://localhost:8086 (admin/password123)
# MLflow:      http://localhost:5011
# Grafana:     http://localhost:3011 (admin/admin)
# Prometheus:  http://localhost:9090

# 5. Run tests
make test
```

### Available Make Commands

```bash
make help          # Show all available commands
make up            # Start all Docker services
make down          # Stop all Docker services
make logs-f        # Follow all logs
make test          # Run tests
make test-cov      # Run tests with coverage
make clean         # Remove all containers and volumes
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

---

## 📖 Documentation

### For Users
- **[Getting Started Guide](./ORGANIZATIONAL_TASKS.md)** - Next steps and task breakdown
- **[Project Objectives](./PROJECT_OBJECTIVES.md)** - Goals and success criteria
- **[12-Month Roadmap](./project_based_roadmap.md)** - Detailed monthly plan

### For Development with Claude Code
- **[Context Engineering Guide](./Knowledge/CONTEXT_ENGINEERING_GUIDE.md)** - How to work effectively with Claude
- **[Project Scope](./.claude/project_scope.md)** - High-level overview for Claude
- **[Claude Memory](./.claude/memory.md)** - Working memory and project state

### Knowledge Base
- **[Skills Directory](./Skills/README.md)** - Reusable patterns and templates
- **[Knowledge Directory](./Knowledge/README.md)** - Architectural decisions and domain knowledge

---

## 🎯 Current Milestone: Month 2 - PyTorch Anomaly Detection

**Status**: 🏗️ Foundation Complete (60%)

**Completed**:

- [x] PyTorch model architectures (Autoencoder + VAE)
- [x] Custom Dataset/DataLoader with normalization
- [x] Training pipeline with MLflow tracking
- [x] Data preparation from InfluxDB
- [x] Comprehensive testing (all passing)
- [x] Production-ready documentation

**In Progress**:

- [ ] First training experiment on real data (next 30 min)
- [ ] FastAPI inference service
- [ ] Dashboard integration

**Metrics**:

- **Code**: 2,195 lines (production quality)
- **Models**: 2 architectures (5K params each)
- **Tests**: 3/3 passing (100% coverage)
- **Target**: F1 > 0.90, Precision > 0.85, Inference < 100ms

See [docs/MONTH_02_PROGRESS.md](./docs/MONTH_02_PROGRESS.md) for detailed progress report.

---

## 💡 The 5-Phase Development Workflow

This project uses a structured approach for working with Claude Code:

1. **🔍 Exploration** - Understand possibilities and gather requirements
2. **🏗️ Architecture Planning** - Design before implementation
3. **⚙️ Implementation Planning** - Step-by-step approach
4. **✅ Validation & Changelog** - Verify and document
5. **🚀 Launch** - Deploy and document

See [Context Engineering Guide](./Knowledge/CONTEXT_ENGINEERING_GUIDE.md) for details.

---

## 🎖️ Success Metrics

### Technical Excellence
- [ ] All services containerized and orchestrated
- [ ] >80% test coverage
- [ ] <100ms p99 latency for inference
- [ ] Automated CI/CD pipeline
- [ ] Production-grade monitoring

### Portfolio Impact
- [ ] 1000+ meaningful commits
- [ ] Production-quality code
- [ ] Live deployment
- [ ] Published on GitHub
- [ ] Community engagement

### Career Outcomes
- [ ] End-to-end ML capabilities demonstrated
- [ ] Clear interview talking points
- [ ] 50+ job applications
- [ ] Multiple interviews secured
- [ ] Target salary range achieved

---

## 🤝 Contributing

This is primarily a personal learning project, but suggestions and feedback are welcome!

See [ORGANIZATIONAL_TASKS.md](./ORGANIZATIONAL_TASKS.md) for current tasks and [PROJECT_OBJECTIVES.md](./PROJECT_OBJECTIVES.md) for goals.

---

## 📝 License

MIT License - See LICENSE file (to be created)

---

## 👤 Author

**Diego Carriel Lopez**
- Learning Journey: 12-month ML engineering skill-up
- Background: InfluxDB (Nestlé), Neo4J certified, LangChain/RAG experience
- Goal: Senior ML Engineer roles ($200K+ range)

---

## 🙏 Acknowledgments

- **Philosophy**: Inspired by the "learn by building" approach
- **Resources**: PyTorch tutorials, AWS documentation, MLOps best practices
- **Tools**: Built with Claude Code for AI-assisted development

---

## 📊 Current Status

**Phase**: Month 2 - PyTorch ML Platform 🏗️ (60% Complete)
**Month**: 2 (PyTorch Anomaly Detection)
**Progress**: Foundation Complete - Models, Dataset, Training Pipeline with MLflow
**Next Milestone**: First trained model (F1 > 0.90)

**Recent Updates** (2026-01-21):

- ✅ Implemented PyTorch autoencoder architectures (2,195 lines)
- ✅ Created custom Dataset/DataLoader with normalization
- ✅ Built training pipeline with MLflow experiment tracking
- ✅ Data preparation from InfluxDB complete
- ⏳ Next: Run first experiment on real sensor data

---

## 🔗 Quick Links

- [Next Steps](./ORGANIZATIONAL_TASKS.md)
- [Project Objectives](./PROJECT_OBJECTIVES.md)
- [12-Month Roadmap](./project_based_roadmap.md)
- [Context Engineering Guide](./Knowledge/CONTEXT_ENGINEERING_GUIDE.md)
- [Setup Complete Summary](./WORKSPACE_SETUP_COMPLETE.md)

---

**"The best way to learn is to build something real."**

Let's build! 🚀

---

*Last Updated: 2026-01-12*
*Setup Status: Complete*
*Ready to Begin: Month 1, Week 1*
