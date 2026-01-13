# Skills Index - IndustrialMind

**Created**: 2026-01-12
**Status**: Months 1-4 Complete

This directory contains reusable patterns and templates for the IndustrialMind project. Each skill provides production-ready code templates that you can adapt to your specific needs.

---

## 📚 Skills by Category

### PyTorch Skills
Essential patterns for PyTorch model development (Month 2+)

| Skill | Purpose | Month | Complexity |
|-------|---------|-------|------------|
| [training_loop.md](./pytorch/training_loop.md) | Standard training loop with validation, early stopping, checkpointing | 2 | ⭐⭐⭐ |
| [custom_dataset.md](./pytorch/custom_dataset.md) | Custom Dataset classes for time series sensor data | 2 | ⭐⭐ |

**Key Features**:
- ✅ Training with validation split
- ✅ Early stopping to prevent overfitting
- ✅ Automatic checkpointing (latest + best)
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ Time series windowing
- ✅ Data normalization
- ✅ InfluxDB integration

---

### Data Processing Skills
Patterns for preprocessing and feature engineering (Month 1-2)

| Skill | Purpose | Month | Complexity |
|-------|---------|-------|------------|
| [time_series_preprocessing.md](./data_processing/time_series_preprocessing.md) | Complete time series preprocessing pipeline | 1-2 | ⭐⭐⭐ |

**Key Features**:
- ✅ Missing value handling
- ✅ Outlier detection and removal
- ✅ Resampling to regular intervals
- ✅ Normalization (standard, minmax, robust)
- ✅ Temporal feature engineering
- ✅ Rolling statistics
- ✅ Lag features

---

### MLOps Skills
Production ML operations patterns (Month 3+)

| Skill | Purpose | Month | Complexity |
|-------|---------|-------|------------|
| [mlflow_tracking.md](./mlops/mlflow_tracking.md) | Experiment tracking and model registry with MLflow | 3 | ⭐⭐⭐ |

**Key Features**:
- ✅ Experiment tracking
- ✅ Parameter and metric logging
- ✅ Artifact management
- ✅ Model registry integration
- ✅ Automatic git tracking
- ✅ Hyperparameter tuning patterns
- ✅ Model comparison utilities

---

### API Design Skills
FastAPI patterns for model serving (Month 2+)

| Skill | Purpose | Month | Complexity |
|-------|---------|-------|------------|
| [fastapi_ml_service.md](./api_design/fastapi_ml_service.md) | Complete FastAPI service for ML model inference | 2-3 | ⭐⭐⭐⭐ |

**Key Features**:
- ✅ Request/response validation with Pydantic
- ✅ Single and batch inference endpoints
- ✅ Health check endpoint
- ✅ Proper error handling
- ✅ Logging and monitoring
- ✅ OpenAPI documentation
- ✅ Performance optimization

---

### Testing Skills
Comprehensive testing patterns for ML (All Months)

| Skill | Purpose | Month | Complexity |
|-------|---------|-------|------------|
| [ml_model_testing.md](./testing/ml_model_testing.md) | Unit, integration, and performance tests for ML models | 1-12 | ⭐⭐⭐ |

**Key Features**:
- ✅ Model unit tests (forward pass, shape checking)
- ✅ Dataset tests
- ✅ Training pipeline tests
- ✅ Integration tests
- ✅ Performance benchmarks
- ✅ pytest fixtures and configuration

---

## 🚀 Quick Start

### Using a Skill

1. **Read the skill documentation** to understand when and how to use it
2. **Copy the template code** to your project
3. **Adapt to your specific use case**
4. **Run the examples** to verify it works

### Example: Using the Training Loop Skill

```python
# 1. Read Skills/pytorch/training_loop.md

# 2. Use the pattern in your code
from Skills.pytorch.training_loop import Trainer

trainer = Trainer(
    model=your_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    device='cuda'
)

history = trainer.train(epochs=100, early_stopping_patience=10)
```

---

## 📖 Skill Format

Each skill follows this structure:

```markdown
# Skill Name

## Purpose
What this skill helps accomplish

## When to Use
Situations where this pattern applies

## Prerequisites
What needs to exist before using this

## Template
Code template with placeholders

## Example Usage for IndustrialMind
Concrete examples with project context

## Variations
Common modifications or alternatives

## Common Pitfalls
Mistakes to avoid

## References
Links to documentation
```

---

## 🎯 Skills Roadmap

### ✅ Completed (Months 1-4)

**Month 1-2: Foundation**
- [x] Time series preprocessing
- [x] Custom PyTorch datasets
- [x] Training loops with validation

**Month 2-3: First Models & MLOps**
- [x] MLflow experiment tracking
- [x] FastAPI model serving
- [x] ML model testing

### 📋 Planned (Months 5-12)

**Month 5-6: Advanced ML**
- [ ] Graph neural network patterns (Neo4J + PyTorch Geometric)
- [ ] RAG system patterns (LangChain + ChromaDB)
- [ ] Vector database integration

**Month 7-8: LLM & Deployment**
- [ ] LLM fine-tuning (LoRA/QLoRA)
- [ ] Kubernetes deployment patterns
- [ ] Helm chart templates

**Month 9-10: Cloud & CI/CD**
- [ ] AWS SageMaker integration
- [ ] Terraform patterns
- [ ] GitHub Actions ML workflows

**Month 11-12: Production**
- [ ] Prometheus monitoring patterns
- [ ] Grafana dashboard templates
- [ ] Data drift detection
- [ ] Model performance monitoring

---

## 💡 Contributing Skills

As you develop the IndustrialMind project, extract reusable patterns into skills:

### When to Create a Skill

Create a skill when you:
- ✅ Solve a problem that might recur
- ✅ Find a pattern worth reusing
- ✅ Discover a best practice
- ✅ Build something portfolio-worthy

### Skill Quality Checklist

Good skills have:
- [ ] Clear purpose statement
- [ ] Working code template
- [ ] Project-specific examples
- [ ] Common pitfalls documented
- [ ] References to resources

---

## 🔍 Finding the Right Skill

### By Project Phase

**Week 1-4 (Data Pipeline)**
- `data_processing/time_series_preprocessing.md`

**Week 5-8 (First Model)**
- `pytorch/custom_dataset.md`
- `pytorch/training_loop.md`
- `testing/ml_model_testing.md`

**Week 9-12 (MLOps)**
- `mlops/mlflow_tracking.md`
- `api_design/fastapi_ml_service.md`

### By Task Type

**Training a model?**
→ `pytorch/training_loop.md`

**Loading time series data?**
→ `pytorch/custom_dataset.md`
→ `data_processing/time_series_preprocessing.md`

**Deploying a model?**
→ `api_design/fastapi_ml_service.md`

**Tracking experiments?**
→ `mlops/mlflow_tracking.md`

**Writing tests?**
→ `testing/ml_model_testing.md`

---

## 📊 Complexity Guide

| Level | Description | Example |
|-------|-------------|---------|
| ⭐ | Basic - Simple templates, minimal customization | Data loading |
| ⭐⭐ | Intermediate - Some configuration needed | Dataset classes |
| ⭐⭐⭐ | Advanced - Requires understanding of concepts | Training loops, MLflow |
| ⭐⭐⭐⭐ | Expert - Complex patterns, multiple components | FastAPI service |

---

## 🤝 Using Skills with Claude

When working with Claude Code, reference skills explicitly:

```
"Use the pytorch/training_loop skill to implement training for the
autoencoder model. Apply early stopping with patience=10."

"Following the mlflow_tracking skill, log this experiment with
hyperparameters and metrics."

"Create a FastAPI service using the api_design/fastapi_ml_service
pattern for the anomaly detection model."
```

This helps Claude understand exactly which patterns to apply.

---

## 📚 Additional Resources

### Learning Resources
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pytest Documentation](https://docs.pytest.org/)

### IndustrialMind Docs
- [Context Engineering Guide](../Knowledge/CONTEXT_ENGINEERING_GUIDE.md)
- [Project Objectives](../PROJECT_OBJECTIVES.md)
- [Organizational Tasks](../ORGANIZATIONAL_TASKS.md)

---

## 🎯 Quick Reference

| Need | Skill |
|------|-------|
| Train PyTorch model | `pytorch/training_loop.md` |
| Load sensor data | `pytorch/custom_dataset.md` |
| Clean time series | `data_processing/time_series_preprocessing.md` |
| Track experiments | `mlops/mlflow_tracking.md` |
| Serve model via API | `api_design/fastapi_ml_service.md` |
| Test ML code | `testing/ml_model_testing.md` |

---

**Remember**: Skills are starting points, not final solutions. Adapt them to your specific needs while maintaining the core patterns.

**Happy coding!** 🚀

---

*Last Updated: 2026-01-12*
*Skills Count: 6*
*Coverage: Months 1-4*
