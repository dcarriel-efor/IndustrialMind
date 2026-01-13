# Skills Directory

## Purpose
This directory contains reusable skills, patterns, and templates that Claude Code can reference when working on the IndustrialMind project. Think of these as "skill cards" or "playbooks" for common tasks.

## Structure
```
Skills/
├── pytorch/              # PyTorch-specific patterns
├── mlops/               # MLOps workflows and templates
├── kubernetes/          # K8s configurations and patterns
├── api_design/          # FastAPI and API patterns
├── testing/             # Testing strategies and templates
├── monitoring/          # Observability patterns
└── documentation/       # Documentation templates
```

## How to Use

### For Users
When asking Claude to perform a task, you can reference skills:
```
"Use the pytorch/training_loop pattern to implement training for the new model"
"Apply the mlops/experiment_tracking template for this experiment"
```

### For Claude
Skills provide:
- Proven patterns and best practices
- Code templates with placeholders
- Configuration examples
- Testing approaches
- Common pitfalls to avoid

## Skill Categories

### 1. PyTorch Skills
- Custom Dataset/DataLoader patterns
- Training loop templates
- Model evaluation frameworks
- Checkpoint management
- Multi-GPU training setup

### 2. MLOps Skills
- MLflow experiment tracking
- DVC pipeline configurations
- Model registry workflows
- A/B testing patterns
- Feature store integration

### 3. Kubernetes Skills
- Deployment manifests
- Service configurations
- Helm chart templates
- ConfigMap and Secret management
- HPA (Horizontal Pod Autoscaling) setup

### 4. API Design Skills
- FastAPI application structure
- Request/response validation
- Authentication/authorization patterns
- Error handling
- API versioning

### 5. Testing Skills
- Unit test templates
- Integration test patterns
- Model testing strategies
- Performance testing
- Mock data generation

### 6. Monitoring Skills
- Prometheus metrics definition
- Grafana dashboard configurations
- Alert rule templates
- Logging patterns
- Distributed tracing

### 7. Documentation Skills
- README templates
- API documentation (OpenAPI)
- Runbook templates
- Architecture decision records (ADRs)
- Changelog formats

## Adding New Skills

When you develop a reusable pattern:
1. Extract the generic pattern
2. Create a template with clear placeholders
3. Add usage examples
4. Document assumptions and prerequisites
5. Include common variations

## Skill Template Format

Each skill should include:
```markdown
# [Skill Name]

## Purpose
[What this skill helps accomplish]

## When to Use
[Situations where this pattern applies]

## Prerequisites
[What needs to exist before using this]

## Template
[Code or configuration template]

## Example Usage
[Concrete example with project context]

## Variations
[Common modifications or alternatives]

## Pitfalls
[Common mistakes to avoid]

## References
[Links to documentation or further reading]
```

---

*This directory will be populated as we develop the IndustrialMind project and identify reusable patterns.*

*Version 1.0 | Last Updated: 2026-01-12*
