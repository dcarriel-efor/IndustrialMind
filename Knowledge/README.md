# Knowledge Directory

## Purpose
This directory serves as the central knowledge base for the IndustrialMind project. It contains documentation, architectural decisions, domain knowledge, and reference materials that provide context for development.

## Structure
```
Knowledge/
├── architecture/        # System architecture documentation
├── decisions/          # Architecture Decision Records (ADRs)
├── domain/            # Industrial/manufacturing domain knowledge
├── research/          # Research notes and paper summaries
├── reference/         # Quick reference guides
└── lessons_learned/   # Post-implementation insights
```

## Knowledge Categories

### 1. Architecture
- **System Overview**: High-level architecture diagrams
- **Component Design**: Detailed component specifications
- **Data Flow**: How data moves through the system
- **Integration Points**: External system interfaces
- **Technology Stack**: Detailed tech stack documentation

### 2. Architectural Decisions
- **ADR Format**: Lightweight decision records
- **Decision History**: Evolution of architectural choices
- **Trade-offs**: Analysis of options considered
- **Reversible vs Irreversible**: Decision classification

### 3. Domain Knowledge
- **Industrial Sensors**: Types, characteristics, failure modes
- **Predictive Maintenance**: Industry best practices
- **Manufacturing Processes**: Context for our use cases
- **Anomaly Patterns**: What normal vs abnormal looks like
- **Equipment Relationships**: Dependencies and hierarchies

### 4. Research
- **Paper Summaries**: Key ML/AI papers relevant to project
- **Technique Exploration**: New methods to try
- **Benchmark Results**: Performance comparisons
- **State of the Art**: Current best practices

### 5. Reference Materials
- **PyTorch Patterns**: Common PyTorch idioms
- **MLOps Best Practices**: Industry standards
- **Kubernetes Cheat Sheets**: Quick command reference
- **API Conventions**: Naming and structure standards
- **Error Code Catalog**: Standard error codes and meanings

### 6. Lessons Learned
- **What Worked Well**: Successful patterns and approaches
- **What Didn't Work**: Failures and why
- **Performance Insights**: Optimization learnings
- **Bug Patterns**: Common bugs and solutions
- **Productivity Tips**: Workflow improvements

## How Claude Uses This

### Context Building
Claude references Knowledge/ files to:
- Understand project context and history
- Apply consistent patterns and conventions
- Make informed architectural decisions
- Avoid repeating past mistakes
- Build on previous learnings

### Decision Making
When faced with design choices, Claude can:
- Check existing ADRs for precedent
- Review domain knowledge for context
- Reference research for best practices
- Consider lessons learned from similar tasks

### Code Generation
Knowledge informs:
- Naming conventions
- Error handling patterns
- Testing approaches
- Documentation style
- Integration approaches

## Maintaining Knowledge

### When to Add Knowledge
- After making significant architectural decisions
- When learning domain-specific information
- After researching a new technique
- When discovering a useful pattern
- After completing a challenging implementation

### Knowledge Quality
Good knowledge entries are:
- ✅ Concise but complete
- ✅ Well-structured with clear headings
- ✅ Include examples where helpful
- ✅ Dated for historical context
- ✅ Tagged for easy discovery

### Knowledge Lifecycle
1. **Capture**: Document insights as they occur
2. **Organize**: File in appropriate category
3. **Reference**: Link to from relevant code/docs
4. **Update**: Keep current as project evolves
5. **Prune**: Remove obsolete information

## Knowledge vs Skills vs Memory

### Knowledge (this folder)
**What**: Domain information, decisions, research
**Purpose**: Provide context for understanding
**Example**: "Why we chose PyTorch over TensorFlow"

### Skills (../Skills)
**What**: Reusable code patterns and templates
**Purpose**: Accelerate implementation
**Example**: "PyTorch training loop template"

### Memory (.claude/memory.md)
**What**: Current project state and preferences
**Purpose**: Maintain continuity across sessions
**Example**: "Current phase: Month 2, Week 7"

## Quick Start Guide

### For Users
Before starting a new feature:
1. Check architecture/ for relevant system design
2. Review decisions/ for past choices
3. Read domain/ for business context
4. Look at lessons_learned/ for pitfalls

### For Claude
When working on a task:
1. Reference relevant knowledge files
2. Apply documented patterns
3. Respect architectural decisions
4. Learn from lessons learned

## Knowledge Templates

### Architecture Document
```markdown
# [Component Name] Architecture

## Overview
[High-level description]

## Responsibilities
[What this component does]

## Interfaces
[How it interacts with other components]

## Data Model
[Key data structures]

## Technology
[Tech stack choices]

## Considerations
[Design trade-offs and constraints]
```

### ADR (Architecture Decision Record)
```markdown
# ADR-XXX: [Decision Title]

**Status**: [Proposed | Accepted | Deprecated]
**Date**: YYYY-MM-DD
**Deciders**: [Who was involved]

## Context
[What necessitated this decision]

## Decision
[What we decided to do]

## Rationale
[Why we made this decision]

## Consequences
[Implications, both positive and negative]

## Alternatives Considered
[Other options and why they weren't chosen]
```

### Lesson Learned
```markdown
# [Topic]: [Brief Title]

**Date**: YYYY-MM-DD
**Context**: [What were we trying to do]

## What Happened
[Description of the situation]

## What We Learned
[Key insights]

## What to Do Differently
[Actionable guidance for next time]

## Related
[Links to code, ADRs, or other knowledge]
```

---

## Current Knowledge Status

### Populated
- [x] Context Engineering Guide
- [x] Project Scope
- [x] Project Objectives

### To Be Created (as project progresses)
- [ ] System Architecture Overview
- [ ] Industrial Sensor Domain Knowledge
- [ ] ADR: Why PyTorch Over TensorFlow
- [ ] ADR: Kafka vs RabbitMQ for Streaming
- [ ] PyTorch Best Practices Reference
- [ ] MLflow Integration Patterns

---

*This knowledge base grows with the project. Document insights as you discover them.*

*Version 1.0 | Last Updated: 2026-01-12*
