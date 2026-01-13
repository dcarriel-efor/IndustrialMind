# Context Engineering Guide for IndustrialMind

## Purpose
This guide establishes the structured prompting workflow for collaborating with Claude Code on the IndustrialMind project. It ensures consistent, high-quality interactions that maximize learning and productivity.

## The 5-Phase Prompting Workflow

### Phase 1: Exploration
**Objective**: Understand possibilities and gather requirements

**What to do:**
- Present the problem or feature goal
- Ask Claude to explore multiple approaches
- Request pros/cons analysis of different solutions
- Identify dependencies and prerequisites

**Example Prompts:**
```
"I need to build a real-time anomaly detection system for sensor data.
Explore different ML approaches (autoencoder, isolation forest, LSTM)
and recommend the best option for our use case considering:
- Real-time performance requirements
- PyTorch implementation
- Interpretability for technicians"
```

**Expected Outputs:**
- Multiple solution approaches
- Comparative analysis
- Recommended approach with justification
- Identified risks and considerations

---

### Phase 2: Architecture Planning
**Objective**: Design the system architecture before implementation

**What to do:**
- Define component interactions
- Specify data flows
- Identify integration points
- Plan for scalability and testing

**Example Prompts:**
```
"Based on the autoencoder approach, create an architectural plan for:
1. Model architecture (layers, dimensions)
2. Training pipeline (data loading, validation, checkpointing)
3. Inference service (API design, model loading)
4. Integration with Kafka and InfluxDB

Include file structure, key classes, and data flow diagrams."
```

**Expected Outputs:**
- System architecture diagram (in text/markdown)
- File/folder structure
- Class/module specifications
- Data flow descriptions
- API contracts

---

### Phase 3: Implementation Planning
**Objective**: Create step-by-step implementation plan

**What to do:**
- Break down into concrete tasks
- Sequence tasks appropriately
- Identify test requirements
- Plan for incremental delivery

**Example Prompts:**
```
"Create a step-by-step implementation plan for the anomaly detection model:
1. Break down into atomic tasks
2. Specify order of implementation
3. Define completion criteria for each step
4. Include testing strategy

Start with minimal viable implementation, then iterate."
```

**Expected Outputs:**
- Ordered task list
- Acceptance criteria per task
- Testing approach
- Implementation timeline estimate
- Risk mitigation steps

---

### Phase 4: Validation & Changelog
**Objective**: Verify implementations and maintain project history

**What to do:**
- Test each implementation
- Document what was built
- Record decisions made
- Track changes and learnings

**Example Prompts:**
```
"We've completed the autoencoder model. Help me:
1. Create comprehensive tests (unit + integration)
2. Validate against our acceptance criteria
3. Document the implementation in CHANGELOG.md
4. Identify any technical debt or improvements needed"
```

**Expected Outputs:**
- Test suite
- Validation report
- Changelog entry
- Technical debt backlog
- Performance metrics

---

### Phase 5: Launch
**Objective**: Deploy the solution and prepare documentation

**What to do:**
- Prepare deployment artifacts
- Create runbooks
- Write user documentation
- Plan rollback strategy

**Example Prompts:**
```
"We're ready to deploy the anomaly detection service. Help me:
1. Create Docker container and K8s manifests
2. Write deployment runbook
3. Set up monitoring and alerts
4. Create API documentation
5. Plan staged rollout strategy"
```

**Expected Outputs:**
- Deployment configurations
- Operational runbooks
- User/API documentation
- Monitoring setup
- Rollback procedures

---

## Context Engineering Best Practices

### 1. Provide Rich Context
Always include:
- **Current state**: What exists now
- **Goal**: What you want to achieve
- **Constraints**: Limitations (budget, time, tech stack)
- **Success criteria**: How you'll know it's done

### 2. Reference Project Files
```
"Looking at the data simulator in data-simulator/simulator.py:42-67,
I want to add realistic degradation patterns. The current implementation
only handles normal operation."
```

### 3. Be Specific About Scope
Bad: "Help me build the ML model"
Good: "Help me implement the PyTorch autoencoder's training loop with validation, early stopping, and MLflow logging. The model architecture is already defined in ml-models/anomaly_detector/model.py."

### 4. Iterate Incrementally
- Start with minimal viable solution
- Get it working end-to-end
- Then add complexity/features
- Avoid over-engineering early

### 5. Request Learning Explanations
```
"Implement the attention mechanism for the time series model,
and explain why multi-head attention is beneficial for our use case
with industrial sensor data."
```

### 6. Validate Assumptions
```
"Before we implement the graph neural network, verify these assumptions:
1. Neo4J supports the required queries at scale
2. We can efficiently batch node embeddings
3. The graph structure allows for meaningful aggregation"
```

---

## Prompt Templates by Task Type

### Feature Development
```
**Context**: [What exists now]
**Goal**: [What feature to add]
**Approach**: [Preferred/suggested approach]
**Constraints**: [Technical/business constraints]

**Request**:
1. [Specific ask 1]
2. [Specific ask 2]
3. [Specific ask 3]

**Success Criteria**:
- [Criterion 1]
- [Criterion 2]
```

### Debugging
```
**Problem**: [Description of the issue]
**Expected**: [What should happen]
**Actual**: [What is happening]
**Code Location**: [file:line_number]
**Error Message**: [If applicable]

**Request**: Help me debug this and explain the root cause.
```

### Architecture Decision
```
**Decision**: [What needs to be decided]
**Options**: [Option A, Option B, Option C]
**Criteria**: [How to evaluate options]

**Request**:
Analyze each option and recommend the best approach for our use case.
Consider: performance, maintainability, learning value, portfolio impact.
```

### Code Review
```
**Code**: [Reference to files/lines]
**Purpose**: [What it's supposed to do]

**Request**:
Review for:
1. Correctness and edge cases
2. Performance issues
3. Code quality and maintainability
4. Testing gaps
5. Security concerns
```

---

## Managing Claude's Context

### Keep Context Fresh
- Reference recent files explicitly
- Summarize multi-turn conversations periodically
- Reset context when switching major topics

### Use Project Files as Memory
- Store decisions in Knowledge/ folder
- Reference decisions when they're relevant
- Update context files as project evolves

### Progressive Disclosure
- Don't dump all information at once
- Provide context as it becomes relevant
- Build up complexity gradually

---

## Quality Checklist for Prompts

Before submitting a prompt, verify:
- [ ] Clear objective stated
- [ ] Relevant context provided
- [ ] Specific asks enumerated
- [ ] Success criteria defined
- [ ] Appropriate phase of workflow
- [ ] References to existing code/docs
- [ ] Constraints mentioned

---

## Example: Full Workflow for a Feature

### 1. Exploration Prompt
```
I need to add a real-time dashboard to visualize anomaly scores
from our PyTorch model. Explore options for:
- Frontend framework (Streamlit vs React)
- Real-time data updates (WebSockets vs polling)
- Visualization library (Plotly vs Chart.js)

Consider: ease of implementation, portfolio impact, integration with FastAPI
```

### 2. Architecture Prompt
```
Based on Streamlit + WebSockets + Plotly, design the architecture for:
1. Streamlit app structure and components
2. WebSocket connection to FastAPI backend
3. Real-time data flow from Kafka to frontend
4. State management for dashboard

Include file structure and key implementation details.
```

### 3. Implementation Prompt
```
Create step-by-step implementation plan for the dashboard:
1. Set up Streamlit app with WebSocket client
2. Create FastAPI WebSocket endpoint
3. Connect to Kafka consumer for anomaly scores
4. Implement Plotly real-time charts
5. Add error handling and reconnection logic

Provide code structure and testing approach for each step.
```

### 4. Validation Prompt
```
We've implemented the real-time dashboard. Help me:
1. Create integration tests for WebSocket communication
2. Test real-time updates under load
3. Validate against acceptance criteria
4. Document in CHANGELOG.md
5. Identify performance optimizations
```

### 5. Launch Prompt
```
Deploy the dashboard:
1. Create Docker container for Streamlit app
2. Update docker-compose.yml and K8s manifests
3. Write operational runbook
4. Create user guide
5. Set up monitoring for WebSocket connections
```

---

## Anti-Patterns to Avoid

### ❌ Vague Requests
"Make the model better"
"Fix the bug"
"Improve performance"

### ❌ Skipping Architecture
Jumping straight to code without planning

### ❌ No Success Criteria
Not defining what "done" looks like

### ❌ Context Overload
Dumping entire files when only a few lines are relevant

### ❌ Premature Optimization
Asking for advanced features before basic functionality works

---

## Measuring Prompt Effectiveness

Good prompts result in:
- ✅ Actionable, concrete outputs
- ✅ Minimal back-and-forth clarification
- ✅ Code that works first try (or close)
- ✅ Learning alongside implementation
- ✅ Clear next steps

If you're not getting these, refine your prompts using this guide.

---

*Remember: Clear prompts lead to clear implementations. Invest time in context engineering for better results.*

*Version 1.0 | Last Updated: 2026-01-12*
