# AI Engineering Assessment

Two independent projects: a local agent harness for operations work, and a machine-learning prototype for aircraft MRO unscheduled removal prediction.

## Projects

### [Operations Assistant Agent Harness](task-1-agent-harness/)

A local, single-process agent harness with a FastAPI API, a terminal REPL, an explicit ReAct loop against an OpenAI-compatible model, SQLite persistence, approval-gated incident creation, structured logs, and multi-turn conversations. Includes mock tools, a Postman collection, automated tests, and a multi-turn conversation example.

### [Aircraft MRO Unscheduled Removal Prediction](task-2-mro-prediction/)

A machine-learning prototype that predicts whether a component will experience an unscheduled removal within the next 30 flight cycles. Includes dataset generation and validation, temporal and entity-based splits, model training and evaluation, frozen thresholds, reports, and automated tests.

## Repository Layout

```text
task-1-agent-harness/    Agent harness source, tests, docs, data, Postman, examples
task-2-mro-prediction/   MRO prediction source, configs, tests, reports, models
```

## Getting Started

Each project is self-contained and has its own README with setup, configuration, commands, tests, and limitations:

- `task-1-agent-harness/README.md`
- `task-2-mro-prediction/README.md`
