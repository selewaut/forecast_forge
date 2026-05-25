# Forecast Forge Docs

This directory tracks project context for planning, specifications, decisions, and implementation tasks.

## Structure

- `epics/`: larger initiatives that group related tasks and specs.
- `tasks/`: implementation-sized work items with status, scope, and validation notes.
- `ADRs/`: architecture decision records for decisions that should remain easy to review later.
- `templates/`: reusable templates for epics, tasks, requirements, technical specs, and ADRs.
- `mlflow.md`: local MLflow setup and run inspection guide for forecast jobs.

## Workflow

1. Create or update an epic when work spans multiple tasks.
2. Write tasks with concrete scope, acceptance criteria, and validation commands.
3. Record important technical decisions as ADRs.
4. Update user-facing setup docs when a task changes install, environment, Docker, or runtime behavior.
5. Keep task status current as implementation progresses.

## Documentation Rules

- Changes to Python dependencies or Python version should update `pyproject.toml`, `uv.lock`, and the related task document.
- Changes to local runtime setup should update `README.md` and the related task document.
- Changes to Docker or Spark cluster behavior should update `README.md`, `spark-setup/` docs or commands, and the related task document.
- Decisions with lasting tradeoffs should get an ADR.

Prefer small, direct documents over long planning artifacts. The goal is to preserve enough context that future agentic coding sessions can resume quickly.
