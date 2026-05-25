# Task 0004: Update Developer Documentation

## Status

Done

## Epic

[Epic 0001: Local UV, Python, and Spark Modernization](../epics/0001-uv-python-spark-modernization.md)

## Objective

Refresh developer-facing setup and run instructions after the UV and Spark modernization work is complete.

## Scope

- Update README installation instructions to use `uv`.
- Document local Python/PySpark workflow.
- Document Docker Spark cluster workflow as the current expected path, with runtime validation deferred to the Docker epic.
- Replace Java 8 instructions with Java 17 instructions.
- Clarify macOS, Linux, and Windows/WSL setup paths.
- Document validation commands.
- Ensure docs reflect the project-local `.envrc` and `direnv` setup.
- Ensure task docs record implementation results for Tasks 0001, 0002, and 0005, with Docker follow-up tracked separately.

## Acceptance Criteria

- README has current setup instructions.
- README has local and Docker Spark execution examples.
- Old `pip install -r requirements.txt`-first workflow is removed or clearly marked legacy.
- `docs/README.md` explains when implementation tasks must update setup docs.

## Validation

```sh
make help
make check
uv run spark-submit --master 'local[*]' src/forecast_forge/univariate_weekly.py --help
```

## Result

- README documents `uv`, Python 3.13, Java 17, `direnv`, local PySpark execution, Kaggle access-token authentication, model selection, and Docker Spark commands.
- MLflow setup and usage are documented in `docs/mlflow.md`.
- Docker runtime validation is intentionally tracked in the separate Docker epic.
