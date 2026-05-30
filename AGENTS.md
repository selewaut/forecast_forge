# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python 3.13 package for dataset-agnostic forecasting workflows. Core code lives in `src/forecast_forge/`, including data loading, processing, model abstractions, registries, runnable scripts, and YAML configuration.

Exploratory work belongs in `notebooks/`. Spark container assets and local cluster commands are in `spark-setup/`. Local datasets may live under `data/`, but keep that directory untracked and replaceable.

## Build, Test, and Development Commands

Install dependencies with `uv`:

```sh
uv sync
```

Run tests with:

```sh
uv run pytest
```

Run the local Spark forecasting pipeline from the repository root:

```sh
make run
```

Download the local dataset from the repository root:

```sh
make data
```

Use `uv run` for ad hoc commands and local Spark execution, for example:

```sh
uv run spark-submit --master 'local[*]' src/forecast_forge/univariate_weekly.py
```

Start the Spark Docker environment from `spark-setup/`:

```sh
cd spark-setup
make run-d
make submit app=src/forecast_forge/univariate_weekly.py
```

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, clear module names, and snake_case for functions, variables, and files. Keep classes in PascalCase. Prefer small, testable functions over script-level logic. Avoid hard-coding dataset names, column names, or paths when they can be expressed through configuration.

No formatter or linter is currently configured. If adding one, document the command and avoid broad reformatting unrelated files in the same change.

## Development Workflow

Use test-driven development: write the test first, then implement the feature to make it pass. Tests define the contract; features satisfy it.

## Testing Guidelines

Pytest is the expected test framework, configured through `pyproject.toml`. Place tests under `tests/` using names like `test_data_processing.py` and `test_builds_time_series_features()`. Cover transformations, model registry behavior, and configuration parsing before larger Spark integration tests. Use small synthetic fixtures.

## Dataset-Agnostic Design

Treat any current sample dataset as an example, not a platform assumption. New loaders and feature builders should accept configurable schema mappings, timestamp columns, entity keys, target columns, and forecast horizons. Keep dataset-specific transformations behind config or adapter modules.

## Commit & Pull Request Guidelines

Recent history uses short imperative commits and occasional Conventional Commit prefixes, for example `feat: spark cluster configuration.`. Prefer concise messages in the form `type: summary` for feature work, fixes, and docs, such as `fix: handle missing holiday flags`.

Pull requests should include a clear description, validation commands, and any data or Spark setup assumptions. Include screenshots or notebook output only when they clarify analysis changes. Link related issues when applicable.

## Security & Configuration Tips

Do not commit credentials, local data extracts, generated Spark output, or secrets. Keep API keys and dataset access tokens outside the repository, and document required environment variables or local paths in README updates instead of hard-coding them.
