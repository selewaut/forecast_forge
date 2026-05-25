# Task 0003: Modernize Docker Spark Setup

## Status

In progress

## Epic

[Epic 0001: UV, Python, and Spark Modernization](../epics/0001-uv-python-spark-modernization.md)

## Objective

Update the Docker-based Spark cluster to align with Python 3.13, modern Spark, Java 17, and the new `uv` project setup.

## Scope

- Update Docker base image to Python 3.13.
- Upgrade Spark from 3.3.3 to a Python 3.13-compatible Spark 4.x line.
- Upgrade Java from 11 to 17.
- Fix Dockerfile copy paths for dependencies and entrypoint scripts.
- Decide whether Docker installs dependencies through `uv sync` or an exported requirements file.
- Update `docker compose` service definitions and Makefile commands.
- Add a Spark smoke-test target.
- Update README Docker instructions and this task with validation results.

## Implementation Notes

- Updated Docker base image to `python:3.13-slim-bookworm`.
- Updated Spark default build argument to `SPARK_VERSION=4.1.1`.
- Installed `openjdk-17-jdk` in the image and exposed it through a stable `/opt/java/openjdk` symlink for cross-architecture Debian images.
- Changed Compose build context to the repository root so Docker can copy `pyproject.toml`, `uv.lock`, `README.md`, `LICENSE`, `.python-version`, and `src/`.
- Installed Python dependencies inside the image with `uv sync --frozen --no-dev`.
- Mounted the repository root at `/opt/spark/apps` for job submission.
- Updated Makefile commands to use `docker compose`.
- Added `make smoke` using Spark's bundled Python `pi.py` example.
- Added `.dockerignore` to keep local data, virtualenvs, git metadata, and outputs out of the build context.

## Validation

Docker is not installed or not on `PATH` in the current environment, so build/run validation could not be executed here.

Static and local validation performed:

```sh
bash -n spark-setup/entrypoint.sh
make -n build
make -n submit app=src/forecast_forge/univariate_weekly.py
make -n smoke
direnv exec . uv run spark-submit --version
```

Expected validation commands:

```sh
cd spark-setup
make build
make run-d
make smoke
make submit app=src/forecast_forge/univariate_weekly.py
make down
```

## Acceptance Criteria

- `docker compose build` succeeds from `spark-setup/`.
- Spark master, worker, and history server start successfully.
- A smoke-test Spark job succeeds.
- `make submit app=...` works with the documented path convention.
- README documents the final Docker workflow.
