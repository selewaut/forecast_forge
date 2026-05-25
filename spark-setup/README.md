# Spark Docker Setup

This directory defines a local Spark standalone cluster for Forecast Forge.

## Runtime

- Python 3.13
- Spark 4.1.1
- Java 17
- Project dependencies installed with `uv sync --frozen --no-dev`

## Commands

Run commands from this directory:

```sh
cd spark-setup
make build
make run-d
make smoke
```

Submit a project script mounted from the repository root:

```sh
make submit app=src/forecast_forge/univariate_weekly.py
```

Stop and remove cluster containers and volumes:

```sh
make down
```

## Services

- Spark master UI: http://localhost:9090
- Spark master: `spark://spark-master:7077`
- Spark history server: http://localhost:18080

The repository root is mounted into the containers at `/opt/spark/apps`.
